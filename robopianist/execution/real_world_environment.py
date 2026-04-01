import time
from typing import Union, Any

import dm_env
import numpy as np
from dm_env import Environment, StepType
from dm_env._environment import TimeStep
from dm_env.specs import BoundedArray
from dm_env_wrappers._src.base import EnvironmentWrapper
from mujoco_utils import spec_utils

from configs import Defaults
from robopianist.execution.robot_apis import Robot, PianoAPI, HandAPI, ArmAPI, NotesAPI
from robopianist.models.hands import HandSide
from robopianist.wrappers.evaluation import compute_key_press_metrics, EpisodeMetrics


class RealWorldEnvironment(Environment):
    def __init__(self, args: Defaults, robot: Robot):
        self.args = args
        self.piano = PianoAPI()
        self.hand = HandAPI(robot)
        self.arm = ArmAPI(robot, args.physics_timestep)
        self.notes = NotesAPI(args)
        self.control_timestep = args.control_timestep
        self.hardware_order = [self.notes, self.piano, self.hand, self.arm]

    def reset(self) -> TimeStep:
        step = TimeStep(observation=self._observe(), step_type=StepType.FIRST, reward=0.0, discount=1.0)
        self.notes.reset_index()
        return step

    def close(self):
        # Reversed order because:
        # 1. The arm is more important to close immediately
        # 2. The piano takes up to 1s to close
        for component in reversed(self.hardware_order):
            component.close()

    def step(self, action) -> TimeStep:
        # Act
        self.hand.take_actions(action["hand"])
        self.arm.take_actions(action["arm"])

        # Wait for action to take effect
        time.sleep(self.control_timestep)

        # Observe
        observations = self._observe()
        if self.notes.was_first:
            step_type = StepType.FIRST
        elif self.notes.was_last:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID

        return TimeStep(observation=observations, step_type=step_type, reward=0.0, discount=1.0)

    def _observe(self):
        observation_parts = {x.spec.name: x.observe() for x in self.hardware_order}
        return observation_parts

    def observation_spec(self):
        return {x.spec.name: x.spec for x in self.hardware_order}

    def action_spec(self):
        return {x.spec.name: x.spec for x in [self.hand, self.arm]}


class RWEvaluationWrapper(EnvironmentWrapper):
    """
    Wrapper that computes musical metrics for the real-world environment.
    """

    def __init__(self, env: Union[RealWorldEnvironment, dm_env.Environment]):
        super().__init__(env)
        self.notes, _ = NotesAPI.load_song(env.args)
        self.keypresses = []

    def reset(self) -> dm_env.TimeStep:
        self.keypresses = []
        return self._environment.reset()

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        activation = timestep.observation['piano']
        self.keypresses.append(activation)
        return timestep

    def get_musical_metrics(self) -> EpisodeMetrics:
        return compute_key_press_metrics(note_seq=self.notes,
                                         n_keys=49,
                                         key_presses=self.keypresses)


class RemoveExcludedHandActionsWrapper(EnvironmentWrapper):
    """
    Wrapper that removes unused joints from observations and adds default actions based on args.excluded_dofs.
    """

    def __init__(self, env: Union[RealWorldEnvironment, dm_env.Environment]):
        super().__init__(env)
        actuator_names = ["ffa0", "ffa1", "ffa2", "ffa3",
                          "mfa0", "mfa1", "mfa2", "mfa3",
                          "rfa0", "rfa1", "rfa2", "rfa3",
                          "tha0", "tha1", "tha2", "tha3"]
        self.used_finger_ids = []

        # fill used_finger_ids only if they are not excluded
        for i, name in enumerate(actuator_names):
            if name not in env.args.excluded_dofs:
                self.used_finger_ids.append(i)

    def step(self, action) -> dm_env.TimeStep:
        full_action = self._add_unused_actions(action)

        timestep = self._environment.step(full_action)

        return self._remove_unused_observations(timestep)

    def _add_unused_actions(self, action: dict[str, np.ndarray]):
        defaults = HandAPI.get_defaults()
        defaults[self.used_finger_ids] = action["hand"]

        action["hand"] = defaults
        return action

    def _remove_unused_observations(self, timestep: dm_env.TimeStep):
        filtered_observations = timestep.observation["hand"][self.used_finger_ids]

        modified = timestep.observation.copy()
        modified["hand"] = filtered_observations

        return TimeStep(observation=modified, step_type=timestep.step_type, reward=0.0, discount=1.0)

    def _modify_spec(self, old_spec: dict[str, BoundedArray]):
        new_hand_spec = BoundedArray(
            shape=(len(self.used_finger_ids),),
            dtype=old_spec["hand"].dtype,
            minimum=old_spec["hand"].minimum[self.used_finger_ids],
            maximum=old_spec["hand"].maximum[self.used_finger_ids],
            name="hand",
        )

        spec = old_spec.copy()
        spec["hand"] = new_hand_spec
        return spec

    def observation_spec(self):
        old_spec = self._environment.observation_spec()
        return self._modify_spec(old_spec)

    def action_spec(self):
        old_spec = self._environment.action_spec()
        return self._modify_spec(old_spec)

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return self._remove_unused_observations(timestep)


class SplitActionsWrapper(EnvironmentWrapper):
    """
    Wrapper that splits a flat action array into separate hand and arm actions.
    """

    def step(self, action) -> dm_env.TimeStep:
        new_action = {
            "hand": action[:-1],
            "arm": action[-1:],
        }
        return self._environment.step(new_action)

    def action_spec(self):
        old_spec = self._environment.action_spec()
        return spec_utils.merge_specs(list(old_spec.values()))


class ChangeOrderForLeftSideWrapper(EnvironmentWrapper):
    """
    Wrapper that switches the first and ring finger if a left-side hand is used.
    """

    def __init__(self, env: Union[RealWorldEnvironment, dm_env.Environment]):
        super().__init__(env)
        if env.args.hand_side == HandSide.LEFT:
            self.new_order = np.array([8, 9, 10, 11,  # First finger
                                       4, 5, 6, 7,  # Middle finger
                                       0, 1, 2, 3,  # Ring finger
                                       12, 13, 14, 15]  # Thumb
                                      )
        else:
            self.new_order = np.array([0, 1, 2, 3,  # First finger
                                       4, 5, 6, 7,  # Middle finger
                                       8, 9, 10, 11,  # Ring finger
                                       12, 13, 14, 15]  # Thumb
                                      )

    def step(self, action) -> dm_env.TimeStep:
        reordered_action = action.copy()
        reordered_action["hand"] = action["hand"][self.new_order]

        timestep = self._environment.step(reordered_action)

        reordered_observation = timestep.observation.copy()
        reordered_observation["hand"] = timestep.observation["hand"][self.new_order]

        return TimeStep(observation=reordered_observation,
                        step_type=timestep.step_type,
                        reward=0.0,
                        discount=1.0)

    def _reorder_spec(self, spec: BoundedArray):
        return BoundedArray(
            shape=spec.shape,
            dtype=spec.dtype,
            minimum=spec.minimum[self.new_order],
            maximum=spec.maximum[self.new_order],
            name=spec.name,
        )

    def _reorder_dict(self, specs: dict[str, BoundedArray]):
        return {**specs, "hand": self._reorder_spec(specs["hand"])}

    def action_spec(self):
        return self._reorder_dict(self._environment.action_spec())

    def observation_spec(self):
        return self._reorder_dict(self._environment.observation_spec())


class RenameObservationsWrapper(EnvironmentWrapper):
    """
    The simulation uses different naming conventions compared to the real world environment.
    To reuse the wrappers we need to rename the observations.
    """

    def __init__(self, env: Union[RealWorldEnvironment, dm_env.Environment]):
        super().__init__(env)
        self.new_names = [
            ("goals", "goal"),
            ("piano", "piano/activation"),
            (["hand", "arm"], "rh_allegro_hand/joints_pos"),
        ]

    @staticmethod
    def _merge(values: list[Union[np.ndarray, BoundedArray]]) -> Any:
        if isinstance(values[0], np.ndarray):
            return np.concatenate(values, axis=0)
        return spec_utils.merge_specs(values)

    def _rename_keys(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        renamed = {}
        for old, new in self.new_names:
            if isinstance(old, list):
                renamed[new] = self._merge([input_dict[k] for k in old])
                continue
            renamed[new] = input_dict[old]
        return renamed

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return TimeStep(observation=self._rename_keys(timestep.observation),
                        step_type=timestep.step_type,
                        reward=0.0, discount=1.0)

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        return TimeStep(observation=self._rename_keys(timestep.observation),
                        step_type=timestep.step_type,
                        reward=0.0, discount=1.0)

    def observation_spec(self):
        return self._rename_keys(self._environment.observation_spec())


if __name__ == "__main__":
    RealWorldEnvironment(Defaults(), Robot.DummyRobot)
