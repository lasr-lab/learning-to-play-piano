# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piano with shadow hands environment."""
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from absl import app
from dm_control.mjcf import export_with_assets
from dm_env_wrappers import ActionNoiseWrapper
from dm_env_wrappers import CanonicalSpecWrapper
from mujoco import viewer as mujoco_viewer

import configs
from robopianist import suite, viewer
from robopianist.execution.custom_environment_wrapper import PositioningConfig, PositionModes, \
    ObservationWrapper
from robopianist.execution.pd_controller import PDController
from robopianist.execution.policies import *
from robopianist.execution.real_world_test import RobotAPI, Robot
from robopianist.wrappers import PianoSoundVideoWrapper
from robopianist.wrappers.evaluation import compute_key_press_metrics


@dataclass(frozen=False)
class TestingEnv(configs.RealisticEnvConfig):
    # Two Hands
    # environment_name = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    # One Hand
    # environment_name = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    midi_file: str = None
    cube: bool = False
    canonicalize: bool = True  # Turn off for slider, on for model policy
    record: bool = False
    export: bool = False
    headless: bool = True
    action_noise: float = 0.0
    observe_real_fingers: bool = True
    observe_real_forearm: bool = True
    observe_real_piano: bool = True
    model_url: str = "https://api.wandb.ai/files/path/to/the/model/models/1000000.mpk"
    action_config: PositioningConfig = PositioningConfig(fingerMode=PositionModes.COPY_ACTIONS,
                                                         forearmMode=PositionModes.COPY_ACTIONS)
    robot: Robot = Robot.WallRobot
    song: str = "TwinkleTwinkleLittleStar"
    shift_factor: int = -15
    intensity: float = 0.0


@dataclass(frozen=False)
class DoNothing(TestingEnv):
    midi_file: str = Path(os.getcwd()) / "long.mid"
    trim_silence: bool = False


class ModelExecutor:
    def __init__(self, args=DoNothing(), result_dict=None, policy: type = None, quiet=False):
        self.result_dict = result_dict
        self.robot = RobotAPI(args, song=args.song, robot=args.robot)
        self.env = self.load_env(args, quiet)

        if not quiet:
            print(f"Control frequency: {1 / args.control_timestep} Hz")

        if not policy:
            policy = ModelPolicy
        self.policy = policy(self.env, args=args)

        self.mirror_forearm = args.action_config.forearmMode == PositionModes.MIRROR_SIMULATION
        self.mirror_finger = args.action_config.fingerMode == PositionModes.MIRROR_SIMULATION
        self.configuration = args

    def execute(self, evaluate: bool = True) -> Optional[Tuple[list[np.ndarray], list[dict[str, np.ndarray]]]]:
        self.robot.init_servo_mode()
        self.is_running = True
        if self.mirror_forearm or self.mirror_finger:
            mirroring_thread = threading.Thread(target=self.mirror_simulation, args=(), daemon=True)
            mirroring_thread.start()

        out = None
        try:
            if self.configuration.headless:
                out = self.run_headless(evaluate)
            else:
                viewer.launch(self.env, policy=self.policy)
        finally:
            self.is_running = False
            self.robot.go_back_home()
            self.robot.close_connections()
        return out

    def load_env(self, args, quiet) -> dm_env.Environment:
        env = suite.load(
            environment_name=args.environment_name,
            midi_file=args.midi_file,
            stretch=args.stretch_factor,
            shift=args.shift_factor,
            task_kwargs=dict(
                change_color_on_activation=True,
                trim_silence=args.trim_silence,
                control_timestep=args.control_timestep,
                gravity_compensation=args.gravity_compensation,
                primitive_fingertip_collisions=args.primitive_fingertip_collisions,
                reduced_action_space=args.reduced_action_space,
                n_steps_lookahead=args.n_steps_lookahead,
                disable_colorization=args.disable_colorization,
                disable_hand_collisions=args.disable_hand_collisions,
                attachment_yaw=args.attachment_yaw,
                hand_class=args.get_hand_class(),
                forearm_dofs=args.forearm_dofs,
                randomization_config=args.randomization_config,
                energy_penalty_coef=args.energy_penalty_coef,
                action_penalty_coef=args.action_penalty_coef,
                sliding_penalty_coef=args.sliding_penalty_coef,
                reward_functions=args.reward_functions,
                use_both_hands=args.use_both_hands,
                piano_type=args.get_piano_type(),
                excluded_dofs=args.excluded_dofs,
                physics_timestep=args.physics_timestep,
            ),
            cube=args.cube
        )

        if args.export:
            export_with_assets(
                env.task.root_entity.mjcf_model,
                out_dir="/tmp/robopianist/piano_with_shadow_hands",
                out_file_name="scene.xml",
            )
            mujoco_viewer.launch_from_path(
                "/tmp/robopianist/piano_with_shadow_hands/scene.xml"
            )
            return

        if args.record:
            env = PianoSoundVideoWrapper(env, record_every=1)
        if args.action_noise:
            env = ActionNoiseWrapper(env, scale=args.action_noise)
        if args.canonicalize and not args.headless:
            env = CanonicalSpecWrapper(env, clip=args.clip)
        if not quiet:
            if args.headless:
                print("headless")
            else:
                print("window mode")
        env = ObservationWrapper(env)

        if not quiet:
            print(f"Action dimension: {env.action_spec().shape}")

        # Sanity check observables.
        timestep = env.reset()

        if not quiet:
            dim = 0
            for k, v in timestep.observation.items():
                print(f"\t{k}: {v.shape} {v.dtype}")
                dim += int(np.prod(v.shape))

            print(f"Observation dimension: {dim}")

        return env

    def mirror_simulation(self):
        # rate = rospy.Rate(100)
        while self.is_running:
            if self.mirror_forearm:
                tx = self.env.get_forearm_tx()
                self.robot.update_forearm_position(tx=tx, is_scaled=True)
            if self.mirror_finger:
                joints = self.env.get_finger_joints(ordered=False)
                self.robot.set_finger_joints(joints, ignore_used_finger_ids=False)
            time.sleep(0.01)
            # rate.sleep()

    def run_headless(self, evaluate: bool = True) -> Tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
        args = self.configuration
        self.policy.reset()
        # arm_controller = PDController(kp=2000, kd=71.742, time_step=args.physics_timestep)
        arm_controller = PDController(kp=8, kd=4.53736277220056, time_step=args.physics_timestep, mass=1)
        substeps = int(args.control_timestep // args.physics_timestep)
        # rate = rospy.Rate(1.0 / args.physics_timestep)
        start = time.time()
        positions = []
        key_presses = []

        all_observations = []
        all_actions = []

        # run
        if self.result_dict is not None:
            print(self.result_dict["name"])

        for goal in self.robot.get_goals():
            for i in range(substeps):
                self.env.physics.step()
                if args.action_config.forearmMode == PositionModes.COPY_ACTIONS:
                    arm_controller.step()
                    self.robot.update_forearm_position(tx=arm_controller.actual_position, is_scaled=True)
                time_left = (0.00 + args.physics_timestep) - (time.time() - start)
                if time_left > 0 and not args.robot == Robot.DummyRobot:
                    time.sleep(time_left)
                start = time.time()

            # Observe
            key_presses.append(self.robot.get_activation())
            joint_positions = np.concatenate([
                self.robot.get_finger_joints() if args.observe_real_fingers else self.env.get_finger_joints(),
                self.robot.get_forearm_tx() if args.observe_real_forearm else self.env.get_forearm_tx()])
            observations = {
                "goal": goal,
                "piano/activation": key_presses[-1]
                if args.observe_real_piano
                else self.env.get_activation().astype(np.float64),
                "lh_allegro_hand/joints_pos": joint_positions,
            }
            all_observations.append(joint_positions)

            # Actions
            action = self.policy(observations=observations)
            self.env.physics.data.ctrl[self.robot.used_finger_ids] = action[:-1]
            self.env.physics.data.ctrl[-1] = action[-1]
            all_actions.append(action)

            # Apply to robot
            if args.action_config.fingerMode == PositionModes.AVERAGE_ACTIONS:
                positions = positions[-5:] + [action[:-1]]
                self.robot.set_finger_joints(np.average(positions, axis=0))
            elif args.action_config.fingerMode == PositionModes.COPY_ACTIONS:
                self.robot.set_finger_joints(action[:-1])
            if args.action_config.forearmMode == PositionModes.COPY_ACTIONS:
                arm_controller.set_target_position(action[-1])

        # evaluate
        if evaluate:
            result = compute_key_press_metrics(note_seq=self.robot.notes,
                                               n_keys=49,
                                               key_presses=key_presses)
            result_str = f"------ Evaluation ------\nprecision: {result.precision}\nrecall: {result.recall}\nf1: {result.f1}"
            print(result_str)
            if self.result_dict is not None:
                self.result_dict["f1"] = result.f1
                self.result_dict["precision"] = result.precision
                self.result_dict["recall"] = result.recall
        self.env.reset()
        return all_actions, all_observations


def main(_, args=TestingEnv()) -> None:
    executor = ModelExecutor(args, policy=ModelPolicy)
    executor.execute()


if __name__ == "__main__":
    app.run(main)
