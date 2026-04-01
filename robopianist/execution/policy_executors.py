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
from dataclasses import dataclass
from pathlib import Path
from typing import Type

from dm_env_wrappers._src.canonical_spec import CanonicalSpecWrapper
from dm_env_wrappers._src.concatenate_observations import ConcatObservationWrapper

from configs import Defaults, get_config
from robopianist import viewer
from robopianist.execution.policies import *
from robopianist.execution.real_world_environment import RealWorldEnvironment, RWEvaluationWrapper, \
    RemoveExcludedHandActionsWrapper, SplitActionsWrapper, ChangeOrderForLeftSideWrapper, RenameObservationsWrapper
from robopianist.execution.robot_apis import Robot
from robopianist.suite import PianoWithHands
from robopianist.wrappers.evaluation import EpisodeMetrics
from robopianist.wrappers.observation_stacker import ObservationStackerWrapper
from robopianist.wrappers.recurring_nodes import RecurringNodesWrapper
from train import Trainer


@dataclass(frozen=True)
class RunDetails:
    observations: Optional[list]
    actions: Optional[list]
    scores: EpisodeMetrics

    def __str__(self):
        return f"Returned: observations, actions and {self.scores}"

    def save_actions(self, path: Union[Path, str] = "/tmp/robopianist/actions.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.actions)
        df.to_csv(path, index=False)

    def save_observations(self, path: Union[Path, str] = "/tmp/robopianist/observations.csv"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.observations)
        df.to_csv(path, index=False)


class PolicyExecutor(ABC):
    def __init__(self, policy: Type[Policy], args: Defaults, robot: Optional[Robot] = None, **policy_args):
        self.args = args
        self.robot = robot
        self._env = self.load_env()
        self.policy = policy(self._env, **policy_args)

    @property
    def env(self):
        return self._env

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def load_env(self):
        pass

    def _create_trajectory(self) -> RunDetails:
        env = self._env
        policy = self.policy
        policy.reset()
        timestep = env.reset()
        observations = []
        actions = []
        while not timestep.last():
            action = policy(timestep)
            actions.append(action)

            timestep = env.step(action)
            observations.append(timestep.observation)

        return RunDetails(observations=observations, actions=actions, scores=env.get_musical_metrics())


class SimulatedPolicyExecutor(PolicyExecutor, ABC):
    def load_env(self):
        env, task = Trainer.get_env(self.args, evaluate=True)
        env.args = self.args
        return env

    @property
    def task(self) -> PianoWithHands:
        return self.env.task


class WindowedPolicyExecutor(SimulatedPolicyExecutor):
    def execute(self) -> RunDetails:
        env = self.load_env()
        viewer.launch(env, policy=self.policy)

        scores = EpisodeMetrics(**env.get_musical_metrics())
        return RunDetails(observations=None, actions=None, scores=scores)


class HeadlessPolicyExecutor(SimulatedPolicyExecutor):
    def execute(self) -> RunDetails:
        return self._create_trajectory()


class RealWorldPolicyExecutor(PolicyExecutor):
    def load_env(self):
        env = RealWorldEnvironment(self.args, self.robot)
        env = ChangeOrderForLeftSideWrapper(env)
        env = RWEvaluationWrapper(env)
        env = RemoveExcludedHandActionsWrapper(env)
        env = SplitActionsWrapper(env)
        env = RenameObservationsWrapper(env)
        if self.args.historical_stack_size > 1:
            env = ObservationStackerWrapper(env, ["rh_allegro_hand/joints_pos"], stacks=self.args.historical_stack_size)
        env = ConcatObservationWrapper(env)
        if self.args.n_recurring_nodes > 0:
            env = RecurringNodesWrapper(env, self.args.n_recurring_nodes)
        env = CanonicalSpecWrapper(env, clip=self.args.clip)
        return env

    def execute(self) -> RunDetails:
        try:
            return self._create_trajectory()
        finally:
            self.env.close()


if __name__ == "__main__":
    # TODO: Before running replace root.default.geom.conaffinity = 1 with root.default.geom.conaffinity = 0 to disable collisions with the piano
    args = get_config()
    args.historical_stack_size = 1
    # args.control_timestep = 1
    # args.environment_name = "RoboPianist-debug-long.mid-v0"
    args.gravity_compensation = False
    # args.stretch_factor = 100.0

    args.excluded_dofs = (
        "ffa0",  # base
        "ffa1",  # first
        # "ffa2", # second
        "ffa3",  # third
        "mfa0",
        "mfa1",
        # "mfa2",
        "mfa3",
        "rfa0",
        "rfa1",
        # "rfa2",
        "rfa3",
        "tha0",
        "tha1",
        "tha2",
        "tha3"
    )
    args.forearm_dofs = ()

    model_url = "https://api.wandb.ai/files/iamtypingsomethingrandom/robopianist/ieltvuqm/models/0970000.mpk"
    # path = Path("data") / "all_joints"
    path = Path("data") / "one_at_a_time" / "second"
    executor = RealWorldPolicyExecutor(policy=JumpPolicy, args=args, robot=Robot.OnlyHand)
    metrics = executor.execute()
    metrics.save_actions(path / "actions.csv")
    metrics.save_observations(path / "observations_rw.csv")
    print(metrics)
