import json
import threading
import time
from math import log, exp
from pathlib import Path
from typing import Union, Any

import numpy as np
import pandas as pd
import wandb
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from dm_control.rl.control import PhysicsError
from numpy._typing import NDArray

from configs import Defaults, get_config
from robopianist.execution.policies import ActionFilePolicy
from robopianist.execution.policy_executors import HeadlessPolicyExecutor, RunDetails
from robopianist.models.hands import AllegroHand
from robopianist.models.hands import allegro_hand_constants as consts

CALCULATED = {
    "damping-base": 0.138063872057794,
    "kp-base": 2.8229025988256984,
    "kd-base": 0.050774907306925195,
    "sluggish-base": 0.9097949314240089,
    "max_u-base": 0.6826666996111753,

    "damping-first": 0.10050661186538688,
    "kp-first": 2.686667079916346,
    "kd-first": 0.07610389459841495,
    "sluggish-first": 0.9499266776648303,
    "max_u-first": 1.4373962660961466,

    "damping-second": 0.10061229036839336,
    "kp-second": 2.7261047584377733,
    "kd-second": 0.02959827407640562,
    "sluggish-second": 0.9312686402848721,
    "max_u-second": 0.751814300239831,

    "damping-third": 0.15591532754333393,
    "kp-third": 2.8275032380243212,
    "kd-third": 0.06657578010288571,
    "sluggish-third": 0.9308860172957093,
    "max_u-third": 0.7770288460903263,

    "damping-arm": 8.63313117308672,
    "kp-arm": 9.964950811868137,
    "kd-arm": 8.517193191416238,
    "sluggish-arm": -0.17736741963576946,
    "max_u-arm": 11.132956644313257
}


class PanningDomainReductionTransformer(SequentialDomainReductionTransformer):
    def _trim(self, new_bounds: NDArray[np.floating[Any]], global_bounds: NDArray[np.floating[Any]]) -> NDArray[
        np.floating[Any]]:
        new_bounds = np.sort(new_bounds, axis=1)

        # enforce minimum windows size
        pbounds: NDArray[np.floating[Any]]
        for i, pbounds in enumerate(new_bounds):
            width = abs(pbounds[1] - pbounds[0])
            if width < self.minimum_window[i]:
                half = self.minimum_window[i] / 2
                center = pbounds.mean()
                pbounds[0] = center - half
                pbounds[1] = center + half

        return new_bounds


class ParameterConverter:
    def __init__(self, group_actuator_mapping: dict[str, list[str]], known_params: dict[str, float] = None):
        actuators = [
            "ffa0", "ffa1", "ffa2", "ffa3",
            "mfa0", "mfa1", "mfa2", "mfa3",
            "rfa0", "rfa1", "rfa2", "rfa3",
            "forearm_tx"
        ]
        for group in group_actuator_mapping.values():
            for actuator in group:
                if actuator not in actuators:
                    raise ValueError(f"Actuator '{actuator}' is not a valid actuator name.")
        self.group_actuator_mapping = group_actuator_mapping
        self._known_params = {} if known_params is None else known_params

    @property
    def known_params(self) -> dict[str, float]:
        return self._known_params

    def get_bounds(self, param_bounds: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        bounds = {}
        for group in self.group_actuator_mapping:
            for param, bound in param_bounds.items():
                param = f"{param}-{group}"
                if param in self.known_params:
                    continue
                bounds[param] = bound
        return bounds

    @staticmethod
    def _get_actuator_index(hand: AllegroHand, actuator: str) -> int:
        actuators = [a.name for a in hand.actuators]
        return actuators.index(actuator)

    @classmethod
    def _apply_controller_param(cls, parameter: str, hand: AllegroHand, actuator: str, value: float):
        index = cls._get_actuator_index(hand, actuator)
        assert parameter in hand.controller.__dict__, f"Parameter '{parameter}' not found in controller."
        if parameter == "sluggish":
            value = max(0.0, value)
            value = min(1.0, value)
        hand.controller.__dict__[parameter][index] = value

    @classmethod
    def _apply_damping(cls, hand: AllegroHand, actuator: str, value: float):
        index = cls._get_actuator_index(hand, actuator)
        hand.joints[index].damping = value

    def apply_parameter(self, hand: AllegroHand, name: str, value):
        parameter, group = name.split("-", 1)
        if group not in self.group_actuator_mapping:
            raise ValueError(f"Group '{group}' is not recognized.")

        for actuator in self.group_actuator_mapping[group]:
            if parameter == "damping":
                self._apply_damping(hand, actuator, value)
            else:
                self._apply_controller_param(parameter, hand, actuator, value)


class Flag:
    def __init__(self):
        self._flag = threading.Event()

    def set(self):
        self._flag.set()

    def is_set(self) -> bool:
        return self._flag.is_set()


class ParameterAutoTuner:
    def __init__(self, action_file_path: Path, target_observations_path: Path, ignore_first_n_timesteps: int,
                 args: Defaults, running_flag: Flag, group: str):
        self.action_file = action_file_path
        self.ignore_first_n_timesteps = ignore_first_n_timesteps
        self.target_observations = self._cut_off_beginning(pd.read_csv(target_observations_path))
        self.args = args
        self.worst_error = 0.0
        self.best_error = float('inf')
        self.group = group
        postfix = time.strftime("%Y-%m-%d_%H:%M:%S") + f"_{group}"
        self.log_dir = Path("data") / "auto_tuner_logs" / postfix
        self.observations_dir = self.log_dir / "observations"
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.running_flag = running_flag

        self.paramConverter = None

        self.step = 0

    def _calc_error(self, observations: list) -> float:
        achieved = self._cut_off_beginning(observations)
        error = achieved.values - self.target_observations.values
        return np.sum(error ** 2)

    def _cut_off_beginning(self, observations: Union[pd.DataFrame, list]) -> pd.DataFrame:
        return pd.DataFrame(observations).iloc[self.ignore_first_n_timesteps:].reset_index(drop=True)

    def _run_simulation(self, **params) -> RunDetails:
        executor = HeadlessPolicyExecutor(args=self.args, policy=ActionFilePolicy, path=self.action_file)

        assert len(executor.task.hands) == 1
        hand: AllegroHand = executor.task.hands[0]
        for name, value in params.items():
            self.paramConverter.apply_parameter(hand, name, value)

        return executor.execute()

    def _objective(self, **params):
        self.step += 1
        if self.running_flag.is_set():
            raise StopIteration("Optimization interrupted by running flag.")

        if self.exponentiate_params:
            params = {k: exp(v) for k, v in params.items()}

        wandb_dir = {k.replace("-", "/"): v for k, v in params.items()}
        wandb_dir["my_step"] = self.step

        try:
            metrics = self._run_simulation(**params, **self.paramConverter.known_params)
            error = self._calc_error(metrics.observations)
            self.worst_error = max(self.worst_error, error)
            if error < self.best_error:
                self.best_error = error
                metrics.save_observations(path=f"{self.observations_dir}/observations_{error}.csv")
                self.save_results(params)
            wandb_dir[f"eval/error_{self.group}"] = error
            wandb_dir[f"eval/best_error_{self.group}"] = self.best_error
            wandb.log(wandb_dir)
            return -error
        except PhysicsError as e:
            print("PhysicsError: ", e)
            return -self.worst_error

    def start(self) -> RunDetails:
        print(time.ctime(time.time()))
        groups = {
            "base": ["ffa0", "mfa0", "rfa0"],
            "first": ["ffa1", "mfa1", "rfa1"],
            "second": ["ffa2", "mfa2", "rfa2"],
            "third": ["ffa3", "mfa3", "rfa3"],
            "arm": ["forearm_tx"],
        }

        group = self.group
        self.paramConverter = ParameterConverter({
            group: groups[group]
        })

        span = 0.05
        bounds = self.paramConverter.get_bounds({
            "damping": (consts.DAMPING[group] * (1 - span), consts.DAMPING[group] * (1 + span)),
            "kp": (consts.KP[group] * (1 - span), consts.KP[group] * (1 + span)),
            "kd": (consts.KD[group] * (1 - span), consts.KD[group] * (1 + span)),
            "sluggish": (consts.SLUGGISH[group] * (1 - span), consts.SLUGGISH[group] * (1 + span)),
            "max_u": (consts.MAX_U[group] * (1 - span), consts.MAX_U[group] * (1 + span)),
        })

        self.exponentiate_params = group == "arm"
        if self.exponentiate_params:
            bounds = {k: (log(x), log(y)) for k, (x, y) in bounds.items()}

        window_size = {
            k: (y - x) * 0.3 for k, (x, y) in bounds.items()
        }
        optimizer = BayesianOptimization(f=self._objective, pbounds=bounds, verbose=1, random_state=5,
                                         bounds_transformer=PanningDomainReductionTransformer(
                                             minimum_window=window_size,
                                             eta=0.999
                                         ))
        try:
            optimizer.maximize(init_points=100, n_iter=3000)
        except StopIteration:
            print("Optimization interrupted.")
        finally:
            self.save_results(optimizer.max["params"])

    def save_results(self, params: dict):
        full_data = {
            "target": self.best_error,
            "params": params,
        }
        json_string = json.dumps(full_data, indent=4)
        print(f"Best result:\n{json_string}")
        with open(self.log_dir / f"best_params_{self.group}.json", "w") as f:
            f.write(json_string)


def get_excluded_dofs(group: str) -> tuple[str]:
    all_actuators = [
        "ffa0",
        "ffa1",
        "ffa2",
        "ffa3",
        "mfa0",
        "mfa1",
        "mfa2",
        "mfa3",
        "rfa0",
        "rfa1",
        "rfa2",
        "rfa3",
        "tha0",
        "tha1",
        "tha2",
        "tha3"
    ]
    postfix_mapping = {
        "base": "0",
        "first": "1",
        "second": "2",
        "third": "3",
        "arm": "x",
    }
    filtered = [a for a in all_actuators if not a.endswith(postfix_mapping[group])]
    return tuple(filtered)


def get_group_config(group: str) -> Defaults:
    args = get_config()
    args.environment_name = "RoboPianist-debug-AreYouSleepingOneHand-v0"
    args.intensity = 0.0
    # args.excluded_dofs = get_excluded_dofs(group)
    # args.historical_stack_size = 1
    # if group != "arm":
    #     args.forearm_dofs = ()
    return args


def optimize(group: str, flag: Flag = Flag()):
    # path = Path("data") / "one_at_a_time" / group
    path = Path("data") / "songs" / "2nd_iteration_fences"
    tuner = ParameterAutoTuner(
        action_file_path=path / "0_actions.csv",
        target_observations_path=path / "0_observations_rw.csv",
        ignore_first_n_timesteps=5,
        args=get_group_config(group),
        running_flag=flag,
        group=group
    )
    tuner.start()


if __name__ == "__main__":
    postfix = time.strftime("%Y-%m-%d_%H:%M:%S")
    wandb.init(project="robopianist-auto-tuner",
               entity="iamtypingsomethingrandom",
               name=f"OptimizedFixedSequence_{postfix}",
               dir="/tmp/wandb_logs_auto_tuner" + postfix)
    wandb.define_metric("*", "my_step")
    flag = Flag()
    threads = []
    # for group in ("base", "first", "second", "third", "arm"):
    for group in ("base", "first", "second", "third"):
        t = threading.Thread(target=optimize, args=(group, flag))
        threads.append(t)

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("Interrupt received, stopping optimization...")
        flag.set()
        for t in threads:
            t.join()
    finally:
        wandb.finish()
