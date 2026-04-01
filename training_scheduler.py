import os
import random
import time
from copy import deepcopy
from itertools import product
from typing import Union

import numpy as np

import wandb
from configs import Defaults, get_config
from train import Trainer


class TrainingScheduler:
    def __init__(self, base_args: Defaults, name: str):
        self.variables = {}
        self.base_args = base_args
        self.name = name

    def add_variable_range(self, name, first: Union[int, float], last: Union[int, float], number_of_samples: int):
        if type(first) != type(last):
            raise ValueError("Start and end values must be of the same type.")
        explicit_values = np.linspace(first, last, number_of_samples)
        if type(first) == float:
            self.add_variable_explicit(name, tuple(explicit_values))
        elif type(first) == int:
            for val in explicit_values:
                if not val.is_integer():
                    raise ValueError(f"The provided range can not be sampled as integers. "
                                     f"Requested {number_of_samples} samples from {first} to {last} "
                                     f"but {val:.5f} is not an integer")
            self.add_variable_explicit(name, tuple(int(val) for val in explicit_values))
        else:
            raise ValueError("Start and end values must be either int or float.")

    def add_variable_explicit(self, name, values: tuple):
        if name in self.variables:
            raise ValueError(f"Variable '{name}' is already added.")
        if not hasattr(self.base_args, name):
            raise ValueError(f"Variable '{name}' is not a valid attribute of Defaults.")
        for value in values:
            if not isinstance(value, type(getattr(self.base_args, name))):
                raise ValueError(f"Value '{value}' does not match the type of '{name}'.")
        self.variables[name] = values

    @property
    def __envs(self) -> list[Defaults]:
        def __trim(value) -> str:
            if isinstance(value, float):
                return f"{value:.8f}".rstrip('0').rstrip('.')
            return str(value)

        keys = list(self.variables.keys())
        value_combinations = list(product(*self.variables.values()))
        envs = []
        run_id = random.randint(0, 2 ** 16)
        for combination in value_combinations:
            args_copy = deepcopy(self.base_args)
            name = self.name
            for key, value in zip(keys, combination):
                setattr(args_copy, key, value)
                name += f" | {key}: {__trim(value)}"
            args_copy.run_id = str(run_id)
            args_copy.name = f"{name} | ID: {run_id}"
            envs.append(args_copy)
        return envs

    def verify_and_start(self, execute_shutdown=False):
        envs = self.__envs
        total_iterations = 0
        for env in envs:
            env.prettyprint()
            total_iterations += env.max_steps
        print(f"\nGenerated {len(envs)} configurations")
        training_time = total_iterations / 3_000_000 * 3  # assuming 3 mio steps in 3h (based on sutton)
        print(f"Estimated time per configuration: {training_time / len(envs):.2f} hours ")
        print(f"Estimated total training time: {training_time:.2f} hours ")
        print(f"Estimated finish time: {time.ctime(time.time() + training_time * 3600)}")
        start = input(
            f"\nStart training with these configurations{' and shutdown' if execute_shutdown else ''}? (y/n): ")
        if start.lower() != "y":
            print("Aborting training.")
            return
        self._start_training()
        if execute_shutdown:
            for i in range(20):
                print(f"System will shutdown in {20 - i} seconds...")
                time.sleep(1)
            os.system('systemctl poweroff')

    def _start_training(self):
        for args in self.__envs:
            try:
                Trainer.start_training(args)
            except Exception as e:
                print(f"Training failed for configuration:\n{args}\nError: {e}")
            finally:
                wandb.finish()


if __name__ == "__main__":
    # Next trainings
    # 1. OT Reward but only for one dimension - running
    # 2. Add sliding penalty
    # 3. Fences
    scheduler = TrainingScheduler(get_config(), "DR Sim2Real2Sim")
    scheduler.add_variable_range("intensity", 0.0, 0.25, 11)
    scheduler.add_variable_explicit("environment_name", (
        "RoboPianist-debug-AreYouSleepingOneHand-v0",
        "RoboPianist-debug-HappyBirthdayOneHand-v0",
        "RoboPianist-debug-OdeToJoyOneHand-v0",
        "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        "RoboPianist-debug-CMajorScaleOneHand-v0",
    ))
    scheduler.verify_and_start(execute_shutdown=False)
