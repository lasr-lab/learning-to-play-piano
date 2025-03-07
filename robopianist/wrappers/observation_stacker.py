from typing import List, Dict

import dm_env
import numpy as np
import tree
from dm_env_wrappers._src.base import EnvironmentWrapper


class ObservationStackerWrapper(EnvironmentWrapper):
    """
        Stacks observations over a window.
    """

    def __init__(self, environment: dm_env.Environment, stack_backwards: int) -> None:
        super().__init__(environment)
        observation_spec = environment.observation_spec()
        self._input_spec = observation_spec
        self._observation_keys = list(observation_spec.keys())

        dummy_obs = tree.map_structure(lambda x: np.zeros(x.shape, x.dtype), observation_spec)
        self._init_historical_observations()

        dummy_output = self._stack_observation(dummy_obs)
        self._observation_spec = dm_env.specs.BoundedArray(
            shape=dummy_output.shape,
            dtype=dummy_output.dtype,
            minimum=-np.inf,
            maximum=np.inf,
            name="state",
        )

    def _stack_observation(self, observation):
        self._historical_observations.append(observation)
        self._historical_observations.pop(0)
        print(self._historical_observations)

        out = {k: [] for k in self._observation_keys}
        for observation in self._historical_observations:
            for k in self._observation_keys:
                out[k].extend(observation[k])

        return tree.map_structure(
            lambda x: np.array(x), out
        )

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return timestep._replace(
            observation=self._stack_observation(timestep.observation)
        )

    def _init_historical_observations(self):
        dummy_obs = tree.map_structure(lambda x: np.zeros(x.shape, x.dtype), self._input_spec)
        self._historical_observations: List[Dict] = [dummy_obs.copy() for _ in self._observation_keys]

    def reset(self) -> dm_env.TimeStep:
        self._init_historical_observations()
        return self._environment.reset()

    def observation_spec(self):
        return self._observation_spec
