import dm_env
import numpy as np
from dm_env_wrappers._src.base import EnvironmentWrapper


class ObservationStackerWrapper(EnvironmentWrapper):
    """
        Stacks observations over a window.
    """

    def __init__(self, environment: dm_env.Environment, observation_names: list[str], stacks: int) -> None:
        super().__init__(environment)
        self._stacks = stacks
        self._observation_names = observation_names
        assert set(observation_names) - set(environment.observation_spec().keys()) == set(), \
            f"Some observation names are not in the environment's observation spec. {environment.observation_spec().keys()}"
        self._reset_buffer()

    def _reset_buffer(self):
        self._observation_buffers = {}
        for name in self._observation_names:
            shape = self._environment.observation_spec()[name].shape
            self._observation_buffers[name] = [np.zeros(shape, dtype=np.float32) for _ in range(self._stacks)]

    def _stack(self, name: str, obs: np.ndarray) -> dm_env.TimeStep:
        buffer = self._observation_buffers[name]
        buffer.pop(0)
        buffer.append(obs)
        return np.concatenate(buffer, axis=0)

    def _convert_timestep(self, timestep: dm_env.TimeStep):
        only_stacked = {
            name: self._stack(name, timestep.observation[name]) for name in self._observation_names
        }
        all_observations = {**timestep.observation, **only_stacked}
        return timestep._replace(observation=all_observations)

    def step(self, action) -> dm_env.TimeStep:
        timestep = self._environment.step(action)
        return self._convert_timestep(timestep)

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        self._reset_buffer()
        return self._convert_timestep(timestep)

    def observation_spec(self):
        spec = self._environment.observation_spec()
        new_spec = spec.copy()
        for name in self._observation_names:
            old_spec = spec[name]
            new_spec[name] = dm_env.specs.Array(
                shape=(old_spec.shape[0] * self._stacks,),
                dtype=old_spec.dtype,
                name=old_spec.name,
            )
        return new_spec
