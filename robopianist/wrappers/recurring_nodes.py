import dm_env
import numpy as np
from dm_env.specs import BoundedArray
from dm_env_wrappers import EnvironmentWrapper


class RecurringNodesWrapper(EnvironmentWrapper):
    """
        A wrapper that adds recurring nodes to the environment.
    """

    def __init__(self, environment: dm_env.Environment, n_nodes: int = 1) -> None:
        super().__init__(environment)
        self.n_nodes = n_nodes

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        recurring_nodes, model_actions = action[:self.n_nodes], action[self.n_nodes:]
        timestep = self._environment.step(model_actions)
        return dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=np.concatenate([recurring_nodes, timestep.observation], axis=0),
        )

    def _get_new_spec(self, old_spec: BoundedArray) -> BoundedArray:
        return BoundedArray(
            shape=(old_spec.shape[0] + self.n_nodes,),
            dtype=old_spec.dtype,
            name=old_spec.name,
            minimum=np.concatenate([-np.ones(self.n_nodes), old_spec.minimum], axis=0)
            if old_spec.minimum.shape != () else -np.inf,
            maximum=np.concatenate([np.ones(self.n_nodes), old_spec.maximum], axis=0)
            if old_spec.maximum.shape != () else np.inf,
        )

    def reset(self) -> dm_env.TimeStep:
        timestep = self._environment.reset()
        recurring_nodes = np.zeros((self.n_nodes,), dtype=np.float32)
        return dm_env.TimeStep(
            step_type=timestep.step_type,
            reward=timestep.reward,
            discount=timestep.discount,
            observation=np.concatenate([recurring_nodes, timestep.observation], axis=0),
        )

    def observation_spec(self):
        old_spec = self._environment.observation_spec()
        return self._get_new_spec(old_spec)

    def action_spec(self):
        old_spec = self._environment.action_spec()
        return self._get_new_spec(old_spec)
