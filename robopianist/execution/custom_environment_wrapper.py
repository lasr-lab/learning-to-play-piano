import enum
from dataclasses import dataclass

import numpy as np
from dm_env_wrappers._src import base


class PositionModes(enum.Enum):
    MIRROR_SIMULATION: str = "mirror_simulation"
    COPY_ACTIONS: str = "copy_actions"
    AVERAGE_ACTIONS: str = "copy_actions"
    ONLY_SIMULATION: str = "only_simulation"


@dataclass(frozen=True)
class PositioningConfig:
    fingerMode: PositionModes
    forearmMode: PositionModes


class ObservationWrapper(base.EnvironmentWrapper):

    def get_forearm_tx(self) -> np.ndarray:
        id = np.array(self.physics.model.actuator_trnid).transpose()[0][-1]
        tx = self.physics.data.qpos[id]
        return np.array([tx])

    def get_finger_joints(self, ordered=True) -> np.ndarray:
        ids = np.array(self.physics.model.actuator_trnid).transpose()[0][:-5]
        if ordered:
            ids = np.sort(ids, kind="stable")
        qpos = self.physics.data.qpos[ids]
        return qpos

    def get_activation(self) -> np.ndarray:
        ids = np.array(self.physics.model.actuator_trnid).transpose()[0]
        qpos = np.delete(self.physics.data.qpos, ids)
        ranges = np.delete(self.physics.model.jnt_range, ids, axis=0)[:, 1]
        activation = (
                np.abs(qpos - ranges) / ranges <= 0.5
        )
        return activation
