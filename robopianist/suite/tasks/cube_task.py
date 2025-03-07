import abc
from pathlib import Path

from dm_control import composer, mjcf
from dm_control.mjcf import physics
from mujoco_utils import mjcf_utils, types

from robopianist.models.hands.cube_model import Cube
from robopianist.suite import PianoWithHands

import numpy as np

class CubeTask(PianoWithHands):
    def __init__(
            self,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._add_cube()
        self.new_pos = np.array((0, 0, 0), dtype=np.float32)

    def _add_cube(self):
        self.cube = Cube()
        self.need_update = False
        self._arena.attach(self.cube)

    def move_cube(self, direction):
        self.need_update = True
        self.new_pos += np.array(direction, dtype=np.float32) * 0.02

    def set_cube(self, pos):
        self.need_update = True
        self.new_pos = np.array(pos)

    def before_step(
            self,
            physics: mjcf.Physics,
            action: np.ndarray,
            random_state: np.random.RandomState,
    ) -> None:
        super().before_step(physics, action, random_state)
        if self.need_update:
            self.cube.set_pose(physics, self.new_pos)


