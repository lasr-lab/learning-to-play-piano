from typing import Optional

import numpy as np
from dm_control import composer, mjcf
from mujoco_utils import mjcf_utils, types

import robopianist.models.arm.robot_arm_constants as consts
from robopianist.models.hands import base, HandSide, AllegroHand


class RobotArm(AllegroHand):
    def _build(self, name: Optional[str] = None, side: base.HandSide = base.HandSide.RIGHT,
               use_action_curriculum: bool = False, **kwargs) -> None:
        if side == HandSide.RIGHT:
            name = name or "right_arm"
            xml_file = consts.RIGHT_ALLEGRO_HAND_XML
        elif side == HandSide.LEFT:
            name = name or "left_arm"
            xml_file = consts.LEFT_ALLEGRO_HAND_XML
        self.action_curriculum_factor = 0.0 if use_action_curriculum else 1.0
        super()._build(name=name, xml_file=xml_file, side=side, forearm_dofs=[], **kwargs)

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root, "body", "link_base")

    @property
    def action_labels(self) -> list[str]:
        labels = [f"{self.hand_side.name}_arm_J{i}" for i in range(1, 8)]
        return labels + super().action_labels

    def notify_episode(self, episode: int):
        def get_factor(episode, start, duration):
            if episode < start:
                return 0.0
            if episode > duration + start:
                return 1.0
            return (episode - start) / duration

        if self.action_curriculum_factor != 1.0:
            self.action_curriculum_factor = get_factor(episode=episode, start=1.75e6, duration=500000)

    def apply_action(
            self,
            physics: mjcf.Physics,
            action: np.ndarray,
            random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        physics.bind(self.actuators[0:-16]).ctrl = action[0:-16]
        physics.bind(self.actuators[-16:]).ctrl = action[-16:] * self.action_curriculum_factor
