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

"""Base piano composer task."""

from typing import Sequence, Tuple

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils

from robopianist.models.arm import robot_arm_constants
from robopianist.models.arm.robot_arm import RobotArm
from robopianist.models.hands import HandSide, Hand, ShadowHand, AllegroHand, shadow_hand_constants, \
    allegro_hand_constants
from robopianist.models.piano import piano
from robopianist.models.piano.piano_constants import PianoType

# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP = 0.005

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 0.05  # 20 Hz.

# Rotation towards the center to simulate human posture
_ATTACHMENT_YAW = 0  # Degrees.


class PianoOnlyTask(composer.Task):
    """Piano task with no hands."""

    def __init__(
            self,
            arena: composer_utils.Arena,
            change_color_on_activation: bool = False,
            add_piano_actuators: bool = False,
            physics_timestep: float = _PHYSICS_TIMESTEP,
            control_timestep: float = _CONTROL_TIMESTEP,
            piano_type: PianoType = PianoType.CONVENTIONAL_KAWAII_PIANO,
    ) -> None:
        self._arena = arena
        self._piano = piano.Piano(
            change_color_on_activation=change_color_on_activation,
            add_actuators=add_piano_actuators,
            piano_type=piano_type,
        )
        arena.attach(self._piano)

        # Harden the piano keys.
        # The default solref parameters are (0.02, 1). In particular, the first
        # parameter specifies -stiffness, and so decreasing it makes the contacts
        # harder. The documentation recommends keeping the stiffness at least 2x larger
        # than the physics timestep, see:
        # https://mujoco.readthedocs.io/en/latest/modeling.html?highlight=stiffness#solver-parameters
        self._piano.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    @property
    def piano(self) -> piano.Piano:
        return self._piano

    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0


class PianoTask(PianoOnlyTask):
    """Base class for piano tasks."""

    def __init__(
            self,
            arena: composer_utils.Arena,
            gravity_compensation: bool = False,
            change_color_on_activation: bool = False,
            hand_class: type(Hand) = ShadowHand,
            primitive_fingertip_collisions: bool = False,
            reduced_action_space: bool = False,
            attachment_yaw: float = _ATTACHMENT_YAW,
            forearm_dofs: Sequence[str] = None,
            physics_timestep: float = _PHYSICS_TIMESTEP,
            control_timestep: float = _CONTROL_TIMESTEP,
            use_action_curriculum: bool = False,
            use_both_hands: bool = False,
            piano_type: PianoType = PianoType.M_AUDIO_KEYSTATION_49E,
            excluded_dofs: Tuple[str, ...] = ()
    ) -> None:
        super().__init__(
            arena=arena,
            change_color_on_activation=change_color_on_activation,
            add_piano_actuators=False,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            piano_type=piano_type,
        )

        self.hand_class = hand_class
        self._left_hand = self._add_hand(
            hand_side=HandSide.LEFT,
            hand_class=hand_class,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
            use_action_curriculum=use_action_curriculum,
            centered=not use_both_hands,
            excluded_dofs=excluded_dofs
        )
        self.hands = [self._left_hand]
        if use_both_hands:
            self._right_hand = self._add_hand(
                hand_side=HandSide.RIGHT,
                hand_class=hand_class,
                gravity_compensation=gravity_compensation,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                reduced_action_space=reduced_action_space,
                attachment_yaw=attachment_yaw,
                forearm_dofs=forearm_dofs,
                use_action_curriculum=use_action_curriculum,
                centered=not use_both_hands,
                excluded_dofs=excluded_dofs
            )
            self.hands.append(self._right_hand)
        else:
            self._right_hand = None

    # Accessors.
    @property
    def left_hand(self) -> Hand:
        return self._left_hand

    @property
    def right_hand(self) -> Hand:
        return self._right_hand

    # Helper methods.

    def _add_hand(
            self,
            hand_side: HandSide,
            hand_class: type(Hand),
            gravity_compensation: bool,
            primitive_fingertip_collisions: bool,
            reduced_action_space: bool,
            attachment_yaw: float,
            forearm_dofs: Sequence[str],
            use_action_curriculum: bool,
            centered: bool,
            excluded_dofs: Tuple[str, ...] = ()
    ) -> Hand:

        if hand_class == ShadowHand:
            if hand_side == HandSide.LEFT:
                position = shadow_hand_constants.LEFT_HAND_POSITION
            elif hand_side == HandSide.RIGHT:
                position = shadow_hand_constants.RIGHT_HAND_POSITION
            if forearm_dofs:
                hand = ShadowHand(
                    side=hand_side,
                    primitive_fingertip_collisions=primitive_fingertip_collisions,
                    restrict_wrist_yaw_range=False,
                    reduced_action_space=reduced_action_space,
                    forearm_dofs=forearm_dofs
                )
            else:
                hand = ShadowHand(
                    side=hand_side,
                    primitive_fingertip_collisions=primitive_fingertip_collisions,
                    restrict_wrist_yaw_range=False,
                    reduced_action_space=reduced_action_space
                )
            current_quaternion = shadow_hand_constants.QUATERNION

        elif hand_class == AllegroHand:
            if hand_side == HandSide.LEFT:
                position = allegro_hand_constants.LEFT_HAND_POSITION
            elif hand_side == HandSide.RIGHT:
                position = allegro_hand_constants.RIGHT_HAND_POSITION
            if centered:
                position[1] = 0

            if forearm_dofs:
                hand = AllegroHand(
                    side=hand_side,
                    primitive_fingertip_collisions=primitive_fingertip_collisions,
                    reduced_action_space=reduced_action_space,
                    forearm_dofs=forearm_dofs,
                    excluded_dofs=excluded_dofs
                )
            else:
                hand = AllegroHand(
                    side=hand_side,
                    primitive_fingertip_collisions=primitive_fingertip_collisions,
                    reduced_action_space=reduced_action_space,
                    excluded_dofs=excluded_dofs
                )
            current_quaternion = allegro_hand_constants.QUATERNION

        elif hand_class == RobotArm:
            if hand_side == HandSide.LEFT:
                position = robot_arm_constants.LEFT_ARM_POSITION
            elif hand_side == HandSide.RIGHT:
                position = robot_arm_constants.RIGHT_ARM_POSITION
            if centered:
                position[1] = 0

            hand = RobotArm(side=hand_side, use_action_curriculum=use_action_curriculum)

        hand.root_body.pos = position

        if hand_class != RobotArm:
            # Adapt sliding range to keyboard size
            joint_range = [-self._piano.size[1], self._piano.size[1]]

            # Offset the joint range by the hand's initial position.
            joint_range[0] -= position[1]
            joint_range[1] -= position[1]

            # Override forearm translation joint range.
            forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
            if forearm_tx_joint is not None:
                forearm_tx_joint.range = joint_range
            forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
            if forearm_tx_actuator is not None:
                forearm_tx_actuator.ctrlrange = joint_range

            # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
            rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
            rotate_by = np.zeros(4, dtype=np.float64)
            sign = -1 if hand_side == HandSide.LEFT else 1
            angle = np.radians(sign * attachment_yaw)
            mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
            final_quaternion = np.zeros(4, dtype=np.float64)
            mujoco.mju_mulQuat(final_quaternion, rotate_by, current_quaternion)
            hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        self._arena.attach(hand)
        return hand
