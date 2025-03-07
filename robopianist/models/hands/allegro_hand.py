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

"""Allegro hand composer class."""
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from dm_control import composer, mjcf
from dm_env import specs
from mujoco_utils import mjcf_utils, spec_utils, types

from robopianist.models.hands import allegro_hand_constants as consts
from robopianist.models.hands import base
from robopianist.models.hands.base import Dof
from robopianist.models.hands.shadow_hand import ShadowHandObservables

_FINGERTIP_OFFSET = 0.026
_THUMBTIP_OFFSET = 0.0275

_FOREARM_DOFS: Dict[str, Dof] = {
    "forearm_tx": Dof(
        joint_type="slide",
        axis=(0, -1, 0),
        stiffness=12000,
        # Note this is a dummy range, it will be set to the piano's length at task
        # initialization, see `robopianist/suite/tasks/base.py`.
        joint_range=(-1, 1),
    ),
    "forearm_ty": Dof(
        joint_type="slide", axis=(0, 0, 1), stiffness=300, joint_range=(-0.1, 0.0)
    ),
    "forearm_tz": Dof(
        joint_type="slide", axis=(-1, 0, 0), stiffness=1000, joint_range=(-0.07, 0.03)
    ),
    "forearm_roll": Dof(
        joint_type="hinge", axis=(0, 0, 1), stiffness=300, joint_range=(-0.5, 0.5)
    ),
    "forearm_pitch": Dof(
        joint_type="hinge", axis=(0, -1, 0), stiffness=50, joint_range=(-1, 1)
    ),
    "forearm_yaw": Dof(
        joint_type="hinge",
        axis=(1, 0, 0),
        stiffness=300,
        joint_range=(0.5, 1.5),
        reflect=True,
    ),
}


class AllegroHand(base.Hand):
    """An Allegro Hand."""

    def _build(
            self,
            name: Optional[str] = None,
            side: base.HandSide = base.HandSide.RIGHT,
            primitive_fingertip_collisions: bool = False,
            reduced_action_space: bool = False,
            forearm_dofs: Sequence[str] = ("forearm_tx",),
            xml_file: str = None,
            excluded_dofs: Tuple[str, ...] = ()
    ) -> None:
        """Initializes a AllegroHand.

        Args:
            name: Name of the hand. Used as a prefix in the MJCF name attributes.
            side: Which side (left or right) to model.
            primitive_fingertip_collisions: Whether to use capsule approximations for
                the fingertip colliders or the true meshes. Using primitive colliders
                speeds up the simulation.
            reduced_action_space: Whether to use a reduced action space.
            forearm_dofs: Which dofs to add to the forearm.
        """
        super()._build(forearm_dofs={k: v for k, v in _FOREARM_DOFS.items() if k in forearm_dofs}, side=side)
        if side == base.HandSide.RIGHT:
            name = name or "rh_allegro_hand"
            xml_file = xml_file or consts.RIGHT_ALLEGRO_HAND_XML
        elif side == base.HandSide.LEFT:
            name = name or "lh_allegro_hand"
            xml_file = xml_file or consts.LEFT_ALLEGRO_HAND_XML

        self._hand_side = side
        self._mjcf_root = mjcf.from_path(str(xml_file))
        self._mjcf_root.model = name
        self._reduce_action_space = reduced_action_space

        self._add_dofs()

        self._parse_mjcf_elements(excluded_dofs)
        self._add_mjcf_elements()

        if primitive_fingertip_collisions:
            for geom in self._mjcf_root.find_all("geom"):
                if (
                        geom.dclass.dclass == "plastic_collision"
                        and geom.mesh is not None
                        and geom.mesh.name is not None
                        and geom.mesh.name.endswith("distal_pst")
                ):
                    geom.type = "capsule"

        self._action_spec = None

    def _build_observables(self) -> "AllegroHandObservables":
        return AllegroHandObservables(self)

    def _parse_mjcf_elements(self, excluded_dofs: Tuple[str, ...]) -> None:
        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
        for act_name in excluded_dofs:
            act = [a for a in actuators if a.name == act_name][0]
            actuators.remove(act)
            # act.remove()

            joint = act.joint
            joints.remove(joint)

        self._joints = tuple(joints)
        self._actuators = tuple(actuators)

        # Turn thumb down (should be used together with _REDUCED_ACTION_SPACE_EXCLUDED_DOFS)
        # thumb_base = mjcf_utils.safe_find(
        #     self._mjcf_root, "body", "th_base"
        # )
        # if self._hand_side == base.HandSide.RIGHT:
        #     thumb_base.quat = "0.031 -0.706 -0.092 -0.701"
        # elif self._hand_side == base.HandSide.LEFT:
        #     thumb_base.quat = "0.031 0.706 -0.092 0.701"

    def _add_mjcf_elements(self) -> None:
        # Add sites to the tips of the fingers.
        fingertip_sites = []
        for tip_name in consts.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", tip_name
            )
            offset = _THUMBTIP_OFFSET if tip_name == "thdistal" else _FINGERTIP_OFFSET
            tip_site = tip_elem.add(
                "site",
                name=tip_name + "_site",
                pos=(0.0, 0.0, offset),
                type="sphere",
                size=(0.004,),
                group=composer.SENSOR_SITES_GROUP,
            )
            fingertip_sites.append(tip_site)
        self._fingertip_sites = tuple(fingertip_sites)

        # Add joint torque sensors.
        joint_torque_sensors = []
        for joint_elem in self._joints:
            site_elem = joint_elem.parent.add(
                "site",
                name=joint_elem.name + "_site",
                size=(0.001, 0.001, 0.001),
                type="box",
                rgba=(0, 1, 0, 1),
                group=composer.SENSOR_SITES_GROUP,
            )
            torque_sensor_elem = joint_elem.root.sensor.add(
                "torque",
                site=site_elem,
                name=joint_elem.name + "_torque",
            )
            joint_torque_sensors.append(torque_sensor_elem)
        self._joint_torque_sensors = tuple(joint_torque_sensors)

        # Add velocity and force sensors to the actuators.
        actuator_velocity_sensors = []
        actuator_force_sensors = []
        for actuator_elem in self._actuators:
            velocity_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorvel",
                actuator=actuator_elem,
                name=actuator_elem.name + "_velocity",
            )
            actuator_velocity_sensors.append(velocity_sensor_elem)

            force_sensor_elem = self._mjcf_root.sensor.add(
                "actuatorfrc",
                actuator=actuator_elem,
                name=actuator_elem.name + "_force",
            )
            actuator_force_sensors.append(force_sensor_elem)
        self._actuator_velocity_sensors = tuple(actuator_velocity_sensors)
        self._actuator_force_sensors = tuple(actuator_force_sensors)

        # Add touch sensors to the fingertips.
        fingertip_touch_sensors = []
        for tip_name in consts.FINGERTIP_BODIES:
            tip_elem = mjcf_utils.safe_find(
                self._mjcf_root, "body", tip_name
            )
            offset = _THUMBTIP_OFFSET if tip_name == "thdistal" else _FINGERTIP_OFFSET
            touch_site = tip_elem.add(
                "site",
                name=tip_name + "_touch_site",
                pos=(0.0, 0.0, offset),
                type="sphere",
                size=(0.01,),
                group=composer.SENSOR_SITES_GROUP,
                rgba=(0, 1, 0, 0.6),
            )
            touch_sensor = self._mjcf_root.sensor.add(
                "touch",
                site=touch_site,
                name=tip_name + "_touch",
            )
            fingertip_touch_sensors.append(touch_sensor)
        self._fingertip_touch_sensors = tuple(fingertip_touch_sensors)

    # Accessors.

    @property
    def hand_side(self) -> base.HandSide:
        return self._hand_side

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def name(self) -> str:
        return self._mjcf_root.model

    @composer.cached_property
    def root_body(self) -> types.MjcfElement:
        return mjcf_utils.safe_find(self._mjcf_root, "body", "palm")

    @composer.cached_property
    def fingertip_bodies(self) -> Sequence[types.MjcfElement]:
        return tuple(
            mjcf_utils.safe_find(self._mjcf_root, "body", name)
            for name in consts.FINGERTIP_BODIES
        )

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        return self._actuators

    @property
    def joint_torque_sensors(self) -> Sequence[types.MjcfElement]:
        return self._joint_torque_sensors

    @property
    def fingertip_sites(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_sites

    @property
    def actuator_velocity_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_velocity_sensors

    @property
    def actuator_force_sensors(self) -> Sequence[types.MjcfElement]:
        return self._actuator_force_sensors

    @property
    def fingertip_touch_sensors(self) -> Sequence[types.MjcfElement]:
        return self._fingertip_touch_sensors

    # Action specs.

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = spec_utils.create_action_spec(
                physics=physics, actuators=self.actuators, prefix=self.name
            )

        return self._action_spec

    def apply_action(
            self,
            physics: mjcf.Physics,
            action: np.ndarray,
            random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        physics.bind(self.actuators).ctrl = action

    @property
    def action_labels(self) -> list[str]:
        side = self.hand_side.name
        labels = []
        for finger in ("point", "middle", "ring"):
            for move in ("rot", "ankle", "middle", "tip"):
                labels += [f"{side}_{finger}_{move}"]
        for move in ("ankle", "rot", "middle", "tip"):
            labels += [f"{side}_thumb_{move}"]

        labels += self._forearm_dofs
        return labels


class AllegroHandObservables(ShadowHandObservables):
    """AllegroHand observables."""

    _entity: AllegroHand
