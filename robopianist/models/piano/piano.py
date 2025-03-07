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

"""Piano composer class."""

from typing import Sequence

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from mujoco_utils import mjcf_utils, types

from robopianist.models.piano import midi_module, piano_mjcf
from robopianist.models.piano.piano_constants import PianoConstants
from robopianist.models.piano.piano_constants import PianoType

# Key color when it is pressed.
_ACTIVATION_COLOR = (0.2, 0.8, 0.2, 1.0)


class Piano(composer.Entity):
    """A full-size standard (88-key) digital piano."""

    def _build(
            self,
            name: str = "piano",
            add_actuators: bool = False,
            change_color_on_activation: bool = True,
            piano_type: PianoType = PianoType.CONVENTIONAL_KAWAII_PIANO,
    ) -> None:
        """Initializes the piano.

        Args:
            name: Name of the piano. Used as a prefix in the MJCF name attributes.
            add_actuators: If True, actuators are added to the piano. This is used by
                the self-actuated piano task.
            change_color_on_activation: If True, the color of the key changes when it
                becomes activated.
        """
        self._key_threshold = 0.5
        self._change_color_on_activation = change_color_on_activation
        self._add_actuators = add_actuators

        self.piano_const = PianoConstants(type=piano_type)
        self._mjcf_root = piano_mjcf.build(add_actuators=add_actuators, piano_consts=self.piano_const)
        self._mjcf_root.model = name

        self._midi_module = midi_module.MidiModule(self.piano_const.NUM_KEYS)

        self._parse_mjcf_elements()
        self._add_mjcf_elements()
        self._initialize_state()  # Must be defined here for observables.

        _physics = mjcf.Physics.from_mjcf_model(self._mjcf_root)
        self._qpos_range = _physics.bind(self.joints).range
        if add_actuators:
            self._ctrl_midpoint = np.mean(
                _physics.bind(self.actuators).ctrlrange, axis=1
            )

    def _build_observables(self) -> "PianoObservables":
        return PianoObservables(self)

    def _parse_mjcf_elements(self) -> None:
        keys = mjcf_utils.safe_find_all(self._mjcf_root, "body")
        keys = keys[1:]  # Remove the base body.
        # Sort by increasing key number.
        sorted_idxs = np.argsort([int(key.name.split("_")[-1]) for key in keys])
        self._keys = tuple([keys[idx] for idx in sorted_idxs])

        key_geoms = mjcf_utils.safe_find_all(self._mjcf_root, "geom")
        key_geoms = key_geoms[1:]  # Remove the base geom.
        self._key_geoms = tuple([key_geoms[idx] for idx in sorted_idxs])

        joints = mjcf_utils.safe_find_all(self._mjcf_root, "joint")
        self._joints = tuple([joints[idx] for idx in sorted_idxs])

        sites = mjcf_utils.safe_find_all(self._mjcf_root, "site")
        self._sites = tuple([sites[idx] for idx in sorted_idxs])

        size = self._mjcf_root.find("body", "base").geom[0].size
        self._size = tuple(size)

        if self._add_actuators:
            actuators = mjcf_utils.safe_find_all(self._mjcf_root, "actuator")
            self._actuators = tuple([actuators[idx] for idx in sorted_idxs])

    def _add_mjcf_elements(self) -> None:
        # Add cameras.
        self._mjcf_root.worldbody.add(
            "camera",
            name="closeup",
            pos="-0.313 0.024 0.455",
            xyaxes="0.003 -1.000 -0.000 0.607 0.002 0.795",
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="left",
            pos="0.393 -0.791 0.638",
            xyaxes="0.808 0.589 0.000 -0.388 0.533 0.752",
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="right",
            pos="0.472 0.598 0.580",
            xyaxes="-0.637 0.771 -0.000 -0.510 -0.421 0.750",
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="back",
            pos="-0.569 0.008 0.841",
            xyaxes="-0.009 -1.000 0.000 0.783 -0.007 0.622",
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="arm",
            pos="-0.3 1 0.65",
            xyaxes="-0.637 -0.35 -0.000 -0.0 -0.421 0.750",
        )
        self._mjcf_root.worldbody.add(
            "camera",
            name="egocentric",
            pos="0.417 -0.039 0.717",
            xyaxes="-0.002 1.000 0.000 -0.867 -0.002 0.498",
        )
        pad_y = 0.5
        distance = 1.0
        fovy_radians = 2 * np.arctan2(pad_y * self._size[1], distance)
        self._mjcf_root.worldbody.add(
            "camera",
            name="topdown",
            pos=[0, 0, distance],
            quat=[1, 0, 0, 1],
            fovy=np.rad2deg(fovy_radians),
        )

    # Composer methods.

    def initialize_episode(
            self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._initialize_state()
        self._midi_module.initialize_episode(physics)
        self._update_key_state(physics)
        self._update_key_color(physics)

    def after_substep(
            self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        del random_state  # Unused.
        self._update_key_state(physics)
        self._update_key_color(physics)
        self._midi_module.after_substep(
            physics, self._activation
        )

    # Methods.

    def _initialize_state(self) -> None:
        self._state = np.zeros(self.piano_const.NUM_KEYS, dtype=np.float64)
        self._activation = np.zeros(self.piano_const.NUM_KEYS, dtype=bool)
        self._normalized_state = np.zeros(self.piano_const.NUM_KEYS, dtype=np.float64)

    def is_key_black(self, key_id: int) -> bool:
        """Returns True if the piano key id corresponds to a black key."""
        black_keys = [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        return bool(black_keys[key_id % 12])

    def _update_key_state(self, physics: mjcf.Physics) -> None:
        """Updates the state of the piano keys."""
        if self._add_actuators:
            ctrl_idxs = physics.bind(self.actuators).ctrl >= self._ctrl_midpoint
            self._activation[:] = ctrl_idxs
        else:
            # MuJoCo joint limits are soft, so we clip any joint positions that are
            # outside their limits.
            joints_pos = physics.bind(self.joints).qpos
            self._state[:] = np.clip(joints_pos, *self._qpos_range.T)
            self._normalized_state[:] = self._state / self._qpos_range[:, 1]
            self._activation[:] = (
                    np.abs(self._state - self._qpos_range[:, 1]) / self._qpos_range[:, 1] <= self._key_threshold
            )

    def _update_key_color(self, physics: mjcf.Physics) -> None:
        """Colors the piano keys if they are pressed."""
        if self._change_color_on_activation:
            physics.bind(self._key_geoms).rgba = np.where(
                self._activation[:, None],
                _ACTIVATION_COLOR,
                # Hacky way of restoring key color: we set the rgba of the geom to the
                # default gray so the inherited material, which specifies the white or
                # black rgba, kicks in.
                (0.5, 0.5, 0.5, 1.0),
            )
        else:
            physics.bind(self._key_geoms).rgba = (0.5, 0.5, 0.5, 1.0)

    def apply_action(
            self,
            physics: mjcf.Physics,
            action: np.ndarray,
            random_state: np.random.RandomState,
    ) -> None:
        del random_state  # Unused.
        if not self._add_actuators:
            raise ValueError("Cannot apply action if `add_actuators` is False.")
        physics.bind(self._actuators).ctrl = action[:]

    def set_activation_threshold(self, threshold: float):
        self._key_threshold = threshold

    # Accessors.

    @property
    def mjcf_model(self) -> types.MjcfRootElement:
        return self._mjcf_root

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    @property
    def joints(self) -> Sequence[types.MjcfElement]:
        return self._joints

    @property
    def keys(self) -> Sequence[types.MjcfElement]:
        return self._keys

    @property
    def key_geoms(self) -> Sequence[types.MjcfElement]:
        return self._key_geoms

    @property
    def activation(self) -> np.ndarray:
        return self._activation

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def normalized_state(self) -> np.ndarray:
        return self._normalized_state

    @property
    def size(self) -> Sequence[float]:
        return self._size

    @property
    def actuators(self) -> Sequence[types.MjcfElement]:
        if not self._add_actuators:
            raise ValueError("You must set add_actuators=True to use this property.")
        return self._actuators

    @property
    def midi_module(self) -> midi_module.MidiModule:
        return self._midi_module

    @property
    def action_labels(self) -> list[str]:
        return []


class PianoObservables(composer.Observables):
    """Observables for the piano."""

    _entity: Piano

    # TODO(kevin): Check if necessary to return copies of the underlying arrays.

    @composer.observable
    def joints_pos(self):
        """Returns the piano key joint positions."""

        def _get_joints_pos(physics: mjcf.Physics) -> np.ndarray:
            # We use the physics bind method because we need to preserve the order of
            # the joints specified in the class constructor and not the order in which
            # they are defined in the MJCF file.
            return physics.bind(self._entity.joints).qpos

        return observable.Generic(raw_observation_callable=_get_joints_pos)

    @composer.observable
    def activation(self):
        """Returns the piano key activations."""

        def _get_activation(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return self._entity.activation.astype(np.float64)

        return observable.Generic(raw_observation_callable=_get_activation)

    @composer.observable
    def state(self):
        """Returns the piano key states."""

        def _get_normalized_state(physics: mjcf.Physics) -> np.ndarray:
            del physics  # Unused.
            return self._entity.normalized_state

        return observable.Generic(raw_observation_callable=_get_normalized_state)
