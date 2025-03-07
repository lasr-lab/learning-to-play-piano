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

"""Programatically build a piano MJCF model."""

import math

from dm_control import mjcf
from mujoco_utils import types

from robopianist.models.piano.piano_constants import PianoType, PianoConstants


def build(add_actuators: bool = False,
          piano_consts: PianoConstants = PianoConstants(PianoType.CONVENTIONAL_KAWAII_PIANO)) -> types.MjcfRootElement:
    """Programatically build a piano MJCF.

    Args:
        add_actuators: Whether to add actuators to the piano keys.
        piano_consts: Which kind of Piano should be built.
    """

    root = mjcf.RootElement()
    root.model = "piano"

    root.compiler.autolimits = True
    root.compiler.angle = "radian"

    # Add materials.
    root.asset.add("material", name="white", rgba=piano_consts.WHITE_KEY_COLOR)
    root.asset.add("material", name="black", rgba=piano_consts.BLACK_KEY_COLOR)

    root.default.geom.type = "box"
    root.default.joint.type = "hinge"
    root.default.joint.axis = [0, 1, 0]
    root.default.site.type = "box"
    root.default.site.group = 4
    root.default.site.rgba = [1, 0, 0, 1]

    # This effectively disables key-key collisions but still allows hand-key collisions,
    # assuming we've kept the default hand contype = conaffinity = 1.
    # See https://mujoco.readthedocs.io/en/latest/computation.html#selection for more
    # details.
    root.default.geom.contype = 0
    root.default.geom.conaffinity = 1
    root.default.geom.priority = 1

    # Actuator defaults (torque control).
    if add_actuators:
        root.default.general.dyntype = "none"
        root.default.general.dynprm = (piano_consts.ACTUATOR_DYNPRM, 0, 0)
        root.default.general.gaintype = "fixed"
        root.default.general.gainprm = (piano_consts.ACTUATOR_GAINPRM, 0, 0)
        root.default.general.biastype = "none"
        root.default.general.biasprm = (0, 0, 0)

    # White key defaults.
    white_default = root.default.add("default", dclass="white_key")
    white_default.geom.material = "white"
    white_default.geom.size = [
        piano_consts.WHITE_KEY_LENGTH / 2,
        piano_consts.WHITE_KEY_WIDTH / 2,
        piano_consts.WHITE_KEY_HEIGHT / 2,
    ]
    white_default.geom.mass = piano_consts.WHITE_KEY_MASS
    white_default.site.size = white_default.geom.size
    white_default.joint.pos = [-piano_consts.WHITE_KEY_LENGTH / 2, 0, 0]
    white_default.joint.damping = piano_consts.WHITE_JOINT_DAMPING
    white_default.joint.armature = piano_consts.WHITE_JOINT_ARMATURE
    white_default.joint.stiffness = piano_consts.WHITE_KEY_STIFFNESS
    white_default.joint.springref = piano_consts.WHITE_KEY_SPRINGREF * math.pi / 180
    white_default.joint.range = [0, piano_consts.WHITE_KEY_JOINT_MAX_ANGLE]
    if add_actuators:
        white_default.general.ctrlrange = [0, piano_consts.WHITE_KEY_JOINT_MAX_ANGLE]

    # Black key defaults.
    black_default = root.default.add("default", dclass="black_key")
    black_default.geom.material = "black"
    black_default.geom.size = [
        piano_consts.BLACK_KEY_LENGTH / 2,
        piano_consts.BLACK_KEY_WIDTH / 2,
        piano_consts.BLACK_KEY_HEIGHT / 2,
    ]
    black_default.site.size = black_default.geom.size
    black_default.geom.mass = piano_consts.BLACK_KEY_MASS
    black_default.joint.pos = [-piano_consts.BLACK_KEY_LENGTH / 2, 0, 0]
    black_default.joint.damping = piano_consts.BLACK_JOINT_DAMPING
    black_default.joint.armature = piano_consts.BLACK_JOINT_ARMATURE
    black_default.joint.stiffness = piano_consts.BLACK_KEY_STIFFNESS
    black_default.joint.springref = piano_consts.BLACK_KEY_SPRINGREF * math.pi / 180
    black_default.joint.range = [0, piano_consts.BLACK_KEY_JOINT_MAX_ANGLE]
    if add_actuators:
        black_default.general.ctrlrange = [0, piano_consts.BLACK_KEY_JOINT_MAX_ANGLE]

    # Add base.
    base_body = root.worldbody.add("body", name="base", pos=piano_consts.BASE_POS)
    base_body.add("geom", type="box", size=piano_consts.BASE_SIZE, rgba=piano_consts.BASE_COLOR)

    white_key_indices = piano_consts.WHITE_KEY_INDICES

    # These will hold kwargs. We'll subsequently use them to create the actual objects.
    geoms = []
    bodies = []
    joints = []
    sites = []
    actuators = []

    for i in range(piano_consts.NUM_WHITE_KEYS):
        y_coord = (
                -piano_consts.PIANO_LENGTH * 0.5
                + piano_consts.WHITE_KEY_WIDTH * 0.5
                + i * (piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS)
        )
        bodies.append(
            {
                "name": f"white_key_{white_key_indices[i]}",
                "pos": [piano_consts.WHITE_KEY_X_OFFSET, y_coord, piano_consts.WHITE_KEY_Z_OFFSET],
            }
        )
        geoms.append(
            {
                "name": f"white_key_geom_{white_key_indices[i]}",
                "dclass": "white_key",
            }
        )
        joints.append(
            {
                "name": f"white_joint_{white_key_indices[i]}",
                "dclass": "white_key",
            }
        )
        sites.append(
            {
                "name": f"white_key_site_{white_key_indices[i]}",
                "dclass": "white_key",
            }
        )
        if add_actuators:
            actuators.append(
                {
                    "joint": f"white_joint_{white_key_indices[i]}",
                    "name": f"white_actuator_{white_key_indices[i]}",
                    "dclass": "white_key",
                }
            )

    black_twin_key_indices = piano_consts.BLACK_TWIN_KEY_INDICES
    black_triplet_key_indices = piano_consts.BLACK_TRIPLET_KEY_INDICES

    if (piano_consts.piano_type == PianoType.CONVENTIONAL_KAWAII_PIANO
            or piano_consts.piano_type == PianoType.KAWAII_PIANO_BIG_KEYS):
        # Place the lone black key on the far left.
        y_coord = piano_consts.WHITE_KEY_WIDTH + 0.5 * (
                -piano_consts.PIANO_LENGTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS
        )
        bodies.append(
            {
                "name": f"black_key_{black_triplet_key_indices[0]}",
                "pos": [piano_consts.BLACK_KEY_X_OFFSET, y_coord, piano_consts.BLACK_KEY_Z_OFFSET],
            }
        )
        geoms.append(
            {
                "name": f"black_key_geom_{black_triplet_key_indices[0]}",
                "dclass": "black_key",
            }
        )
        joints.append(
            {
                "name": f"black_joint_{black_triplet_key_indices[0]}",
                "dclass": "black_key",
            }
        )
        sites.append(
            {
                "name": f"black_key_site_{black_triplet_key_indices[0]}",
                "dclass": "black_key",
            }
        )
        if add_actuators:
            actuators.append(
                {
                    "joint": f"black_joint_{black_triplet_key_indices[0]}",
                    "name": f"black_actuator_{black_triplet_key_indices[0]}",
                    "dclass": "black_key",
                }
            )
        TWIN_INDICES = list(range(2, piano_consts.NUM_WHITE_KEYS - 1, 7))
    else:
        TWIN_INDICES = list(range(0, piano_consts.NUM_WHITE_KEYS - 1, 7))
    n = 0
    # Place the twin black keys.
    for twin_index in TWIN_INDICES:
        for j in range(2):
            y_coord = (
                    -piano_consts.PIANO_LENGTH * 0.5
                    + (j + 1) * (piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS)
                    + twin_index
                    * (piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS)
            )
            bodies.append(
                {
                    "name": f"black_key_{black_twin_key_indices[n]}",
                    "pos": [
                        piano_consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        piano_consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{black_twin_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{black_twin_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{black_twin_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            if add_actuators:
                actuators.append(
                    {
                        "joint": f"black_joint_{black_twin_key_indices[n]}",
                        "name": f"black_actuator_{black_twin_key_indices[n]}",
                        "dclass": "black_key",
                    }
                )
            n += 1

    # Place the triplet black keys.
    if (piano_consts.piano_type == PianoType.CONVENTIONAL_KAWAII_PIANO
            or piano_consts.piano_type == PianoType.KAWAII_PIANO_BIG_KEYS):
        n = 1  # Skip the lone black key.
        TRIPLET_INDICES = list(range(5, piano_consts.NUM_WHITE_KEYS - 1, 7))
    else:
        n = 0
        TRIPLET_INDICES = list(range(3, piano_consts.NUM_WHITE_KEYS - 3, 7))
    for triplet_index in TRIPLET_INDICES:
        for j in range(3):
            y_coord = (
                    -piano_consts.PIANO_LENGTH * 0.5
                    + (j + 1) * (piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS)
                    + triplet_index
                    * (piano_consts.WHITE_KEY_WIDTH + piano_consts.SPACING_BETWEEN_WHITE_KEYS)
            )
            bodies.append(
                {
                    "name": f"black_key_{black_triplet_key_indices[n]}",
                    "pos": [
                        piano_consts.BLACK_KEY_X_OFFSET,
                        y_coord,
                        piano_consts.BLACK_KEY_Z_OFFSET,
                    ],
                }
            )
            geoms.append(
                {
                    "name": f"black_key_geom_{black_triplet_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            joints.append(
                {
                    "name": f"black_joint_{black_triplet_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            sites.append(
                {
                    "name": f"black_key_site_{black_triplet_key_indices[n]}",
                    "dclass": "black_key",
                }
            )
            if add_actuators:
                actuators.append(
                    {
                        "joint": f"black_joint_{black_triplet_key_indices[n]}",
                        "name": f"black_actuator_{black_triplet_key_indices[n]}",
                        "dclass": "black_key",
                    }
                )
            n += 1

    # Sort the elements based on the key number.
    names: list[str] = [body["name"] for body in bodies]  # type: ignore
    indices = sorted(range(len(names)), key=lambda k: int(names[k].split("_")[-1]))
    bodies = [bodies[i] for i in indices]
    geoms = [geoms[i] for i in indices]
    joints = [joints[i] for i in indices]
    sites = [sites[i] for i in indices]
    if add_actuators:
        actuators = [actuators[i] for i in indices]

    # Now create the corresponding MJCF elements and add them to the root.
    for i in range(len(bodies)):
        body = root.worldbody.add("body", **bodies[i])
        body.add("geom", **geoms[i])
        body.add("joint", **joints[i])
        body.add("site", **sites[i])
        if add_actuators:
            root.actuator.add("general", **actuators[i])

    return root
