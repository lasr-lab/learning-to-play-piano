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

from pathlib import Path
from typing import Dict, Tuple

_HERE = Path(__file__).resolve().parent
_ALLEGRO_HAND_DIR = _HERE / "third_party" / "wonik_allegro_with_fingers"

NQ = 16  # Number of joints.
NU = 16  # Number of actuators.

JOINT_GROUP: Dict[str, Tuple[str, ...]] = {
    "thumb": ("thj0", "thj1", "thj2", "thj3"),
    "first": ("ffj0", "ffj1", "ffj2", "ffj3"),
    "middle": ("mfj0", "mfj1", "mfj2", "mfj3"),
    "ring": ("rfj0", "rfj1", "rfj2", "rfj3"),
}

FINGERTIP_BODIES: Tuple[str, ...] = (
    # Important: the order of these names should not be changed.
    "th_distal",
    "ff_distal",
    "mf_distal",
    "rf_distal",
)

FINGERTIP_COLORS: Tuple[Tuple[float, float, float], ...] = (
    # Important: the order of these colors should not be changed.
    (0.8, 0.2, 0.8),  # Purple.
    (0.8, 0.2, 0.2),  # Red.
    (0.2, 0.2, 0.8),  # Blue.
    (0.8, 0.8, 0.2),  # Yellow.
)

# Path to the allegro hand E3M5 XML file.
RIGHT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "right_hand.xml"
LEFT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "left_hand.xml"

# Default position and orientation of the hands.
# (back, right, up)
# LEFT_HAND_POSITION = (0.13, -0.18, 0.09)
LEFT_HAND_POSITION = [0.09, -0.18, 0.12]
RIGHT_HAND_POSITION = [0.09, 0.18, 0.12]
QUATERNION = (0, 1, 0, -1)
