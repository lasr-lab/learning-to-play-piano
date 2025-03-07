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

_HERE = Path(__file__).resolve().parent
_ALLEGRO_HAND_DIR = _HERE / "third_party" / "xarm7_and_allegro_with_fingers"

# Path to the allegro hand E3M5 XML file.
RIGHT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "right.xml"
LEFT_ALLEGRO_HAND_XML = _ALLEGRO_HAND_DIR / "left.xml"

# Default position and orientation of the arms.
# (back, right, up)
# LEFT_HAND_POSITION = (0.09, -0.18, 0.09)
LEFT_ARM_POSITION = [0.7, -0.22, 0]
RIGHT_ARM_POSITION = [0.7, 0.22, 0]

INITIAL_JOINT_STATES = (0.0, 0.64, 0.0, 0.92, 3.14, 1.29, 3.14)
