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
_ALLEGRO_HAND_DIR = _HERE / "third_party" / "wonik_allegro_with_fingers_and_motors"

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

# PD Controller params
ALL_JOINT_NAMES = ["thj0", "thj1", "thj2", "thj3",
                   "ffj0", "ffj1", "ffj2", "ffj3",
                   "mfj0", "mfj1", "mfj2", "mfj3",
                   "rfj0", "rfj1", "rfj2", "rfj3", "forearm_tx"]

JOINT_TYPES = {
    "base": [x for x in ALL_JOINT_NAMES if x.endswith("0")],
    "first": [x for x in ALL_JOINT_NAMES if x.endswith("1")],
    "second": [x for x in ALL_JOINT_NAMES if x.endswith("2")],
    "third": [x for x in ALL_JOINT_NAMES if x.endswith("3")],
    "arm": ["forearm_tx"],
}

# Sim2Real2Sim without policy
# Targets / errors:
# base   -> 22.113590285387307
# first  -> 18.760239558718762
# second -> 126.23096883895487
# third  -> 127.19495325444211,
#
# DAMPING = {
#     "base": 0.3562994788700621,
#     "first": 1.3005083283130898,
#     "second": 1.1031178080904172,
#     "third": 0.7114966337244237,
#     "arm": 7950.00,
# }
#
# KP = {
#     "base": 146.09426482292076,
#     "first": 127.07420948800751,
#     "second": 79.97201432888261,
#     "third": 24.886718028688726,
#     "arm": 20_000,
# }
#
# KD = {
#     "base": 2.156796524312884,
#     "first": 4.002046481144024,
#     "second": 3.38154091485196,
#     "third": 1.9688878844806832,
#     "arm": 50,
# }
#
# SLUGGISH = {
#     "base": 0.9251654849367172,
#     "first": 0.9464929771780912,
#     "second": 0.958292241122525,
#     "third": 0.9709488204659543,
#     "arm": 0,
# }
#
# MAX_U = {
#     "base": 1.9460824135565866,
#     "first": 13.78864406393914,
#     "second": 10.135552669630892,
#     "third": 5.390989938642146,
#     "arm": 10_000,
# }

# Fences adjusted (bad parameters)
# DAMPING = {
#     "base": 0.33265201303703357,
#     "first": 1.3762093143184446,
#     "second": 1.237875596877689,
#     "third": 0.6373440389586467,
#     "arm": 7950.00,
# }
#
# KP = {
#     "base": 138.93216074602546,
#     "first": 127.37046964464001,
#     "second": 74.48343407082653,
#     "third": 23.181698663561303,
#     "arm": 20_000,
# }
#
# KD = {
#     "base": 2.1711937499830616,
#     "first": 4.326872554475312,
#     "second": 3.4258282879712687,
#     "third": 2.757579591360838,
#     "arm": 50,
# }
#
# SLUGGISH = {
#     "base": 0.8301124942351779,
#     "first": 0.9401242799577209,
#     "second": 0.9473782799434868,
#     "third": 0.9722220866132817,
#     "arm": 0,
# }
#
# MAX_U = {
#     "base": 1.7508249864125298,
#     "first": 14.64619427553635,
#     "second": 10.969145042555443,
#     "third": 4.705082499819736,
#     "arm": 10_000,
# }

# Before Fence adjusted but good on fences
DAMPING = {
    "base": 0.3262472550145872,
    "first": 1.3194740698809841,
    "second": 1.1440821850216027,
    "third": 0.6321203415532928,
    "arm": 7950.00,
}

KP = {
    "base": 144.519637390145,
    "first": 126.19912938004411,
    "second": 77.30142712168927,
    "third": 23.703264098163096,
    "arm": 20_000,
}

KD = {
    "base": 2.0937383192743986,
    "first": 4.1224883253438085,
    "second": 3.3066503332721138,
    "third": 2.5894844992533796,
    "arm": 50,
}

SLUGGISH = {
    "base": 0.8740293979043061,
    "first": 0.9318846615067442,
    "second": 0.9339833564637361,
    "third": 0.9727989108542677,
    "arm": 0,
}

MAX_U = {
    "base": 1.7878593456159815,
    "first": 14.226684215715363,
    "second": 10.563325879975567,
    "third": 4.918927888489592,
    "arm": 10_000,
}
