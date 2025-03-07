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

"""Piano modeling constants.

Inspired by: https://kawaius.com/wp-content/uploads/2019/04/Kawai-Upright-Piano-Regulation-Manual.pdf
"""
import enum
from math import atan


class PianoType(enum.Enum):
    CONVENTIONAL_KAWAII_PIANO = "kawaii_regular"
    KAWAII_PIANO_BIG_KEYS = "kawaii_big"
    M_AUDIO_KEYSTATION_49E = "m_audio_keystation"


class PianoConstants:
    def __init__(self, type: PianoType):
        self.piano_type = type
        if type is PianoType.CONVENTIONAL_KAWAII_PIANO:
            self.build_kawaii(piano_size=0.0225)
        elif type is PianoType.KAWAII_PIANO_BIG_KEYS:
            self.build_kawaii(piano_size=0.035)
        else:
            self.build_m_audio_keystation()

    def build_kawaii(self, piano_size: float):
        self.NUM_KEYS = 88
        self.WHITE_KEY_WIDTH = piano_size

        self.NUM_WHITE_KEYS = 52
        self.WHITE_KEY_INDICES = [
            0,
            2,
            3,
            5,
            7,
            8,
            10,
            12,
            14,
            15,
            17,
            19,
            20,
            22,
            24,
            26,
            27,
            29,
            31,
            32,
            34,
            36,
            38,
            39,
            41,
            43,
            44,
            46,
            48,
            50,
            51,
            53,
            55,
            56,
            58,
            60,
            62,
            63,
            65,
            67,
            68,
            70,
            72,
            74,
            75,
            77,
            79,
            80,
            82,
            84,
            86,
            87,
        ]
        self.BLACK_TWIN_KEY_INDICES = [
            4,
            6,
            16,
            18,
            28,
            30,
            40,
            42,
            52,
            54,
            64,
            66,
            76,
            78,
        ]
        self.BLACK_TRIPLET_KEY_INDICES = [
            1,
            9,
            11,
            13,
            21,
            23,
            25,
            33,
            35,
            37,
            45,
            47,
            49,
            57,
            59,
            61,
            69,
            71,
            73,
            81,
            83,
            85,
        ]
        self.WHITE_KEY_LENGTH = 0.15

        self.WHITE_KEY_HEIGHT = 0.0225

        self.SPACING_BETWEEN_WHITE_KEYS = 0.001
        self.N_SPACES_BETWEEN_WHITE_KEYS = self.NUM_WHITE_KEYS - 1
        self.BLACK_KEY_WIDTH = 0.01
        self.BLACK_KEY_LENGTH = 0.09
        # Unlike the other dimensions, the height of the black key was roughly set such that
        # when a white key is fully depressed, the bottom of the black key is barely visible.
        self.BLACK_KEY_HEIGHT = 0.018
        self.PIANO_LENGTH = (self.NUM_WHITE_KEYS * self.WHITE_KEY_WIDTH) + (
                self.N_SPACES_BETWEEN_WHITE_KEYS * self.SPACING_BETWEEN_WHITE_KEYS
        )

        self.WHITE_KEY_X_OFFSET = 0
        self.WHITE_KEY_Z_OFFSET = self.WHITE_KEY_HEIGHT / 2
        self.BLACK_KEY_X_OFFSET = -self.WHITE_KEY_LENGTH / 2 + self.BLACK_KEY_LENGTH / 2
        # The top of the black key should be 12.5 mm above the top of the white key.
        self.BLACK_OFFSET_FROM_WHITE = 0.0125
        self.BLACK_KEY_Z_OFFSET = self.WHITE_KEY_HEIGHT + self.BLACK_OFFSET_FROM_WHITE - self.BLACK_KEY_HEIGHT / 2

        self.BASE_HEIGHT = 0.04
        self.BASE_LENGTH = 0.1
        self.BASE_WIDTH = self.PIANO_LENGTH
        self.BASE_SIZE = [self.BASE_LENGTH / 2, self.BASE_WIDTH / 2, self.BASE_HEIGHT / 2]
        self.BASE_X_OFFSET = -self.WHITE_KEY_LENGTH / 2 - 0.5 * self.BASE_LENGTH - 0.002
        self.BASE_POS = [self.BASE_X_OFFSET, 0, self.BASE_HEIGHT / 2]

        # A key is designed to travel downward 3/8 of an inch (roughly 10mm).
        # Assuming the joint is positioned at the back of the key, we can write:
        # tan(θ) = d / l, where d is the distance the key travels and l is the length of the
        # key. Solving for θ, we get: θ = arctan(d / l).
        self.WHITE_KEY_TRAVEL_DISTANCE = 0.01
        self.WHITE_KEY_JOINT_MAX_ANGLE = atan(self.WHITE_KEY_TRAVEL_DISTANCE / self.WHITE_KEY_LENGTH)
        # TODO(kevin): Figure out black key travel distance.
        self.BLACK_KEY_TRAVEL_DISTANCE = 0.008
        self.BLACK_KEY_JOINT_MAX_ANGLE = atan(self.BLACK_KEY_TRAVEL_DISTANCE / self.BLACK_KEY_LENGTH)
        # Mass in kg.
        self.WHITE_KEY_MASS = 0.04
        self.BLACK_KEY_MASS = 0.02
        # Joint spring reference, in degrees.
        # At equilibrium, the joint should be at 0 degrees.
        self.WHITE_KEY_SPRINGREF = -1
        self.BLACK_KEY_SPRINGREF = -1
        # Joint spring stiffness, in Nm/rad.
        # The spring should be stiff enough to support the weight of the key at equilibrium.
        self.WHITE_KEY_STIFFNESS = 2
        self.BLACK_KEY_STIFFNESS = 2
        # Joint damping and armature for smoothing key motion.
        self.WHITE_JOINT_DAMPING = 0.05
        self.BLACK_JOINT_DAMPING = 0.05
        self.WHITE_JOINT_ARMATURE = 0.001
        self.BLACK_JOINT_ARMATURE = 0.001

        # Actuator parameters (for self-actuated only).
        self.ACTUATOR_DYNPRM = 1
        self.ACTUATOR_GAINPRM = 1

        # Colors.
        self.WHITE_KEY_COLOR = [0.9, 0.9, 0.9, 1]
        self.BLACK_KEY_COLOR = [0.1, 0.1, 0.1, 1]
        self.BASE_COLOR = [0.15, 0.15, 0.15, 1]

    def build_m_audio_keystation(self):
        self.NUM_KEYS = 49
        self.WHITE_KEY_WIDTH = 0.0225

        self.NUM_WHITE_KEYS = 29
        self.WHITE_KEY_INDICES = []
        self.BLACK_TWIN_KEY_INDICES = []
        self.BLACK_TRIPLET_KEY_INDICES = []
        for octave in range(4):
            for key in [0, 2, 4, 5, 7, 9, 11]:
                self.WHITE_KEY_INDICES.append(key + 12 * octave)
            for key in [1, 3]:
                self.BLACK_TWIN_KEY_INDICES.append(key + 12 * octave)
            for key in [6, 8, 10]:
                self.BLACK_TRIPLET_KEY_INDICES.append(key + 12 * octave)
        # Add lonely C on the right
        self.WHITE_KEY_INDICES.append(1 + 4 * 12)
        self.WHITE_KEY_LENGTH = 0.15

        self.WHITE_KEY_HEIGHT = 0.0225

        self.SPACING_BETWEEN_WHITE_KEYS = 0.674 / 29 - self.WHITE_KEY_WIDTH
        self.N_SPACES_BETWEEN_WHITE_KEYS = self.NUM_WHITE_KEYS - 1
        self.BLACK_KEY_WIDTH = 0.01
        self.BLACK_KEY_LENGTH = 0.09
        # Unlike the other dimensions, the height of the black key was roughly set such that
        # when a white key is fully depressed, the bottom of the black key is barely visible.
        self.BLACK_KEY_HEIGHT = 0.018
        self.PIANO_LENGTH = (self.NUM_WHITE_KEYS * self.WHITE_KEY_WIDTH) + (
                self.N_SPACES_BETWEEN_WHITE_KEYS * self.SPACING_BETWEEN_WHITE_KEYS
        )

        self.WHITE_KEY_X_OFFSET = 0
        self.WHITE_KEY_Z_OFFSET = self.WHITE_KEY_HEIGHT / 2
        self.BLACK_KEY_X_OFFSET = -self.WHITE_KEY_LENGTH / 2 + self.BLACK_KEY_LENGTH / 2
        # The top of the black key should be 12.5 mm above the top of the white key.
        self.BLACK_OFFSET_FROM_WHITE = 0.0125
        self.BLACK_KEY_Z_OFFSET = self.WHITE_KEY_HEIGHT + self.BLACK_OFFSET_FROM_WHITE - self.BLACK_KEY_HEIGHT / 2

        self.BASE_HEIGHT = 0.04
        self.BASE_LENGTH = 0.1
        self.BASE_WIDTH = self.PIANO_LENGTH
        self.BASE_SIZE = [self.BASE_LENGTH / 2, self.BASE_WIDTH / 2, self.BASE_HEIGHT / 2]
        self.BASE_X_OFFSET = -self.WHITE_KEY_LENGTH / 2 - 0.5 * self.BASE_LENGTH - 0.002
        self.BASE_POS = [self.BASE_X_OFFSET, 0, self.BASE_HEIGHT / 2]

        # A key is designed to travel downward 3/8 of an inch (roughly 10mm).
        # Assuming the joint is positioned at the back of the key, we can write:
        # tan(θ) = d / l, where d is the distance the key travels and l is the length of the
        # key. Solving for θ, we get: θ = arctan(d / l).
        self.WHITE_KEY_TRAVEL_DISTANCE = 0.01
        self.WHITE_KEY_JOINT_MAX_ANGLE = atan(self.WHITE_KEY_TRAVEL_DISTANCE / self.WHITE_KEY_LENGTH)

        self.BLACK_KEY_TRAVEL_DISTANCE = 0.007
        self.BLACK_KEY_JOINT_MAX_ANGLE = atan(self.BLACK_KEY_TRAVEL_DISTANCE / self.BLACK_KEY_LENGTH)
        # Mass in kg.
        self.WHITE_KEY_MASS = 0.01
        self.BLACK_KEY_MASS = 0.005
        # Joint spring reference, in degrees.
        # At equilibrium, the joint should be at 0 degrees.
        self.WHITE_KEY_SPRINGREF = -1
        self.BLACK_KEY_SPRINGREF = -1
        # Joint spring stiffness, in Nm/rad.aa
        # The spring should be stiff enough to support the weight of the key at equilibrium.
        self.WHITE_KEY_STIFFNESS = 0.5
        self.BLACK_KEY_STIFFNESS = 0.5
        # Joint damping and armature for smoothing key motion.
        self.WHITE_JOINT_DAMPING = 0.05
        self.BLACK_JOINT_DAMPING = 0.05
        self.WHITE_JOINT_ARMATURE = 0.001
        self.BLACK_JOINT_ARMATURE = 0.001

        # Actuator parameters (for self-actuated only).
        self.ACTUATOR_DYNPRM = 1
        self.ACTUATOR_GAINPRM = 1

        # Colors.
        self.WHITE_KEY_COLOR = [0.95, 0.9, 0.9, 1]
        self.BLACK_KEY_COLOR = [0.1, 0.1, 0.1, 1]
        self.BASE_COLOR = [0.75, 0.75, 0.75, 1]
