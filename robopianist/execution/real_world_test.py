import enum
import threading
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rospy
from alsa_midi import SequencerClient, PortType, EventType, ALSAError
from third_party.xArmPythonSDK.xarm.wrapper.xarm_api import XArmAPI

from configs import Defaults
from configs import get_config
from robopianist import music
from robopianist.execution.robot_dummies import ArmDummy, AllegroDummy
from robopianist.music.midi_file import NoteTrajectory
from robopianist.music.synthesizer import Synthesizer
from third_party.allegro import Allegro

"""
This file is used to communicate with the real world robotics.
"""


@dataclass(frozen=True)
class HardwareConfig:
    ip: str
    weight: float
    center_of_gravity: Sequence[float]
    has_hand: bool
    has_arm: bool = True


class Robot(enum.Enum):
    # Replace the following with the actual IP addresses of the robots that are available to you
    DoorRobot = HardwareConfig(ip="192.168.1.1", weight=0.0, center_of_gravity=(0.0, 0.0, 0.0), has_hand=False)
    WallRobot = HardwareConfig(ip="192.168.1.2", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                               has_hand=True)
    DummyRobot = HardwareConfig(ip="192.168.1.3", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                                has_hand=False, has_arm=False)
    OnlyHand = HardwareConfig(ip="192.168.1.4", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                              has_hand=True, has_arm=False)


def rescale(val: np.ndarray, new_min: np.ndarray, new_max: np.ndarray, old_min: np.ndarray = None,
            old_max: np.ndarray = None) -> np.ndarray:
    if old_min is None:
        old_min = -np.ones(len(val))
    if old_max is None:
        old_max = np.ones(len(val))
    if new_min is None:
        new_min = -np.ones(len(val))
    if new_max is None:
        new_max = np.ones(len(val))

    # scale to 0-1
    scaled = val - old_min
    scaled /= old_max - old_min

    # scale to needed values
    scaled *= new_max - new_min
    scaled += new_min

    # clip wrong values
    scaled = np.clip(scaled, new_min, new_max)
    return scaled


class RobotAPI:
    hand_orientation = {"roll": -90.0, "pitch": 45.0 + 180, "yaw": -90.0}
    down = 4  # Used to fine tune the height of the piano
    c1_coords = np.array([485.0, 336.5, 165.0 + down])
    c5_coords = np.array([485.0, -336.5, 168.0 + down])
    c3_coords = 0.5042 * c5_coords + (1 - 0.5042) * c1_coords
    home_joints = np.array([0, 10, 0, 107.3, 180, -10.9, -45 - 180])

    def __init__(self, args: Defaults, robot: Robot, song: str = None):
        self.goals = None
        if song is not None:
            self.midi_file = music.load(song, stretch=args.stretch_factor, shift=args.shift_factor)

            note_traj = NoteTrajectory.from_midi(
                self.midi_file, args.control_timestep
            )
            self.notes = note_traj.notes
            self.goals = []
            for t_start in range(len(self.notes)):
                goal = np.zeros(
                    (args.n_steps_lookahead + 1, 49),
                    dtype=np.float64,
                )
                t_end = min(t_start + args.n_steps_lookahead + 1, len(self.notes))
                for i, t in enumerate(range(t_start, t_end)):
                    keys = [note.key for note in self.notes[t]]
                    goal[i, keys] = 1.0
                self.goals.append(goal.ravel())

        actuator_names = ["ffa0", "ffa1", "ffa2", "ffa3",
                          "mfa0", "mfa1", "mfa2", "mfa3",
                          "rfa0", "rfa1", "rfa2", "rfa3",
                          "tha0", "tha1", "tha2", "tha3"]
        self.used_finger_ids = []
        for i, name in enumerate(actuator_names):
            if name not in args.excluded_dofs:
                self.used_finger_ids.append(i)

        self.pressed_keys = np.zeros(49)
        self.monitor_piano = True
        self.midi_monitoring = threading.Thread(target=self.start_midi_monitoring, daemon=True)
        self.midi_monitoring.start()

        self.arm = self.init_arm(robot.value)
        self.hand = self.init_hand(use_hand_dummy=not robot.value.has_hand)
        self.position_piano(checked=True, only_finger=True)

        coords = self.convert_tx_coords(0.0)
        self.arm.set_position(*coords, wait=True, **self.hand_orientation)

    def move_to_input(self):
        i = 0
        while True:
            print(i)
            coords = self.convert_tx_coords(i)
            print(coords)
            print(self.convert_coords_tx(coords))
            self.arm.set_position(*coords, wait=True, **self.hand_orientation)
            input_val = input("Go To C1")
            if len(input_val) == 0:
                break
            i = float(input_val)

    def init_servo_mode(self):
        self._set_arm_mode(1)

    def disable_servo_mode(self):
        self._set_arm_mode(0)

    def _set_arm_mode(self, mode: int):
        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        self.arm.set_mode(mode)
        self.arm.set_state(state=0)

    def update_forearm_position(self, simulation_coords: np.ndarray = None, tx: float = None, is_scaled=False):
        orientation = [self.hand_orientation[x] for x in ("roll", "pitch", "yaw")]
        if tx is None:
            tx = self.convert_coords_tx(simulation_coords)
        if not is_scaled:
            tx = rescale(val=np.array([tx]), new_min=np.array([-1 / 3]), new_max=np.array([1 / 3]))[0]
        coords = self.convert_tx_coords(tx)
        self.arm.set_servo_cartesian(coords.tolist() + orientation)

    def start_midi_monitoring(self):
        client = SequencerClient("midi client")
        try:
            hardware_port = client.list_ports(input=True, type=PortType.MIDI_GENERIC | PortType.HARDWARE)
        except ALSAError as e:
            warnings.warn("Midi Keyboard not detected", stacklevel=2)
            return
        if len(hardware_port) == 0:
            warnings.warn("Midi Keyboard not detected", stacklevel=2)
            return
        hardware_port = hardware_port[0]
        software_port = client.create_port("input")
        client.subscribe_port(sender=hardware_port, dest=software_port)
        synth = Synthesizer()
        synth.start()

        while self.monitor_piano:
            event = client.event_input(timeout=1)
            if event is None:
                continue
            if event.type == EventType.NOTEON:
                if event.velocity == 0:
                    synth.note_off(event.note)
                    self.pressed_keys[event.note - 36] = 0
                else:
                    synth.note_on(event.note, 127)
                    self.pressed_keys[event.note - 36] = 1
        synth.stop()

    def init_arm(self, config: HardwareConfig):
        if config.has_arm:
            arm = XArmAPI(config.ip)
            arm.motion_enable(enable=True)
            arm.clean_error()
            arm.set_mode(0)
            arm.set_state(state=0)
            arm.set_tcp_load(
                weight=config.weight,
                center_of_gravity=config.center_of_gravity,
                wait=True,
            )
        else:
            arm = ArmDummy()
        return arm

    def init_hand(self, use_hand_dummy=False):
        if use_hand_dummy:
            hand = AllegroDummy(hand_topic_prefix="allegroHand_0")
        else:
            hand = Allegro(hand_topic_prefix="allegroHand_0")
            rospy.init_node("robot")
        self.hand_lower = np.array([-0.594, -0.296, -0.274, -0.327,
                                    -0.594, -0.296, -0.274, -0.327,
                                    -0.594, -0.296, -0.274, -0.327,
                                    0.363, -0.205, -0.289, -0.262])
        self.hand_upper = np.array([0.571, 1.736, 1.809, 1.718,
                                    0.571, 1.736, 1.809, 1.718,
                                    0.571, 1.736, 1.809, 1.718,
                                    1.496, 1.263, 1.744, 1.819])
        return hand

    def set_finger_joints(self, input_joints, ignore_used_finger_ids=False):
        if ignore_used_finger_ids:
            joints = input_joints
        else:
            joints = np.array([0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.363, 0.0, 0.0, 0.0])
            joints[self.used_finger_ids] = input_joints
        clipped = np.clip(joints, self.hand_lower, self.hand_upper)
        self.hand.command_joint_position(clipped)

    def position_piano(self, checked=False, only_finger=False):
        if not checked and len(input("Home Position")) > 0:
            return
        self.go_back_home()
        if not checked and len(input("Activate Allegro Hand")) > 0:
            return
        self.set_finger_joints(np.array([0, 0, 0, 0,  # index
                                         0.0, 0, 0, 0,  # middle
                                         0.0, 0, 0, 0,  # ring
                                         0.363, 0, 0, 0]), ignore_used_finger_ids=True)  # thumb
        if only_finger:
            return
        if not checked and len(input("Go To C1")) > 0:
            return
        self.arm.set_position(*self.c1_coords, wait=True, **self.hand_orientation)
        if not checked and len(input("Go To C5")) > 0:
            return
        self.go_back_home()
        self.arm.set_position(*self.c5_coords, wait=True, **self.hand_orientation)
        if not checked and len(input("Done")) > 0:
            return

    def go_back_home(self) -> None:
        self.arm.set_servo_angle(angle=self.home_joints, wait=True)

    def close_connections(self) -> None:
        self.monitor_piano = False
        self.disable_servo_mode()
        self.arm.disconnect()
        self.hand.disconnect()
        self.midi_monitoring.join()

    def convert_tx_coords(self, tx: float) -> np.ndarray:
        if abs(tx) > 0.4:
            raise ValueError("Make sure that: -0.33 <= x <= 0.33")

        if tx < 0:
            tx *= -3
            return (1.0 - tx) * self.c3_coords + tx * self.c1_coords
        else:
            tx *= 3
            return (1.0 - tx) * self.c3_coords + tx * self.c5_coords

    def convert_coords_tx(self, a: np.ndarray) -> float:
        def interpolate(p1, p2, x: np.ndarray):
            ta = x - p1
            g = p2 - p1
            length = np.sqrt(g.dot(g))
            g /= length
            p_distance = g.dot(ta)
            return p_distance / length

        tx = interpolate(self.c3_coords, self.c1_coords, a)
        if tx < 0:
            tx = interpolate(self.c3_coords, self.c5_coords, a)
            tx *= 1 / 3
        else:
            tx *= -1 / 3
        return tx

    def get_goals(self) -> np.ndarray:
        return self.goals

    def get_finger_joints(self) -> np.ndarray:
        joints = np.array(self.hand.poll_joint_position(wait=False)[0])[self.used_finger_ids]
        idx = np.array([8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])
        return joints[idx]

    def get_forearm_tx(self) -> np.ndarray:
        coords = self.arm.get_position()[1][0:3]
        tx = self.convert_coords_tx(coords)
        return np.array([tx])

    def get_activation(self) -> np.ndarray:
        return self.pressed_keys.copy()


if __name__ == "__main__":
    sim = RobotAPI(get_config(), song="TwinkleTwinkleLittleStar", robot=Robot.WallRobot)
    try:
        sim.move_to_input()
        sim.position_piano(checked=False, only_finger=False)
    finally:
        sim.go_back_home()
        sim.close_connections()
