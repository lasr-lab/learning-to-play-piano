import enum
import signal
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rospy
from alsa_midi import SequencerClient, PortType, EventType, ALSAError

from configs import Defaults, get_config
from robopianist import music
from robopianist.execution.pd_controller import PDController
from robopianist.execution.robot_dummies import ArmDummy, AllegroDummy
from robopianist.execution.utilities import KindBoundedArray
from robopianist.models.hands import allegro_hand_constants as consts
from robopianist.music.midi_file import NoteTrajectory
from robopianist.music.synthesizer import Synthesizer
from third_party.allegro import Allegro
from third_party.xArmPythonSDK.xarm.wrapper.xarm_api import XArmAPI

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
    NoHandRobot = HardwareConfig(ip="192.168.1.196", weight=0.0, center_of_gravity=(0.0, 0.0, 0.0), has_hand=False)
    FullRobot = HardwareConfig(ip="192.168.1.196", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                               has_hand=True)
    DummyRobot = HardwareConfig(ip="1.2.3.4", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                                has_hand=False, has_arm=False)
    OnlyHand = HardwareConfig(ip="1.2.3.4", weight=1.091, center_of_gravity=(-9.72, 18.38, 157.69),
                              has_hand=True, has_arm=False)


class HardwareAPI(ABC):
    @property
    @abstractmethod
    def spec(self) -> KindBoundedArray:
        pass

    @abstractmethod
    def observe(self) -> np.ndarray:
        pass

    @abstractmethod
    def take_actions(self, action: np.ndarray):
        pass

    @abstractmethod
    def close(self):
        pass


class NotesAPI(HardwareAPI):
    # TODO: This class replicates the behavior during training but could be optimized.
    # atm the agent never observes the keys it needs to press in the current timestep.

    def __init__(self, args: Defaults):
        _, self.goals = self.load_song(args)
        self.goals = self.goals[1:]  # Skip first step because it is never observed in simulation
        self.idx = 0

    @property
    def was_first(self) -> bool:
        return self.idx == 1  # Idx is incremented after observation

    @property
    def was_last(self) -> bool:
        return self.idx >= len(self.goals) + 1  # Simulation is repeating the last note once...

    @staticmethod
    def load_song(args: Defaults) -> (Sequence, Sequence):
        # environment_format: RoboPianist-debug-TwinkleTwinkleLittleStar-v0
        song = args.environment_name.split("-")[2]
        midi_file = music.load(song, stretch=args.stretch_factor, shift=args.shift_factor)

        note_traj = NoteTrajectory.from_midi(
            midi_file, args.control_timestep
        )
        note_traj.add_initial_buffer_time(args.initial_buffer_time)
        notes = note_traj.notes
        goals = []
        for t_start in range(len(notes)):
            goal = np.zeros(
                (args.n_steps_lookahead + 1, 49),
                dtype=np.float64,
            )
            t_end = min(t_start + args.n_steps_lookahead + 1, len(notes))
            for i, t in enumerate(range(t_start, t_end)):
                keys = [note.key for note in notes[t]]
                goal[i, keys] = 1.0
            goals.append(goal.ravel())
        return notes, goals

    def observe(self) -> np.ndarray:
        i = min(self.idx, len(self.goals) - 1)
        current_goal = self.goals[i]
        self.idx += 1
        return current_goal

    @property
    def spec(self) -> KindBoundedArray:
        dimensions = self.goals[0].shape[0]
        return KindBoundedArray(
            shape=(dimensions,),
            dtype=np.float32,
            minimum=np.zeros(dimensions),
            maximum=np.ones(dimensions),
            name="goals",
        )

    def take_actions(self, action: np.ndarray):
        raise NotImplementedError("NotesAPI is only get the goal observations")

    def close(self):
        pass

    def reset_index(self):
        self.idx = 0


class PianoAPI(HardwareAPI):
    def __init__(self):
        self.pressed_keys = np.zeros(49)
        self.monitor_piano = True
        self.midi_monitoring = threading.Thread(target=self.start_midi_monitoring, daemon=True)
        self.midi_monitoring.start()

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

    def observe(self) -> np.ndarray:
        return self.pressed_keys.copy()

    @property
    def spec(self) -> KindBoundedArray:
        dimensions = 49
        return KindBoundedArray(
            shape=(dimensions,),
            dtype=np.float32,
            minimum=np.zeros(dimensions),
            maximum=np.ones(dimensions),
            name="piano",
        )

    def take_actions(self, action: np.ndarray):
        raise NotImplementedError("Piano is only observed via MIDI")

    def close(self):
        self.monitor_piano = False
        self.midi_monitoring.join()


class ArmAPI(HardwareAPI):
    home_joints = np.array([0, 10, 0, 107.3, 180, -10.9, 45])
    hand_orientation = {"roll": -90.0, "pitch": -135, "yaw": -90.0}

    class Converter:
        back = 0
        up = 5.0
        c1_coords = np.array([435.0 + back, 338, 159.0 + up])
        c5_coords = np.array([435.0 + back, -338, 158.0 + up])
        c3_coords = 0.5042 * c5_coords + (1 - 0.5042) * c1_coords

        @staticmethod
        def _interpolate(p1, p2, x: np.ndarray):
            ta = x - p1
            g = p2 - p1
            length = np.sqrt(g.dot(g))
            g /= length
            p_distance = g.dot(ta)
            return p_distance / length

        @classmethod
        def to_tx(cls, coords: np.ndarray) -> float:
            tx = cls._interpolate(cls.c3_coords, cls.c1_coords, coords)
            if tx < 0:
                tx = cls._interpolate(cls.c3_coords, cls.c5_coords, coords)
                tx *= 1 / 3
            else:
                tx *= -1 / 3
            return tx

        @classmethod
        def to_coords(cls, tx: float) -> np.ndarray:
            if abs(tx) > 0.4:
                raise ValueError("Make sure that: -0.33 <= x <= 0.33")

            if tx < 0:
                tx *= -3
                return (1.0 - tx) * cls.c3_coords + tx * cls.c1_coords
            else:
                tx *= 3
                return (1.0 - tx) * cls.c3_coords + tx * cls.c5_coords

    def __init__(self, robot: Robot, physics_timestep):
        self.is_dummy = not robot.value.has_arm
        self.arm = self._init_arm(robot.value)
        self._set_arm_mode(in_servo_mode=False)
        self.go_home()

        self.is_active = True
        coords = self.Converter.to_coords(0.0)
        self.arm.set_position(*coords, wait=True, **self.hand_orientation)
        self.controller = PDController(
            time_step=physics_timestep,
            kp=consts.KP["arm"],
            kd=consts.KD["arm"],
            max_u=consts.MAX_U["arm"],
            sluggish=consts.SLUGGISH["arm"]
        )
        self.controller.set_target_position(0.0)
        self.controller_thread = threading.Thread(target=self._controller_thread, daemon=True)
        self.controller_thread.start()

    def _controller_thread(self):
        self._set_arm_mode(in_servo_mode=True)
        orientation = [self.hand_orientation[x] for x in ("roll", "pitch", "yaw")]

        pos = 0
        v = 0
        m = 1000.0
        while self.is_active:
            start = time.time()

            # Control
            f = self.controller.step(pos).item()
            f -= consts.DAMPING["arm"] * v

            a = f / m
            v += a * self.controller.time_step
            pos += v * self.controller.time_step

            # Act
            coords = self.Converter.to_coords(pos)
            self.arm.set_servo_cartesian(coords.tolist() + orientation)

            time_left = self.controller.time_step - (time.time() - start)
            if time_left > 0:
                time.sleep(time_left)

        self._set_arm_mode(in_servo_mode=False)

    def _init_arm(self, config: HardwareConfig):
        if self.is_dummy:
            return ArmDummy()
        arm = XArmAPI(config.ip)
        arm.set_tcp_load(
            weight=config.weight,
            center_of_gravity=config.center_of_gravity,
            wait=True,
        )
        return arm

    def _set_arm_mode(self, in_servo_mode: bool):
        self.arm.motion_enable(enable=True)
        self.arm.clean_error()
        mode = 1 if in_servo_mode else 0
        self.arm.set_mode(
            mode)
        self.arm.set_state(state=0)
        while self.arm.mode != mode:
            time.sleep(0.01)

    def take_actions(self, action: np.ndarray):
        self.spec.validate(action)
        clipped = np.clip(action, [-1 / 3], [1 / 3])
        tx = clipped.item()

        self.controller.set_target_position(tx)

    def go_home(self):
        self.arm.set_servo_angle(angle=self.home_joints, wait=True)

    def observe(self) -> np.ndarray:
        coords = self.arm.get_position()[1][0:3]
        tx = self.Converter.to_tx(coords)
        return np.array([tx])

    @property
    def spec(self) -> KindBoundedArray:
        return KindBoundedArray(
            shape=(1,),
            dtype=np.float32,
            minimum=np.array([-1 / 3]),
            maximum=np.array([1 / 3]),
            tolerance=np.array([0.01]),
            name="arm",
        )

    def close(self):
        self.is_active = False
        self.controller_thread.join()
        self.go_home()
        self.arm.disconnect()


class HandAPI(HardwareAPI):
    def __init__(self, robot: Robot):
        self.is_dummy = not robot.value.has_hand
        self.hand = self._init_hand()
        self.take_actions(self.get_defaults())

    @staticmethod
    def get_defaults() -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.363, 0.0, 0.0, 0.0], dtype=np.float32)

    def _init_hand(self):
        if self.is_dummy:
            hand = AllegroDummy(hand_topic_prefix="allegroHand_0")
        else:
            hand = Allegro(hand_topic_prefix="allegroHand_0")
            rospy.init_node("robot", disable_signals=True)
            # Wait until the hand publishes joint states → indicates driver is fully up
            while hand.pub_joint.get_num_connections() == 0:
                time.sleep(0.02)

            def handler(sig, frame):
                rospy.signal_shutdown("SIGTERM received")
                raise KeyboardInterrupt

            signal.signal(signal.SIGTERM, handler)

        self.hand_lower = np.array([-0.47, -0.196, -0.174, -0.227,
                                    -0.47, -0.196, -0.174, -0.227,
                                    -0.47, -0.196, -0.174, -0.227,
                                    0.263, -0.105, -0.189, -0.162])
        self.hand_upper = np.array([0.47, 1.61, 1.709, 1.618,
                                    0.47, 1.61, 1.709, 1.618,
                                    0.47, 1.61, 1.709, 1.618,
                                    1.396, 1.163, 1.644, 1.719])
        return hand

    def take_actions(self, action: np.ndarray):
        self.spec.validate(action)
        self.hand.command_joint_position(action.copy())

    def observe(self) -> np.ndarray:
        return np.array(self.hand.poll_joint_position(wait=False)[0])

    @property
    def spec(self) -> KindBoundedArray:
        return KindBoundedArray(
            shape=(16,),
            dtype=np.float32,
            minimum=self.hand_lower,
            maximum=self.hand_upper,
            tolerance=[0.001] * 16,
            name="hand",
        )

    def close(self):
        self.hand.disconnect()


class RobotCalibration:
    def __init__(self, args: Defaults, robot: Robot):
        self.piano = PianoAPI()
        self.hand = HandAPI(robot)
        self.arm = ArmAPI(robot, args.physics_timestep)

        self.hand.take_actions(np.array([
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.41, 1.71, 0.21,  # middle spiderman pos
            0.0, 0.0, 0.0, 0.0,
            0.363, 0.0, 0.0, 0.0], dtype=np.float32)
        )

    def move_to_input(self):
        try:
            while True:
                input_val = input("Enter desired arm tx position (-0.33 to 0.33) or 'q' to quit: ")
                if input_val.lower() == "q":
                    break
                tx = np.array([input_val], dtype=np.float32)
                self.arm.take_actions(tx)
        finally:
            print("\nShutting down...")
            self.hand.close()
            self.arm.close()
            self.piano.close()


if __name__ == "__main__":
    robot = Robot.DummyRobot
    calibration = RobotCalibration(get_config(), robot)
    calibration.move_to_input()
