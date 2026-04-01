from third_party.allegro import Allegro


class AllegroDummy(Allegro):
    def __init__(self, hand_topic_prefix="allegroHand_0", num_joints=16):  # noqa
        self.pos = [0.0] * num_joints

    def disconnect(self):
        pass

    def command_joint_position(self, desired_pose):
        self.pos = desired_pose

    def poll_joint_position(self, wait=False):
        return self.pos, 42


class ArmDummy:
    def __init__(self, port=None, is_radian=False, do_not_open=False, **kwargs):  # noqa
        pass

    def set_mode(self, mode=0, detection_param=0):
        self.mode = mode

    def set_state(self, state=0):
        self.state = state

    def set_servo_cartesian(self, mvpose, speed=None, mvacc=None, mvtime=0, is_radian=None, is_tool_coord=False,
                            **kwargs):
        self.pos = mvpose

    def motion_enable(self, enable=True, servo_id=None):
        pass

    def clean_error(self):
        pass

    def set_tcp_load(self, weight, center_of_gravity, wait=False, **kwargs):
        pass

    def set_position(self, x=None, y=None, z=None, roll=None, pitch=None, yaw=None, radius=None,
                     speed=None, mvacc=None, mvtime=None, relative=False, is_radian=None,
                     wait=False, timeout=None, **kwargs):
        pass

    def set_servo_angle(self, servo_id=None, angle=None, speed=None, mvacc=None,
                        mvtime=None,
                        relative=False, is_radian=None, wait=False, timeout=None,
                        radius=None, **kwargs):
        pass

    def disconnect(self):
        pass

    def get_position(self, is_radian=None):
        return (None, self.pos)
