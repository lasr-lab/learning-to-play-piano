import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Sequence, Type, Dict

import tyro

from robopianist.models.arm.robot_arm import RobotArm
from robopianist.models.hands import AllegroHand, ShadowHand, Hand
from robopianist.models.piano.piano_constants import PianoType


class Randomization:
    @staticmethod
    def get_intermediate_value(min: float, max: float, intensity: float) -> float:
        diff = max - min
        additional = diff * intensity
        return min + additional

    @staticmethod
    def get_intermediate_tuple(min: tuple, max: tuple, intensity: float) -> Tuple[float, float]:
        return Randomization.get_intermediate_value(min[0], max[0], intensity), Randomization.get_intermediate_value(
            min[1], max[1], intensity)

    @staticmethod
    def get_randomization(intensity):
        return DRConfig(piano_pos_offset=Randomization.get_intermediate_value(0, 0.0085, intensity),
                        hand_pos_offset=Randomization.get_intermediate_value(0, 0.33, intensity),
                        damping_offset=Randomization.get_intermediate_value(0, 0.9, intensity),
                        activation_offset=Randomization.get_intermediate_tuple((0.75, 0.75), (0.1, 0.9),
                                                                               intensity),
                        key_stiffness=Randomization.get_intermediate_tuple((1.0, 1.0), (0.3, 2.3), intensity),
                        sliding_friction=Randomization.get_intermediate_tuple((1.0, 1.0), (0.2, 2.5), intensity))


class PrintableConfig:
    def prettyprint(self):
        return print(json.dumps(dataclasses.asdict(self), indent=4))


@dataclass(frozen=False)
class SACConfig(PrintableConfig):
    """Configuration options for SAC."""
    num_qs: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temp_lr: float = 3e-4
    hidden_dims: Sequence[int] = (256, 256, 256)
    activation: str = "gelu"
    num_min_qs: Optional[int] = None
    critic_dropout_rate: float = 0.01
    critic_layer_norm: bool = True
    tau: float = 0.005
    target_entropy: Optional[float] = None
    init_temperature: float = 1.0
    backup_entropy: bool = True


@dataclass(frozen=False)
class JointConfig(PrintableConfig):
    stiffness: float
    damping: float


@dataclass(frozen=False)
class DRConfig(PrintableConfig):
    piano_pos_offset: float = 0.004
    hand_pos_offset: float = 0.05
    activation_offset: Tuple[float, float] = (0.35, 0.75)
    key_stiffness: Tuple[float, float] = (0.5, 2.0)
    sliding_friction: Tuple[float, float] = (0.7, 1.3)
    hand_configuration: Dict[str, JointConfig] = dataclasses.field(default_factory=lambda: {
        'ffa0': JointConfig(stiffness=1, damping=0.1),
        'ffa1': JointConfig(stiffness=1, damping=0.1),
        'ffa2': JointConfig(stiffness=1, damping=0.1),
        'ffa3': JointConfig(stiffness=1, damping=0.1),
        'mfa0': JointConfig(stiffness=1, damping=0.1),
        'mfa1': JointConfig(stiffness=1, damping=0.1),
        'mfa2': JointConfig(stiffness=1, damping=0.1),
        'mfa3': JointConfig(stiffness=1, damping=0.1),
        'rfa0': JointConfig(stiffness=1, damping=0.1),
        'rfa1': JointConfig(stiffness=1, damping=0.1),
        'rfa2': JointConfig(stiffness=1, damping=0.1),
        'rfa3': JointConfig(stiffness=1, damping=0.1),
        'forearm_tx': JointConfig(stiffness=12_000, damping=6929.213901443674),
    })
    damping_offset: float = 0.0


@dataclass(frozen=False)
class Defaults(PrintableConfig):
    intensity: float = 0.0

    root_dir: str = "/tmp/robopianist/rl/"
    seed: int = 43
    max_steps: int = 5_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.8
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = ""
    name: str = ""
    run_id: str = "default"
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = True
    gravity_compensation: bool = True
    reduced_action_space: bool = True
    control_timestep: float = 0.05
    physics_timestep: float = 0.005
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    reward_functions: Sequence[str] = ("energy_reward", "hand_reward", "keypress_reward")
    energy_penalty_coef: float = 5e-3
    action_penalty_coef: float = 0.1
    sliding_penalty_coef: float = 0.1
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    primitive_fingertip_collisions: bool = True
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str] = "piano/arm"
    action_reward_observation: bool = False
    agent_config: SACConfig = SACConfig()
    robot_hand: str = "RobotArm"
    randomization_config: DRConfig = Randomization.get_randomization(intensity)
    use_action_curriculum: bool = False
    forearm_dofs: Sequence[str] = ("forearm_tx",)
    attachment_yaw: float = 0.0
    use_both_hands: bool = False
    piano_type: str = PianoType.M_AUDIO_KEYSTATION_49E.value
    start_model_upload: int = 900_000
    action_noise: float = 0.0
    excluded_dofs: Tuple[str, ...] = (
        "tha0",
        "tha1",
        "tha2",
        "tha3"
    )
    model_url: str = None

    def get_hand_class(self) -> Type[Hand]:
        if self.robot_hand == "allegro":
            return AllegroHand
        elif self.robot_hand == "shadow":
            return ShadowHand
        else:
            return RobotArm

    def get_piano_type(self) -> PianoType:
        return PianoType(self.piano_type)

    def __post_init__(self):
        self.randomization_config: DRConfig = Randomization.get_randomization(self.intensity)


@dataclass(frozen=False)
class SmallAgentConfig(SACConfig):
    hidden_dims: Sequence[int] = (128, 128, 128)
    temp_lr: float = 3e-5


@dataclass(frozen=False)
class AllegroConfig(Defaults):
    camera_id: Optional[str] = "piano/back"
    robot_hand: str = "allegro"
    reward_functions: Sequence[str] = ("energy_reward", "hand_reward", "friendly_keypress_reward")
    use_both_hands: bool = False
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    agent_config: SACConfig = SmallAgentConfig()
    piano_type = PianoType.M_AUDIO_KEYSTATION_49E.value


@dataclass(frozen=False)
class AllegroWithAdditionalDofsConfig(AllegroConfig):
    forearm_dofs: Sequence[str] = ("forearm_yaw", "forearm_roll", "forearm_tz", "forearm_ty", "forearm_tx")
    reward_functions: Sequence[str] = ("energy_reward", "hand_reward", "keypress_reward")


@dataclass(frozen=False)
class RobotArmConfig(AllegroConfig):
    robot_hand: str = "RobotArm"
    camera_id: Optional[str] = "piano/arm"
    energy_penalty_coef: float = 1.5e-3
    reward_functions: Sequence[str] = ("energy_reward", "hand_reward", "keypress_reward")


@dataclass(frozen=False)
class RealisticEnvConfig(AllegroConfig):
    max_steps: int = 1_000_000
    tqdm_bar: bool = False
    n_steps_lookahead: int = 5
    shift_factor: int = -15
    start_model_upload: int = 750_000
    stretch_factor: float = 1.0

    energy_penalty_coef: float = 0.12
    sliding_penalty_coef: float = 3.0
    action_penalty_coef: float = 0.5
    reward_functions: Sequence[str] = (
        "hand_reward", "sliding_reward", "keypress_reward", "energy_reward")

    name: str = f"Reproduce {random.randint(0, 2 ** 16)}"
    run_id: str = name


SELECTED_CLASS = RealisticEnvConfig


def get_config() -> Defaults:
    return tyro.cli(SELECTED_CLASS)


if __name__ == "__main__":
    get_config().prettyprint()
