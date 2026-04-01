import pickle
import random
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import dm_env_wrappers as wrappers
import numpy as np
import optuna
from flax.serialization import from_bytes
from tqdm import tqdm

import replay
import robopianist.wrappers as robopianist_wrappers
import sac
import specs
from configs import get_config, Defaults
from robopianist import suite
from robopianist.execution.policies import download_model
from robopianist.wrappers.observation_stacker import ObservationStackerWrapper
from robopianist.wrappers.recurring_nodes import RecurringNodesWrapper

warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*jax.*")


class Dummy_wanDB:
    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


class Trainer:
    @classmethod
    def prefix_dict(cls, prefix: str, d: dict) -> dict:
        return {f"{prefix}/{k}": v for k, v in d.items()}

    @classmethod
    def get_env(cls, args: Defaults, record_dir: Optional[Path] = None, evaluate: bool = False):
        env = suite.load(
            environment_name=args.environment_name,
            seed=args.seed,
            stretch=args.stretch_factor,
            shift=args.shift_factor,
            task_kwargs=dict(
                n_steps_lookahead=args.n_steps_lookahead,
                trim_silence=args.trim_silence,
                gravity_compensation=args.gravity_compensation,
                reduced_action_space=args.reduced_action_space,
                control_timestep=args.control_timestep,
                wrong_press_termination=args.wrong_press_termination,
                disable_colorization=args.disable_colorization,
                disable_hand_collisions=args.disable_hand_collisions,
                primitive_fingertip_collisions=args.primitive_fingertip_collisions,
                change_color_on_activation=True,
                hand_class=args.get_hand_class(),
                forearm_dofs=args.forearm_dofs,
                energy_penalty_coef=args.energy_penalty_coef,
                action_penalty_coef=args.action_penalty_coef,
                sliding_penalty_coef=args.sliding_penalty_coef,
                optimal_transport_coef=args.optimal_transport_coef,
                attachment_yaw=args.attachment_yaw,
                intensity=args.intensity,
                reward_functions=args.reward_functions,
                use_both_hands=args.use_both_hands,
                piano_type=args.get_piano_type(),
                excluded_dofs=args.excluded_dofs,
                physics_timestep=args.physics_timestep,
                hand_side=args.hand_side,
                initial_buffer_time=args.initial_buffer_time
            ),
        )
        task = env.task
        if record_dir is not None:
            if args.create_video_flag:
                env = robopianist_wrappers.PianoSoundVideoWrapper(
                    environment=env,
                    record_dir=record_dir,
                    record_every=args.record_every,
                    camera_id=args.camera_id,
                    height=args.record_resolution[0],
                    width=args.record_resolution[1],
                )
            env = wrappers.EpisodeStatisticsWrapper(
                environment=env, deque_size=args.record_every
            )
        else:
            env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=1)

        if record_dir is not None or evaluate:
            env = robopianist_wrappers.MidiEvaluationWrapper(
                environment=env, deque_size=args.record_every
            )

        if args.action_reward_observation:
            env = wrappers.ObservationActionRewardWrapper(env)
        if args.historical_stack_size > 1:
            env = ObservationStackerWrapper(env, ["rh_allegro_hand/joints_pos"], stacks=args.historical_stack_size)
        env = wrappers.ConcatObservationWrapper(env)
        env = wrappers.ActionNoiseWrapper(env, scale=args.action_noise, seed=args.seed)
        if args.frame_stack > 1:
            env = wrappers.FrameStackingWrapper(
                env, num_frames=args.frame_stack, flatten=True
            )
        if args.n_recurring_nodes > 0:
            env = RecurringNodesWrapper(env, args.n_recurring_nodes)
        env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
        env = wrappers.SinglePrecisionWrapper(env)
        env = wrappers.DmControlWrapper(env)
        return env, task

    @classmethod
    def start_training(cls, args: Defaults, optuna_trial=None, f1_list=None, return_list=None) -> float:
        log_dict = None
        if args.name:
            run_name = args.name
        else:
            run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

        # Create experiment directory.
        experiment_dir = Path(args.root_dir) / run_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(args.root_dir) / run_name / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Seed RNGs.
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.mode == "online" or args.mode == "offline":
            import wandb
            wandb.init(
                project=args.project,
                entity=args.entity or None,
                tags=(args.tags.split(",") if args.tags else []),
                notes=args.notes or None,
                config=asdict(args),
                mode=args.mode,
                name=run_name,
            )
        else:
            wandb = Dummy_wanDB()

        env, task_1 = cls.get_env(args)
        eval_env, task_2 = cls.get_env(args, record_dir=experiment_dir / "eval")
        tasks = (task_1, task_2)

        if args.model_url is not None:
            env_path, model_path = download_model(args.model_url)
            spec = pickle.load(open(env_path, "rb"))
        else:
            spec = specs.EnvironmentSpec.make(env)
        env_spec_path = model_dir / "env_spec.pkl"
        with open(env_spec_path, "wb") as file:
            pickle.dump(spec, file, protocol=pickle.HIGHEST_PROTOCOL)
        wandb.save(env_spec_path)

        agent = sac.SAC.initialize(
            spec=spec,
            config=args.agent_config,
            seed=args.seed,
            discount=args.discount,
        )
        if args.model_url is not None:
            with open(model_path, "rb") as file:
                agent = from_bytes(agent, file.read())

        replay_buffer = replay.Buffer(
            state_dim=spec.observation_dim,
            action_dim=spec.action_dim,
            max_size=args.replay_capacity,
            batch_size=args.batch_size,
        )

        timestep = env.reset()
        replay_buffer.insert(timestep, None)

        start_time = time.time()
        high_score = -1.0
        for i in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
            for x in tasks:
                x.notify_episode(i)

            # Act.
            if i < args.warmstart_steps:
                action = spec.sample_action(random_state=env.random_state)
            else:
                agent, action = agent.sample_actions(timestep.observation)

            # Observe.
            timestep = env.step(action)
            replay_buffer.insert(timestep, action)

            # Reset episode.
            if timestep.last():
                wandb.log(cls.prefix_dict("train", env.get_statistics()), step=i)
                timestep = env.reset()
                replay_buffer.insert(timestep, None)

            # Train.
            if i >= args.warmstart_steps:
                if replay_buffer.is_ready():
                    transitions = replay_buffer.sample()
                    agent, metrics = agent.update(transitions)
                    if i % args.log_interval == 0:
                        wandb.log(cls.prefix_dict("train", metrics), step=i)

            # Eval.
            if i % args.eval_interval == 0:
                for _ in range(args.eval_episodes):
                    timestep = eval_env.reset()
                    while not timestep.last():
                        timestep = eval_env.step(agent.eval_actions(timestep.observation))
                log_dict = cls.prefix_dict("eval", eval_env.get_statistics())
                music_dict = cls.prefix_dict("eval", eval_env.get_musical_metrics())
                wandb.log(log_dict | music_dict, step=i)
                print(log_dict | music_dict)
                if args.create_video_flag:
                    video = wandb.Video(str(eval_env.latest_filename), fps=4, format="mp4")
                    wandb.log({"video": video, "global_step": i})
                    eval_env.latest_filename.unlink()
                if i > args.start_model_upload and music_dict["eval/f1"] > high_score:
                    high_score = music_dict["eval/f1"]
                    agent.save(model_dir / f"{i:0{len(str(args.max_steps))}d}.mpk", wandb)

                if optuna_trial is not None:
                    eval_reward = log_dict["eval/return"]
                    f1_score = music_dict["eval/f1"]
                    optuna_trial.report(eval_reward, step=i)

                    f1_list.append(f1_score)
                    return_list.append(eval_reward)

                    # typically logging is disabled!
                    wandb.log({"trial_number": optuna_trial.number, "f1_score": f1_score, "return": eval_reward},
                              step=i)
                    if optuna_trial.should_prune():
                        optuna_trial.set_user_attr("f1 score", f1_score)
                        optuna_trial.set_user_attr("eval return", eval_reward)
                        raise optuna.TrialPruned()

            if i % args.log_interval == 0:
                wandb.log({"train/fps": int(i / (time.time() - start_time))}, step=i)

        wandb.finish()

        if log_dict:
            return log_dict["eval/return"]
        return float("-inf")


if __name__ == "__main__":
    Trainer.start_training(get_config())
