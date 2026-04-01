from dataclasses import asdict

import optuna
import pandas as pd

import wandb
from configs import get_config, SACConfig
from train import Trainer

study_name = ""
domain_randomization_global = 0


def objective(trial):
    # Architektur
    num_layers = trial.suggest_int("num_layers", 2, 3)
    hidden_dims = tuple(trial.suggest_categorical(f"dim_{i}", [64, 128, 256, 512]) for i in range(num_layers))

    # Lernraten
    actor_lr = trial.suggest_loguniform("actor_lr", 1e-5, 1e-3)
    critic_lr = trial.suggest_loguniform("critic_lr", 1e-5, 1e-3)
    temp_lr = trial.suggest_loguniform("temp_lr", 1e-5, 1e-3)
    noise = trial.suggest_loguniform("temp_lr", 1e-5, 1e-3)
    # Dropout
    # critic_dropout_rate = trial.suggest_uniform("critic_dropout_rate", 0.0, 0.3)
    # Polyak-Faktor
    # tau = trial.suggest_uniform("tau", 0.001, 0.02)
    # Temperatur
    # init_temperature = trial.suggest_uniform("init_temperature", 0.1, 2.0)
    # Q-Funktionen
    # num_qs = trial.suggest_int("num_qs", 2, 4)
    # Min-Q-Ensemble
    # num_min_qs = trial.suggest_int("num_min_qs", 1, num_qs)

    # Config anpassen
    args = get_config()
    args.agent_config = SACConfig(
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        temp_lr=temp_lr,
        hidden_dims=hidden_dims,
    )
    # Set Domain Randomization
    args.intensity = domain_randomization_global
    # args.randomization_config = Randomization.get_randomization(domain_randomization_global)

    args.seed = 42
    args.tags = f"optuna,{hidden_dims}{',' + study_name if study_name else ''}"
    args.name = f"o_{study_name} {trial.number}"

    args.mode = "disabled"
    # DEBUG for short train interval
    # args.eval_interval = 100
    # args.max_steps = 1_000
    # args.start_model_upload = 600
    args.create_video_flag = False

    f1_graphs = []
    return_list = []

    eval_return = Trainer.start_training(args, optuna_trial=trial, f1_list=f1_graphs, return_list=return_list)
    print(f"Trial {trial.number} finished with Eval Return={eval_return}")

    try:
        df_trials = pd.read_csv("data/evaluate_dr/optuna_trials_summary.csv")
    except FileNotFoundError:
        df_trials = pd.DataFrame()
    try:
        df_f1_graphs = pd.read_csv("data/evaluate_dr/optuna_f1_graphs.csv")
    except FileNotFoundError:
        df_f1_graphs = pd.DataFrame()

    try:
        trial_params = asdict(args)
        trial_params["trial_number"] = trial.number

        df_trials = pd.concat([df_trials, pd.DataFrame([trial_params])], ignore_index=True)
        df_f1_graphs = pd.concat([df_f1_graphs, pd.DataFrame([f1_graphs])], ignore_index=True)

        df_trials.to_csv("optuna_trials_summary.csv", index=False)
        df_f1_graphs.to_csv("optuna_f1_graphs.csv", index=False)
    except Exception as e:
        print(e)

    return eval_return


if __name__ == "__main__":

    for dom_rand in [0.6, 0.1]:
        domain_randomization_global = dom_rand
        study_name = f"optuna_4_model_size_dom_rand_{dom_rand}"

        init_args = get_config()
        wandb.init(
            project=init_args.project,
            entity=init_args.entity or None,
            tags=(init_args.tags.split(",") if init_args.tags else []),
            notes=init_args.notes or None,
            config=asdict(init_args),
            mode=init_args.mode,
            name=study_name,
        )

        pruner = optuna.pruners.MedianPruner(n_startup_trials=3,
                                             n_warmup_steps=1_500_000)
        sampler = optuna.samplers.TPESampler(multivariate=True)

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            storage="sqlite:///optuna_sac.db",
            load_if_exists=True,
        )

        # study = optuna.load_study(
        #     study_name=study_name,
        #     storage="sqlite:///optuna_sac.db"
        # )

        study.optimize(objective, n_trials=100, n_jobs=1)
        wandb.finish()

        print("Beste Parameter:", study.best_params)
