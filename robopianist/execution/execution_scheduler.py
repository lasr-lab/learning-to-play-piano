import json
import os
import re
import sys
import time
import traceback
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import wandb

from configs import get_config
from robopianist.execution.policies import ModelPolicy
from robopianist.execution.policy_executors import RealWorldPolicyExecutor
from robopianist.execution.robot_apis import Robot


class ExecutionStatus(Enum):
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CRASHED = "CRASHED"
    INTERRUPTED = "INTERRUPTED"


class ExecutionScheduler:
    def __init__(self, group: str, name: Optional[str] = None, overwrite: bool = False):
        self.primary_key = "task_id"

        postfix = f"_{name}" if name is not None else ""
        self.schedule_path = Path("data") / "schedules" / f"{group}_schedule{postfix}.csv"

        if self.schedule_path.exists() and not overwrite:
            print(f"Schedule file {self.schedule_path} already exists. Using existing schedule.")
            return

        os.makedirs(self.schedule_path.parent, exist_ok=True)
        self._create_schedule(group)

    @staticmethod
    def get_latest_model_info(run, folder="models/") -> dict[str, Union[str, int]]:
        model_files = []
        for f in run.files():
            if f.name.startswith(folder) and f.name.endswith(".mpk"):
                match = re.search(r"/(\d+)\.mpk$", f.name)
                if match:
                    model_files.append((int(match.group(1)), f))

        if not model_files:
            return None

        step, f = max(model_files, key=lambda x: x[0])
        return {
            "best_model_step": step,
            "model_url": f.url
        }

    def pull_task_info(self, run) -> dict:
        new_row = pd.json_normalize(run.config).to_dict(orient='records')[0]  # Flattens dictionary
        for key, value in self.get_latest_model_info(run).items():
            new_row[key] = value
        new_row["train_f1"] = run.summary["eval/f1"]
        new_row["train_precision"] = run.summary["eval/precision"]
        new_row["train_recall"] = run.summary["eval/recall"]
        new_row["train_return"] = run.summary["eval/return"]
        return new_row

    def _create_schedule(self, group_name: str) -> None:
        entity = "iamtypingsomethingrandom"
        project = "robopianist"

        api = wandb.Api()
        runs = api.runs(
            f"{entity}/{project}",
            filters={
                "group": group_name
            }
        )

        if len(runs) == 0:
            raise ValueError(f"No runs found for group {group_name} in project {project} and entity {entity}.")
        print(f"Number of runs: {len(runs)}")

        all_tasks = []
        for idx, run in enumerate(runs):
            cache_path = f"/tmp/robopianist/{run.id}/task_info.json"
            if os.path.isfile(cache_path):
                with open(cache_path, "r") as f:
                    row_info = json.load(f)
            else:
                row_info = self.pull_task_info(run)
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(row_info, f)

            row_info[self.primary_key] = idx

            all_tasks.append(row_info)
            print(f"Loaded run: {run.name}")

        print(f"Creating schedule {group_name}...")

        df = pd.DataFrame(all_tasks)

        # Convert non-primitive types to JSON strings
        df = df.map(lambda x: json.dumps(x) if type(x) not in (str, float, int) else x)

        # Keep only columns with more than one unique value
        if len(df) > 1:
            df = df.loc[:, df.nunique(dropna=True) > 1]

        # Add None to columns that are filled after execution and supress warnings
        with pd.option_context("mode.chained_assignment", None):
            for elem in ("precision", "recall", "f1_score"):
                df[elem] = None

        # Set initial values
        df["status"] = ExecutionStatus.SCHEDULED.value
        df["timestamp"] = self._current_timestamp

        df.to_csv(self.schedule_path, index=False)

    @property
    def _current_timestamp(self) -> str:
        return time.ctime(time.time())

    @property
    def __all_tasks(self) -> pd.DataFrame:
        return pd.read_csv(self.schedule_path)

    def _get_next_task(self, include_interrupted_runs=False):
        df = self.__all_tasks

        status_filter = [ExecutionStatus.SCHEDULED.value]
        if include_interrupted_runs:
            status_filter.append(ExecutionStatus.INTERRUPTED.value)

        runs_left = df.loc[df["status"].isin(status_filter)]

        if len(runs_left) == 0:
            return None

        return runs_left.iloc[0]

    def _update_task_status(self, task_id: str, status: ExecutionStatus, f1=None, recall=None, precision=None) -> None:
        """Update the status of the task in the schedule file to the given status and set metrics if provided"""

        if f1 is not None or recall is not None or precision is not None:
            assert status == ExecutionStatus.COMPLETED
        if status == ExecutionStatus.COMPLETED:
            assert f1 is not None
            assert recall is not None
            assert precision is not None

        # Also set values if provided
        df = self.__all_tasks

        df.loc[df[self.primary_key] == task_id, "f1_score"] = f1
        df.loc[df[self.primary_key] == task_id, "recall"] = recall
        df.loc[df[self.primary_key] == task_id, "precision"] = precision
        df.loc[df[self.primary_key] == task_id, "timestamp"] = self._current_timestamp
        df.loc[df[self.primary_key] == task_id, "status"] = status.value

        df.to_csv(self.schedule_path, index=False)

    def _execute_task(self, task) -> bool:
        """
        Execute the given task and update its status in the schedule file.
        Returns True if the task completed successfully, False otherwise.
        """

        task_id = task[self.primary_key]
        print(f"Executing task {task_id}...")

        self._update_task_status(task_id, ExecutionStatus.RUNNING)

        try:
            args = get_config()

            for key in task.index:
                if hasattr(args, key):
                    setattr(args, key, task[key])

            executor = RealWorldPolicyExecutor(policy=ModelPolicy,
                                               args=args,
                                               robot=Robot.FullRobot,
                                               model_url=task["model_url"])
            run_details = executor.execute()
            # path = Path("data") / "songs" / "2nd_iteration_fences"
            # run_details.save_actions(path / f"{task_id}_actions.csv")
            # run_details.save_observations(path / f"{task_id}_observations_rw.csv")
            scores = run_details.scores
            self._update_task_status(task_id, ExecutionStatus.COMPLETED,
                                     f1=scores.f1, recall=scores.recall, precision=scores.precision)
            print(f"Task {task_id} completed successfully.")
            return True
        except KeyboardInterrupt:
            self._update_task_status(task_id, ExecutionStatus.INTERRUPTED)
            print(f"Task {task_id} was interrupted by user.")
            return False
        except Exception as e:
            self._update_task_status(task_id, ExecutionStatus.CRASHED)
            print(f"Task {task_id} failed with error: {e}")
            traceback.print_exc(file=sys.stdout)
            return False

    def execute_schedule(self, include_interrupted_runs):
        while True:
            task = self._get_next_task(include_interrupted_runs)
            if task is None:
                print("All tasks completed.")
                break
            if not self._execute_task(task):
                print("Stopping execution of schedule.")
                break


if __name__ == "__main__":
    path = Path("data") / "schedules"
    execution_scheduler = ExecutionScheduler(group="AllRunsRunId28588", overwrite=False)
    execution_scheduler.execute_schedule(False)
