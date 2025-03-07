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

"""A wrapper for tracking episode statistics pertaining to music performance.

TODO(kevin):
- Look into `mir_eval` for metrics.
"""

from collections import deque
from typing import Deque, Dict, List, NamedTuple, Sequence

import dm_env
import numpy as np
from dm_env_wrappers import EnvironmentWrapper
from sklearn.metrics import precision_recall_fscore_support


class EpisodeMetrics(NamedTuple):
    """A container for storing episode metrics."""

    precision: float
    recall: float
    f1: float


class MidiEvaluationWrapper(EnvironmentWrapper):
    """Track metrics related to musical performance.

    This wrapper calculates the precision, recall, and F1 score of the last `deque_size`
    episodes. The mean precision, recall and F1 score can be retrieved using
    `get_musical_metrics()`.

    By default, `deque_size` is set to 1 which means that only the current episode's
    statistics are tracked.
    """

    def __init__(self, environment: dm_env.Environment, deque_size: int = 1) -> None:
        super().__init__(environment)

        self._key_presses: List[np.ndarray] = []

        # Key press metrics.
        self._key_press_precisions: Deque[float] = deque(maxlen=deque_size)
        self._key_press_recalls: Deque[float] = deque(maxlen=deque_size)
        self._key_press_f1s: Deque[float] = deque(maxlen=deque_size)

    def step(self, action: np.ndarray) -> dm_env.TimeStep:
        timestep = self._environment.step(action)

        key_activation = self._environment.task.piano.activation
        self._key_presses.append(key_activation.astype(np.float64))

        if timestep.last():
            failure_termination = False
            if hasattr(self._environment.task, "_wrong_press_termination"):
                failure_termination = self._environment.task._wrong_press_termination
            key_press_metrics = compute_key_press_metrics(note_seq=self._environment.task._notes,
                                                          n_keys=self._environment.task.piano.n_keys,
                                                          key_presses=self._key_presses,
                                                          failure_termination=failure_termination)
            self._key_press_precisions.append(key_press_metrics.precision)
            self._key_press_recalls.append(key_press_metrics.recall)
            self._key_press_f1s.append(key_press_metrics.f1)

            self._key_presses = []
        return timestep

    def reset(self) -> dm_env.TimeStep:
        self._key_presses = []
        return self._environment.reset()

    def get_musical_metrics(self) -> Dict[str, float]:
        """Returns the mean precision/recall/F1 over the last `deque_size` episodes."""
        if not self._key_press_precisions:
            raise ValueError("No episode metrics available yet.")

        def _mean(seq: Sequence[float]) -> float:
            return sum(seq) / len(seq)

        return {
            "precision": _mean(self._key_press_precisions),
            "recall": _mean(self._key_press_recalls),
            "f1": _mean(self._key_press_f1s),
        }

    # Helper methods.


def compute_key_press_metrics(note_seq, n_keys, key_presses, failure_termination=False) -> EpisodeMetrics:
    """Computes precision/recall/F1 for key presses over the episode."""
    # Get the ground truth key presses.
    ground_truth = []
    for notes in note_seq:
        presses = np.zeros((n_keys,), dtype=np.float64)
        keys = [note.key for note in notes]
        presses[keys] = 1.0
        ground_truth.append(presses)

    # Deal with the case where the episode gets truncated due to a failure. In this
    # case, the length of the key presses will be less than or equal to the length
    # of the ground truth.
    if failure_termination:
        ground_truth = ground_truth[: len(key_presses)]

    assert len(ground_truth) == len(key_presses)

    precisions = []
    recalls = []
    f1s = []
    for y_true, y_pred in zip(ground_truth, key_presses):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="binary", zero_division=1
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)

    return EpisodeMetrics(precision, recall, f1)
