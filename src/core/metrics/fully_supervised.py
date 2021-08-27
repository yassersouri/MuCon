from typing import List, Iterable

import numpy as np

from . import Metric
from .mstcn_code import edit_score, f_score


class Edit(Metric):
    def __init__(self, ignore_ids: Iterable[int] = ()):
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideIniÒÒÒÒt
    def reset(self):
        self.values = []

    def add(self, targets: List[int], predictions: List[int]) -> float:
        current_score = edit_score(
            recognized=predictions,
            ground_truth=targets,
            bg_class=self.ignore_ids,
        )

        self.values.append(current_score)
        return current_score

    def summary(self) -> float:
        if len(self.values) > 0:
            return np.array(self.values).mean()
        else:
            return 0.0


class F1Score(Metric):
    def __init__(
        self,
        overlaps: List[float] = (0.1, 0.25, 0.5),
        ignore_ids: List[int] = (),
    ):
        self.overlaps = overlaps
        self.ignore_ids = ignore_ids
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.tp = [0.0] * len(self.overlaps)
        self.fp = [0.0] * len(self.overlaps)
        self.fn = [0.0] * len(self.overlaps)

    def add(self, targets: List[int], predictions: List[int]) -> List[float]:
        current_result = []

        for s in range(len(self.overlaps)):
            tp1, fp1, fn1 = f_score(
                predictions,
                targets,
                self.overlaps[s],
                bg_class=self.ignore_ids,
            )
            self.tp[s] += tp1
            self.fp[s] += fp1
            self.fn[s] += fn1

            current_f1 = self.get_f1_score(tp1, fp1, fn1)
            current_result.append(current_f1)

        return current_result

    def summary(self) -> List[float]:
        result = []

        for s in range(len(self.overlaps)):
            f1 = self.get_f1_score(tp=self.tp[s], fp=self.fp[s], fn=self.fn[s])
            result.append(f1)

        return result

    @staticmethod
    def get_f1_score(tp: float, fp: float, fn: float) -> float:
        if tp + fp != 0.0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        else:
            precision = 0.0
            recall = 0.0

        if precision + recall != 0.0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
            f1 = f1 * 100
        else:
            f1 = 0.0

        return f1
