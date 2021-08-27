from typing import Iterable

import numpy as np

from core.metrics import Metric
from core.metrics.isba_code import IoD, IoU


def careful_divide(correct: int, total: int, zero_value: float = 0.0) -> float:
    if total == 0:
        return zero_value
    else:
        return correct / total


class MoFAccuracyMetric(Metric):
    def __init__(self, ignore_ids: Iterable[int] = ()):
        self.ignore_ids = ignore_ids

        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.total = 0
        self.correct = 0

    def add(self, targets, predictions) -> float:
        assert len(targets) == len(predictions)
        targets, predictions = np.array(targets), np.array(predictions)

        mask = np.logical_not(np.isin(targets, self.ignore_ids))
        targets, predictions = targets[mask], predictions[mask]

        current_total = len(targets)
        current_correct = (targets == predictions).sum()
        current_result = careful_divide(current_correct, current_total)

        self.correct += current_correct
        self.total += current_total

        return current_result

    def summary(self) -> float:
        return careful_divide(self.correct, self.total)


class MoFAccuracyFromLogitsMetric(MoFAccuracyMetric):
    def add(self, targets, logits) -> float:
        """
        Here we assume the predictions are logits of shape [N x C]
        It can be torch or numpy array.

        N: number of predictions
        C: number of classes

        Implementation is simple, first convert logits to classes,
        then call parent class.
        """

        prediction = logits.argmax(-1)
        return super().add(targets, prediction)


class IoDMetric(Metric):
    def __init__(self, ignore_ids: Iterable[int] = ()):
        self.ignore_ids = ignore_ids
        self.calculation_function = IoD
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.values = []

    def add(self, targets, predictions) -> float:
        assert len(targets) == len(predictions)
        targets, predictions = np.array(targets), np.array(predictions)
        result = self.calculation_function(predictions, targets, self.ignore_ids)
        self.values.append(result)
        return result

    def summary(self) -> float:
        if len(self.values) > 0:
            return sum(self.values) / len(self.values)
        else:
            return 0.0


class IoUMetric(IoDMetric):
    def __init__(self, ignore_ids: Iterable[int] = ()):
        super().__init__(ignore_ids=ignore_ids)
        self.calculation_function = IoU
