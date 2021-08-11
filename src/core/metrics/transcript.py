from typing import List

from edit_distance import SequenceMatcher
import numpy as np
from core.metrics import Metric


def calculate_matching_score(
    gt_transcript: List[int], predicted_transcript: List[int]
) -> float:
    return SequenceMatcher(a=gt_transcript, b=predicted_transcript).ratio()


def calculate_abs_len_diff(
    gt_transcript: List[int], predicted_transcript: List[int]
) -> int:
    return abs(len(predicted_transcript) - len(gt_transcript))


class MatchingScoreMetric(Metric):
    def __init__(self):
        self.reset()

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.values = []

    def add(
        self, target_transcript: List[int], predicted_transcript: List[int]
    ) -> float:
        current_score = calculate_matching_score(
            target_transcript, predicted_transcript
        )

        self.values.append(current_score)
        return current_score

    def summary(self) -> float:
        return np.array(self.values).mean()


class AbsLenDiffMetric(MatchingScoreMetric):
    def add(
        self, target_transcript: List[int], predicted_transcript: List[int]
    ) -> float:
        current_score = calculate_abs_len_diff(target_transcript, predicted_transcript)

        self.values.append(current_score)
        return current_score
