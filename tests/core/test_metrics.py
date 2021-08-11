from unittest import TestCase

import numpy as np
import torch

from core.metrics.segmentation import (
    MoFAccuracyMetric,
    MoFAccuracyFromLogitsMetric,
    IoDMetric,
    # IoUMetric,
)


def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def IoU(P, Y, bg_class=None, **kwargs):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def overlap_(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array(
                [t for t, l in zip(true_intervals, true_labels) if l != bg_class]
            )
            true_labels = np.array([l for l in true_labels if l != bg_class])
            pred_intervals = np.array(
                [t for t, l in zip(pred_intervals, pred_labels) if l != bg_class]
            )
            pred_labels = np.array([l for l in pred_labels if l != bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(
                        pred_intervals[j][1], true_intervals[i][1]
                    ) - max(pred_intervals[j][0], true_intervals[i][0])
                    union = max(pred_intervals[j][1], true_intervals[i][1]) - min(
                        pred_intervals[j][0], true_intervals[i][0]
                    )
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean() * 100

    if type(P) == list:
        return np.mean([overlap_(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_(P, Y, bg_class)


def IoD(P, Y, bg_class=None, **kwargs):
    # From ICRA paper:
    # Learning Convolutional Action Primitives for Fine-grained Action Recognition
    # Colin Lea, Rene Vidal, Greg Hager
    # ICRA 2016

    def overlap_d(p, y, bg_class):
        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = np.array(
                [t for t, l in zip(true_intervals, true_labels) if l != bg_class]
            )
            true_labels = np.array([l for l in true_labels if l != bg_class])
            pred_intervals = np.array(
                [t for t, l in zip(pred_intervals, pred_labels) if l != bg_class]
            )
            pred_labels = np.array([l for l in pred_labels if l != bg_class])

        n_true_segs = true_labels.shape[0]
        n_pred_segs = pred_labels.shape[0]
        seg_scores = np.zeros(n_true_segs, np.float)

        for i in range(n_true_segs):
            for j in range(n_pred_segs):
                if true_labels[i] == pred_labels[j]:
                    intersection = min(
                        pred_intervals[j][1], true_intervals[i][1]
                    ) - max(pred_intervals[j][0], true_intervals[i][0])
                    union = pred_intervals[j][1] - pred_intervals[j][0]
                    score_ = float(intersection) / union
                    seg_scores[i] = max(seg_scores[i], score_)

        return seg_scores.mean() * 100

    if type(P) == list:
        return np.mean([overlap_d(P[i], Y[i], bg_class) for i in range(len(P))])
    else:
        return overlap_d(P, Y, bg_class)


class TestISBAIoDMetric(TestCase):
    def test_simple(self):
        metric = IoDMetric(ignore_ids=[0])
        self.assertEqual(metric.summary(), 0.0)

        pred = [[0, 0, 0, 1, 0, 1, 1, 1, 0]]
        targ = [[0, 0, 1, 1, 2, 1, 1, 0, 0]]

        # pred = [[1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]]
        # targ = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]

        print(f"ISBA IoD: {IoD(pred, targ, bg_class=0)}")
        print(f"ISBA IoU: {IoU(pred, targ, bg_class=0)}")
        self.assertEqual(metric.add(targ[0], pred[0]), IoD(pred, targ, bg_class=0))

        pred = [[1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0]]
        targ = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]

        print(f"ISBA IoD: {IoD(pred, targ, bg_class=0)}")
        print(f"ISBA IoU: {IoU(pred, targ, bg_class=0)}")
        self.assertEqual(metric.add(targ[0], pred[0]), IoD(pred, targ, bg_class=0))

        pred = [
            [0, 0, 0, 1, 0, 1, 1, 1, 0],
            [1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0],
        ]
        targ = [
            [0, 0, 1, 1, 2, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3],
        ]

        print(f"ISBA IoD: {IoD(pred, targ, bg_class=0)}")
        print(f"ISBA IoU: {IoU(pred, targ, bg_class=0)}")
        self.assertEqual(metric.summary(), IoD(pred, targ, bg_class=0))


class TestAccuracyMetric(TestCase):
    def test_simple(self):
        metric = MoFAccuracyMetric()
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        pred = [0, 0, 1, 1, 2, 2, 0, 0]

        self.assertEqual(metric(targ, pred), 6 / 8)

        targ = [0, 1, 0, 0, 2, 2, 0, 0]
        pred = [1, 1, 1, 2, 2, 2, 2, 2]

        self.assertEqual(metric(targ, pred), 3 / 8)

        self.assertEqual(metric.summary(), 9 / 16)

        metric.reset()
        self.assertEqual(metric.summary(), 0.0)

    def test_numpy_array(self):
        metric = MoFAccuracyMetric()
        targ = np.array([0, 1, 1, 0, 2, 2, 0, 0])
        pred = np.array([0, 0, 1, 1, 2, 2, 0, 0])

        self.assertEqual(metric(targ, pred), 6 / 8)

    def test_torch_array(self):
        metric = MoFAccuracyMetric()
        targ = torch.tensor([0, 1, 1, 0, 2, 2, 0, 0])
        pred = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0])

        self.assertEqual(metric(targ, pred), 6 / 8)

    def test_ignore(self):
        metric = MoFAccuracyMetric(ignore_ids=[0])
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        pred = [0, 0, 1, 1, 2, 2, 0, 0]

        self.assertEqual(metric(targ, pred), 3 / 4)

        targ = [0, 1, 0, 0, 2, 2, 0, 0]
        pred = [1, 1, 1, 2, 2, 2, 2, 2]

        self.assertEqual(metric(targ, pred), 3 / 3)

        self.assertEqual(metric.summary(), 6 / 7)

    def test_ignore_multi(self):
        metric = MoFAccuracyMetric(ignore_ids=[0, 2])
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        pred = [0, 0, 1, 1, 2, 2, 0, 0]

        self.assertEqual(metric(targ, pred), 1 / 2)

        targ = [0, 1, 0, 0, 2, 2, 0, 0]
        pred = [1, 1, 1, 2, 2, 2, 2, 2]

        self.assertEqual(metric(targ, pred), 1 / 1)

        self.assertEqual(metric.summary(), 2 / 3)

    def test_ignore_all(self):
        metric = MoFAccuracyMetric(ignore_ids=[0, 1, 2])
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        pred = [0, 0, 1, 1, 2, 2, 0, 0]

        self.assertEqual(metric(targ, pred), 0.0)

        targ = [0, 1, 0, 0, 2, 2, 0, 0]
        pred = [1, 1, 1, 2, 2, 2, 2, 2]

        self.assertEqual(metric(targ, pred), 0.0)

        self.assertEqual(metric.summary(), 0.0)


class TestAccuracyFromLogitsMetric(TestCase):
    def test_simple_numpy(self):
        metric = MoFAccuracyFromLogitsMetric()
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        logi = [
            [0, -1, -3],  # 0
            [0, 2, 0.5],  # 1
            [1, 0, 0.5],  # 1
            [1, 0, -1],  # 0
            [0, 0, 2],  # 2
            [0, 0, 2],  # 2
            [1, 0, 0],  # 0
            [10, 0, 0],  # 0
        ]

        logi = np.array(logi)

        metric.add(targ, logi)
        self.assertEqual(metric.summary(), 7 / 8)

        metric.reset()
        self.assertEqual(metric.summary(), 0.0)

    def test_simple_torch(self):
        metric = MoFAccuracyFromLogitsMetric()
        self.assertEqual(metric.summary(), 0.0)

        targ = [0, 1, 1, 0, 2, 2, 0, 0]
        logi = [
            [0, -1, -3],  # 0
            [0, 2, 0.5],  # 1
            [1, 0, 0.5],  # 1
            [1, 0, -1],  # 0
            [0, 0, 2],  # 2
            [0, 0, 2],  # 2
            [1, 0, 0],  # 0
            [10, 0, 0],  # 0
        ]

        logi = torch.tensor(logi).float()

        metric.add(targ, logi)
        self.assertEqual(metric.summary(), 7 / 8)
