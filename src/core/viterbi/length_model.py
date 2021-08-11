#!/usr/bin/python2.7

from typing import List

import numpy as np


class LengthModel(object):
    def n_classes(self):
        return 0

    def score(self, length, label):
        return 0.0

    def max_length(self):
        return np.inf


class MeanLengthModel(LengthModel):
    def __init__(self, num_classes, max_length=2000, threshold=200.0, alpha=0.9):
        super(MeanLengthModel, self).__init__()
        self.num_classes = num_classes
        self.max_len = max_length
        self.threshold = threshold
        self.alpha = alpha

    def n_classes(self):
        return self.num_classes

    def score(self, length, label):
        if length <= self.threshold:
            return 0.0
        elif length > self.max_len:
            return -np.inf
        else:
            return (length - self.threshold) * np.log(self.alpha)

    def max_length(self):
        return self.max_len


class PoissonModel(LengthModel):
    def __init__(self, model, max_length=2000, renormalize=True):
        super(PoissonModel, self).__init__()
        if type(model) == str:
            self.mean_lengths = np.loadtxt(model)
        else:
            self.mean_lengths = model
        self.num_classes = self.mean_lengths.shape[0]
        self.max_len = max_length
        self.poisson = np.zeros((max_length, self.num_classes))

        # precompute normalizations for mean length model
        self.norms = np.zeros(self.mean_lengths.shape)
        if renormalize:
            self.norms = np.round(self.mean_lengths) * np.log(
                np.round(self.mean_lengths)
            ) - np.round(self.mean_lengths)
            for c in range(len(self.mean_lengths)):
                logFak = 0
                for k in range(2, int(self.mean_lengths[c]) + 1):
                    logFak += np.log(k)
                self.norms[c] = self.norms[c] - logFak
        # precompute Poisson distribution
        self.poisson[0, :] = -np.inf  # length zero can not happen
        logFak = 0
        for l in range(1, self.max_len):
            logFak += np.log(l)
            self.poisson[l, :] = (
                l * np.log(self.mean_lengths) - self.mean_lengths - logFak - self.norms
            )

    def n_classes(self):
        return self.num_classes

    def score(self, length, label):
        if length >= self.max_len:
            return -np.inf
        else:
            return self.poisson[length, label]

    def max_length(self):
        return self.max_len


class MultiPoissonModel(LengthModel):
    def __init__(self, list_of_lengths: List[float], num_classes: int):
        """
        :param list_of_lengths: The list of lengths.
        :param num_classes: number of classes
        """
        self.num_classes = num_classes
        self.poisson_models = []
        for i in range(len(list_of_lengths)):
            self.poisson_models.append(
                PoissonModel(
                    np.full(self.num_classes, list_of_lengths[i], dtype=np.float32)
                )
            )

    def n_classes(self):
        return self.num_classes

    def max_length(self):
        return self.poisson_models[0].max_len

    def score(self, length, label):
        raise NotImplementedError

    def score_multi(self, index, length, label):
        return self.poisson_models[index].score(length, label)
