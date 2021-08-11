#!/usr/bin/python3

import numpy as np


class Grammar(object):

    # @context: tuple containing the previous label indices
    # @label: the current label index
    # @return: the log probability of label given context p(label|context)
    def score(self, context, label):  # score is a log probability
        return 0.0

    # @return: the number of classes
    def n_classes(self):
        return 0

    # @return sequence start symbol
    def start_symbol(self):
        return -1

    # @return sequence end symbol
    def end_symbol(self):
        return -2

    # @context: tuple containing the previous label indices
    # @return: list of all possible successor labels for the given context
    def possible_successors(self, context):
        return set()

    # @context: tuple containing the previous label indices
    # @label: new label index
    # @return: the combined old context and new label
    def update_context(self, context, label):
        return context + (label,)


# n-gram with linear discounting
# ngram-order needs to be at least one (unigram)
class NGram(Grammar):
    def __init__(self, transcript_file, label2index_map, ngram_order):
        assert ngram_order >= 1
        self.ngram_order = ngram_order
        self.num_classes = len(label2index_map)
        self.ngrams, self.vocabulary = self._get_statistics(
            transcript_file, label2index_map
        )
        self.lambdas = self._precompute_lambdas()
        self._precompute_normalizations()

    def _get_statistics(self, transcript_file, label2index_map):
        ngrams = dict()
        vocabulary = set()
        with open(transcript_file, "r") as f:
            lines = f.read().split("\n")[0:-1]
        for line in lines:
            labels = (
                [self.start_symbol()]
                + [label2index_map[label] for label in line.split()]
                + [self.end_symbol()]
            )
            for pos, label in enumerate(labels):
                vocabulary.add(label)
                ngrams[()] = ngrams.get((), 0) + 1
                for order in range(self.ngram_order):
                    context = tuple(labels[max(0, pos - order) : pos + 1])
                    ngrams[context] = ngrams.get(context, 0) + 1
        vocabulary.remove(self.start_symbol())
        return ngrams, vocabulary

    def _precompute_normalizations(self):
        self.normalization = dict()
        for order in range(1, self.ngram_order):
            for key in self.ngrams:
                if len(key) == order + 1:
                    context = tuple(key[:-1])
                    for w in self.vocabulary:
                        if not context + (w,) in self.ngrams:
                            h = tuple(context[0:-1])
                            self.normalization[key] = self.normalization.get(
                                key, 0
                            ) + self._probability(h, w)

    def _precompute_lambdas(self):
        lambdas = [0] * self.ngram_order
        counts = [0] * self.ngram_order
        for context in self.ngrams:
            order = len(context) - 1
            if order >= 0:
                lambdas[order] += 1 if self.ngrams[context] == 1 else 0
                counts[order] += self.ngrams[context]
        for i, c in enumerate(counts):
            lambdas[i] /= max(c, 1)
        return lambdas

    def _probability(self, context, label):
        if context + (label,) in self.ngrams:
            p = self.ngrams[context + (label,)] / self.ngrams[context]
            p = p * (1 - self.lambdas[len(context)])
        else:
            p = self._probability(
                tuple(context[:-1]), context[-1]
            ) / self.normalization.get(context + (label,), 1)
            p = p * self.lambdas[len(context)]
        return p

    def perplexity(self, transcript_file, label2index_map):
        log_pp = 0
        N = 0
        with open(transcript_file, "r") as f:
            lines = f.read().split("\n")[0:-1]
            for line in lines:
                labels = (
                    [self.start_symbol()]
                    + [label2index_map[label] for label in line.split()]
                    + [self.end_symbol()]
                )
                for i, label in enumerate(labels):
                    context = tuple(labels[max(0, i - self.ngram_order + 1) : i])
                    log_pp += self.score(context, label)
                    N += 1
        return np.exp(-log_pp / N)

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.vocabulary

    def score(self, context, label):
        return np.log(self._probability(context, label))

    def update_context(self, context, label):
        context = context + (label,)
        if self.ngram_order == 1:
            return ()
        else:
            return tuple(context[-self.ngram_order + 1 :])


# grammar containing all action transcripts seen in training
# used for inference
class PathGrammar(Grammar):
    def __init__(self, transcript_file, label2index_map):
        self.num_classes = len(label2index_map)
        transcripts = self._read_transcripts(transcript_file, label2index_map)
        # generate successor sets
        self.successors = dict()
        for transcript in transcripts:
            transcript = transcript + [self.end_symbol()]
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                self.successors[context] = {transcript[i]}.union(
                    self.successors.get(context, set())
                )

    def _read_transcripts(self, transcript_file, label2index_map):
        transcripts = []
        with open(transcript_file, "r") as f:
            lines = f.read().split("\n")[0:-1]
        for line in lines:
            transcripts.append([label2index_map[label] for label in line.split()])
        return transcripts

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf


class ModifiedPathGrammar(PathGrammar):
    def __init__(self, transcripts, num_classes):
        # self.num_classes = len(label2index_map)
        self.num_classes = num_classes
        # transcripts = self._read_transcripts(transcript_file, label2index_map)
        # generate successor sets
        self.successors = dict()
        for transcript in transcripts:
            transcript = transcript + [self.end_symbol()]
            for i in range(len(transcript)):
                context = (self.start_symbol(),) + tuple(transcript[0:i])
                self.successors[context] = set([transcript[i]]).union(
                    self.successors.get(context, set())
                )


# grammar that generates only a single transcript
# use during training to align frames to transcript
class SingleTranscriptGrammar(Grammar):
    def __init__(self, transcript, n_classes):
        self.num_classes = n_classes
        transcript = transcript + [self.end_symbol()]
        self.successors = dict()
        for i in range(len(transcript)):
            context = (self.start_symbol(),) + tuple(transcript[0:i])
            self.successors[context] = {transcript[i]}.union(
                self.successors.get(context, set())
            )

    def n_classes(self):
        return self.num_classes

    def possible_successors(self, context):
        return self.successors.get(context, set())

    def score(self, context, label):
        if label in self.possible_successors(context):
            return 0.0
        else:
            return -np.inf
