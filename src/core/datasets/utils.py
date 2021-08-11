from typing import Tuple, List, Any, Iterable

import numpy as np


def create_tf_input(transcript: Iterable[int], sos_i: int) -> np.ndarray:
    return np.array([sos_i] + list(transcript))


def create_tf_target(transcript: Iterable[int], eos_i: int) -> np.ndarray:
    return np.array(list(transcript) + [eos_i])


def summarize_list(the_list: List[Any]) -> Tuple[List[Any], List[int]]:
    """
    Given a list of items, it summarizes them in a way that no two neighboring values are the same.
    It also returns the size of each section.
    e.g. [4, 5, 5, 6] -> [4, 5, 6], [1, 2, 1]
    """
    summary = []
    lens = []
    if len(the_list) > 0:
        current = the_list[0]
        summary.append(current)
        lens.append(1)
        for item in the_list[1:]:
            if item != current:
                current = item
                summary.append(item)
                lens.append(1)
            else:
                lens[-1] += 1
    return summary, lens


def unsummarize_list(labels: List[int], lengths: List[int]) -> List[int]:
    """
    Does the reverse of summarize list. You give it a list of segment labels and their lengths and it returns the full
    labels for the full sequence.
    e.g. ([4, 5, 6], [1, 2, 1]) -> [4, 5, 5, 6]
    """
    assert len(labels) == len(lengths)

    the_sequence = []
    for label, length in zip(labels, lengths):
        the_sequence.extend([label] * length)

    return the_sequence


def segment_to_labels(transcript, lengths) -> np.ndarray:
    """
    Converts a segment-level representation of the action segmentation to frame-level
    representation. Segment-level representation is the transcript and the length of
    each action in the transcript.
    :param transcript: [M], int
    :param lengths: [M], int
    :return: numpy array
    """
    transcript, lengths = np.asarray(transcript), np.asarray(lengths)
    labels = np.repeat(transcript, lengths)
    return labels
