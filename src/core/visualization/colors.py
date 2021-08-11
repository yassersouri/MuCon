import random
from colorsys import hsv_to_rgb

import numpy as np


def generate_n_colors(n: int, shuffle: bool = False) -> np.ndarray:
    assert n > 0

    list_of_colors = []
    sat = 0.4
    val = 0.7
    for i in range(n):
        hue = i / n
        list_of_colors.append(list([x * 255 for x in hsv_to_rgb(hue, sat, val)]))

    if shuffle:
        random.seed(0)
        random.shuffle(list_of_colors)

    return np.array(list_of_colors)
