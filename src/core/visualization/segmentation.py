from typing import List, Dict

import cv2
import numpy as np


def generate_image_for_segmentation(
    labels: List[int],
    lengths: List[int],
    colors: np.ndarray,
    label_name_mapping: Dict[int, str] = None,
    white_label: int = 0,
    split_width: int = 5,
    height: int = 50,
) -> np.ndarray:
    # todo: make the image size fixed and not a function of the number os splits.
    num_splits = 1 + len(labels)

    width = num_splits * split_width + sum(lengths)
    result = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    for i in range(len(labels)):
        start = (i + 1) * split_width + sum(lengths[:i])
        end = start + lengths[i] + 1
        color = colors[labels[i]]

        if labels[i] == white_label:
            color = np.full(3, fill_value=255, dtype=np.uint8)
        result[:, start:end, :] = color

        if label_name_mapping is not None:
            pos_x = start + 2
            pos_y = int(height / 2)
            cv2.putText(
                result,
                label_name_mapping[labels[i]],
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 0, 0),
            )

    return result
