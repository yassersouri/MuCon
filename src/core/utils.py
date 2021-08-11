import numpy as np
import torch
from fandak.utils.torch import tensor_to_numpy
from torch.nn.functional import interpolate


def make_same_size(
    prediction: np.ndarray, target: np.ndarray, background: int = 0
) -> np.ndarray:
    """
    Tries to use some heuristic to make the prediction the same size as the target.
    If the prediction is shorter, it will add background class at the end.
    If the prediction is longer, it will crop to the size of the target.
    :returns predictions. It will return the updated predictions file.
    """

    t_len = len(target)
    p_len = len(prediction)

    if p_len == t_len:
        return prediction
    elif p_len > t_len:
        new_predictions = prediction.copy()
        extra_len = p_len - t_len
        new_predictions = new_predictions[:-extra_len]
    else:  # p_len < t_len
        new_predictions = prediction.copy()
        remaining_len = t_len - p_len
        bg = np.full(remaining_len, fill_value=background)
        new_predictions = np.concatenate((new_predictions, bg), axis=0)
    return new_predictions


def make_same_size_interpolate(
    prediction: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """
    It will use nearest neighbor interpolation to make the prediction the same size as the target.
    """
    t_len = len(target)

    prediction_tensor = torch.tensor([[prediction]]).float()
    prediction_tensor_resized = interpolate(
        prediction_tensor, size=t_len, mode="nearest"
    )

    return tensor_to_numpy(prediction_tensor_resized[0][0].long())
