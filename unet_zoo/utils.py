from collections import deque
from typing import TypeVar

import numpy as np
import torch
from torchinfo import summary

from data.loaders import NUM_SCANS_PER_EXAMPLE


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # Current CUDA device
    return torch.device("cpu")


def print_model_summary(
    model: torch.nn.Module,
    skip_slices: int = 0,
    batch_size: int = 1,
) -> None:
    summary(model, input_size=(batch_size, NUM_SCANS_PER_EXAMPLE, 240, 240))


def get_mask_middle(
    mask: torch.Tensor,
    middle_dim: int = 0,
    window_size: int = 3,
) -> int:
    """Get the middle slice of a binary MRI mask."""
    if len(mask.shape) != 3:
        raise ValueError(f"Unexpected mask shape {mask.shape}.")
    first, last, prior_max = None, None, 0
    dims_for_max = tuple({0, 1, 2}.difference({middle_dim}))
    window = deque(maxlen=window_size)
    for i, slice_has_mask in enumerate(mask.amax(dim=dims_for_max)):
        window.append(slice_has_mask)
        if len(window) < window_size:  # Fill the window before analyzing
            continue
        if first is None and min(window) > 0:
            first = i
        if first is not None and prior_max > 0 and max(window) == 0 and i > 0:
            last = i - 1
        prior_max = max(window)
    return (first + last) // 2


TArr = TypeVar("TArr", np.ndarray, torch.Tensor)


def get_arbitrary_element(a: TArr, element: int, dim: int = 0) -> TArr:
    """
    Get the item at an element along a dim.

    TODO: figure out how to do this in numpy or torch natively.

    Something like a[:, ..., element, ..., :], where element's position is
    controlled by dim.
    """
    return a[(*(slice(None) for _ in range(dim)), element)]
