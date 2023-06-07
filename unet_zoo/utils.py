from collections.abc import Iterable

import scipy
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
    num_slices = 155 - 2 * skip_slices
    summary(model, input_size=(batch_size, NUM_SCANS_PER_EXAMPLE, num_slices, 240, 240))


def get_mask_middle(mask: torch.Tensor, middle_dim: int = 0) -> int:
    """Get the middle slice of a binary MRI mask."""
    if len(mask.shape) != 3:
        raise ValueError(f"Unexpected mask shape {mask.shape}.")
    first, last = None, None
    dims_for_max = tuple({0, 1, 2}.difference({middle_dim}))
    for i, slice_has_mask in enumerate(mask.amax(dim=dims_for_max)):
        if first is None and slice_has_mask > 0:
            first = i
        if first is not None and last is None and slice_has_mask == 0 and i > 0:
            last = i - 1
    return (first + last) // 2


TOP_TO_SIDE = {"angle": -90.0, "axes": (0, 2)}, {"angle": -90.0, "axes": (1, 2)}
TOP_TO_FRONT = ({"angle": -90.0, "axes": (0, 1)},)


def rotate(
    volume: torch.Tensor,
    angle: float,
    axes: tuple[int, int] = (1, 0),
    **rotate_kwargs,
) -> torch.Tensor:
    """Wrap scipy.ndimage.rotate for rotating an MRI."""
    orig_device = volume.device
    return torch.as_tensor(
        scipy.ndimage.rotate(volume.cpu(), angle, axes, **rotate_kwargs),
        device=orig_device,
    )


def multi_rotate(volume: torch.Tensor, multi_kwargs: Iterable[dict]) -> torch.Tensor:
    """Apply 0+ rotations in sequence."""
    for kwargs in multi_kwargs:
        volume = rotate(volume, **kwargs)
    return volume
