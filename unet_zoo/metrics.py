import torch
from pytorch3dunet.unet3d.metrics import MeanIoU as _MeanIoU


class MeanIoU(_MeanIoU):
    """IoU metric that has its internal binarization defeated by default."""

    def __init__(
        self,
        skip_channels=(),
        ignore_index=None,
        binarize: bool = False,
        **kwargs,
    ):
        super().__init__(skip_channels, ignore_index, **kwargs)
        self.binarize = binarize

    def _binarize_predictions(
        self,
        input: torch.Tensor,  # noqa: A002
        n_classes: int,
    ) -> torch.Tensor:
        if self.binarize:
            return super()._binarize_predictions(input, n_classes)
        return input
