from typing import Any

import torch
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.utils import load_checkpoint

from unet_zoo import CHECKPOINTS_FOLDER
from unet_zoo.train import (
    INITIAL_CONV_OUT_CHANNELS,
    MASK_COUNT,
    NUM_GROUPS,
    NUM_SCANS_PER_EXAMPLE,
    get_train_val_data_loaders,
)
from unet_zoo.utils import infer_device

LAST_MODEL = CHECKPOINTS_FOLDER / "last_checkpoint.pytorch"
BEST_MODEL = CHECKPOINTS_FOLDER / "best_checkpoint.pytorch"
THRESHOLD = 0.33


def main() -> None:
    model = UNet3D(
        in_channels=NUM_SCANS_PER_EXAMPLE,
        out_channels=MASK_COUNT,
        final_sigmoid=True,
        f_maps=INITIAL_CONV_OUT_CHANNELS,
        num_groups=NUM_GROUPS,
    ).to(device=infer_device())
    state_dict: dict[str, Any] = load_checkpoint(BEST_MODEL, model)  # noqa: F841

    model.eval()
    data_loaders = get_train_val_data_loaders()
    for images, targets in data_loaders["val"]:
        wt_labels, tc_labels, et_labels = (targets[0][i] for i in range(3))
        with torch.no_grad():
            wt_probs, tc_probs, et_probs = (
                (model(images)[0] > THRESHOLD)[i] for i in range(3)
            )


if __name__ == "__main__":
    main()
