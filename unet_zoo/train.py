from typing import Literal

import torch
from pytorch3dunet.unet3d.losses import BCEDiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.utils import DefaultTensorboardFormatter
from torch.utils.data import DataLoader
from torchinfo import summary

from data.loaders import (
    TRAIN_VAL_DS_KWARGS,
    BraTS2020MRIScansDataset,
    BraTS2020MRISlicesDataset,
    split_train_val,
)
from unet_zoo import CHECKPOINTS_FOLDER
from unet_zoo.utils import infer_device

NUM_SCANS_PER_EXAMPLE = len(BraTS2020MRIScansDataset.NONMASK_EXTENSIONS)
MASK_COUNT = 3  # WT, TC, ET
INITIAL_CONV_OUT_CHANNELS = 18
NUM_GROUPS = 9
SKIP_SLICES = 5
BATCH_SIZE = 1
NUM_EPOCHS = 10


def print_summary(
    model: torch.nn.Module,
    skip_slices: int = SKIP_SLICES,
    batch_size: int = BATCH_SIZE,
) -> None:
    num_slices = 155 - 2 * skip_slices
    summary(model, input_size=(batch_size, NUM_SCANS_PER_EXAMPLE, num_slices, 240, 240))


def get_train_val_datasets(
    skip_slices: int = SKIP_SLICES,
    batch_size: int = BATCH_SIZE,
) -> tuple[BraTS2020MRIScansDataset, BraTS2020MRIScansDataset]:
    train_val_ds = BraTS2020MRIScansDataset(
        device=infer_device(),
        skip_slices=skip_slices,
        **TRAIN_VAL_DS_KWARGS,
    )
    return split_train_val(train_val_ds, batch_size=batch_size)


def main() -> None:
    model = UNet2D(
        in_channels=NUM_SCANS_PER_EXAMPLE,
        out_channels=MASK_COUNT,
        final_sigmoid=True,
        f_maps=INITIAL_CONV_OUT_CHANNELS,
        num_groups=NUM_GROUPS,
    ).to(device=infer_device())
    # print_summary(model)

    data_loaders: dict[Literal["train", "val"], DataLoader] = {
        name: BraTS2020MRISlicesDataset(scans_ds=ds)
        for name, ds in zip(("train", "val"), get_train_val_datasets(), strict=True)
    }
    defeat_max_num_iters = (
        NUM_EPOCHS * max(len(data_loaders[split]) for split in data_loaders) + 1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    trainer = UNetTrainer(
        model,
        optimizer=optimizer,
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        loss_criterion=BCEDiceLoss(alpha=1.0, beta=1.0),
        eval_criterion=MeanIoU(),
        loaders=data_loaders,
        checkpoint_dir=str(CHECKPOINTS_FOLDER),
        max_num_epochs=NUM_EPOCHS,
        max_num_iterations=defeat_max_num_iters,
        tensorboard_formatter=DefaultTensorboardFormatter(),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
