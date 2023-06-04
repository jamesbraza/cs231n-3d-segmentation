from typing import Literal

import torch
from pytorch3dunet.unet3d.losses import BCEDiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.utils import DefaultTensorboardFormatter
from torch.utils.data import DataLoader
from torchinfo import summary

from data.loaders import TRAIN_VAL_DS_KWARGS, BraTS2020Dataset, split_train_val
from unet_zoo import ZOO_FOLDER

NUM_SCANS_PER_EXAMPLE = len(BraTS2020Dataset.NONMASK_EXTENSIONS)
MASK_COUNT = 3  # WT, TC, ET
INITIAL_CONV_OUT_CHANNELS = 12
SKIP_SLICES = 5
BATCH_SIZE = 1
TRAIN_VAL_SPLIT = 0.9  # * 100% training


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # Current CUDA device
    return torch.device("cpu")


def print_summary(
    model: torch.nn.Module,
    skip_slices: int = SKIP_SLICES,
    batch_size: int = BATCH_SIZE,
) -> None:
    num_slices = 155 - 2 * skip_slices
    summary(model, input_size=(batch_size, NUM_SCANS_PER_EXAMPLE, num_slices, 240, 240))


model = UNet3D(
    in_channels=NUM_SCANS_PER_EXAMPLE,
    out_channels=MASK_COUNT,
    final_sigmoid=True,
    f_maps=INITIAL_CONV_OUT_CHANNELS,
    num_groups=6,
).to(device=infer_device())
# print_summary(model)

train_val_ds = BraTS2020Dataset(
    device=infer_device(),
    skip_slices=SKIP_SLICES,
    **TRAIN_VAL_DS_KWARGS,
)
train_ds, val_ds = split_train_val(
    train_val_ds,
    batch_size=BATCH_SIZE,
    fraction=TRAIN_VAL_SPLIT,
)

data_loaders: dict[Literal["train", "val"], DataLoader] = {
    "train": DataLoader(train_ds, batch_size=BATCH_SIZE),
    "val": DataLoader(val_ds, batch_size=BATCH_SIZE),
}
defeat_max_num_iters = max(len(data_loaders[split]) for split in data_loaders) + 1

trainer = UNetTrainer(
    model,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
    lr_scheduler=None,
    loss_criterion=BCEDiceLoss(alpha=1.0, beta=1.0),
    eval_criterion=MeanIoU(),
    loaders=data_loaders,
    checkpoint_dir=str(ZOO_FOLDER),
    max_num_epochs=5,
    max_num_iterations=defeat_max_num_iters,
    tensorboard_formatter=DefaultTensorboardFormatter(),
)
trainer.fit()
