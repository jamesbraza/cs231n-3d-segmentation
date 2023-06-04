from typing import Literal

import torch
from pytorch3dunet.unet3d.losses import BCEDiceLoss
from pytorch3dunet.unet3d.model import UNet3D
from pytorch3dunet.unet3d.trainer import UNetTrainer
from pytorch3dunet.unet3d.utils import DefaultTensorboardFormatter
from torch.utils.data import DataLoader

from data.loaders import TRAIN_DS_KWARGS, VAL_DS_KWARGS, BraTS2020Dataset
from unet_zoo import ZOO_FOLDER

NUM_SCANS_PER_EXAMPLE = len(BraTS2020Dataset.NONMASK_EXTENSIONS)
MASK_COUNT = 3  # WT, TC, ET
INITIAL_CONV_OUT_CHANNELS = 24

model = UNet3D(
    in_channels=NUM_SCANS_PER_EXAMPLE,
    out_channels=MASK_COUNT,
    final_sigmoid=True,
    f_maps=INITIAL_CONV_OUT_CHANNELS,
    num_groups=6,
)
data_loaders: dict[Literal["train", "val"], DataLoader] = {
    "train": DataLoader(BraTS2020Dataset(skip_slices=5, **TRAIN_DS_KWARGS)),
    "val": DataLoader(BraTS2020Dataset(skip_slices=5, **VAL_DS_KWARGS)),
}
trainer = UNetTrainer(
    model,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-4),
    lr_scheduler=None,
    loss_criterion=BCEDiceLoss(alpha=1.0, beta=1.0),
    eval_criterion=None,
    loaders=data_loaders,
    checkpoint_dir=str(ZOO_FOLDER / "logs"),
    max_num_epochs=5,
    max_num_iterations=1000,  # Defeat max number of batches per epoch
    tensorboard_formatter=DefaultTensorboardFormatter(),
    skip_train_validation=True,
)
trainer.fit()
