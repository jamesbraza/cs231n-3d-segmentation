from typing import Literal

import torch
from pytorch3dunet.unet3d.losses import BCEDiceLoss
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import UNet2D
from pytorch3dunet.unet3d.trainer import UNetTrainer
from torch.utils.data import DataLoader

from data.loaders import (
    MASK_COUNT,
    NUM_SCANS_PER_EXAMPLE,
    TRAIN_VAL_DS_KWARGS,
    BraTS2020MRIScansDataset,
    BraTS2020MRISlicesDataset,
    make_generator,
)
from unet_zoo import CHECKPOINTS_FOLDER
from unet_zoo.utils import infer_device, print_model_summary  # noqa: F401

INITIAL_CONV_OUT_CHANNELS = 18
NUM_GROUPS = 9
SKIP_SLICES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 10
GENERATOR = make_generator(42)


def get_train_val_test_scans_datasets(
    skip_slices: int = SKIP_SLICES,
    generator: torch.Generator | None = GENERATOR,
) -> tuple[
    BraTS2020MRIScansDataset,
    BraTS2020MRIScansDataset,
    BraTS2020MRIScansDataset,
]:
    train_val_ds = BraTS2020MRIScansDataset(
        device=infer_device(),
        skip_slices=skip_slices,
        **TRAIN_VAL_DS_KWARGS,
    )
    if generator is None:
        generator = torch.default_generator
    return tuple(
        torch.utils.data.random_split(
            train_val_ds,
            lengths=(0.8, 0.1, 0.1),
            generator=generator,
        ),
    )


def scans_to_slices_data_loader(
    ds: BraTS2020MRIScansDataset,
    batch_size: int = BATCH_SIZE,
    insert_z_dim: bool = True,
) -> DataLoader:
    """Convert a BraTS MRI scans dataset into a data loader of MRI slices."""
    return DataLoader(
        BraTS2020MRISlicesDataset(scans_ds=ds, insert_z_dim=insert_z_dim),
        batch_size=batch_size,
    )


def main() -> None:
    model = UNet2D(
        in_channels=NUM_SCANS_PER_EXAMPLE,
        out_channels=MASK_COUNT,
        final_sigmoid=True,
        f_maps=INITIAL_CONV_OUT_CHANNELS,
        num_groups=NUM_GROUPS,
    ).to(device=infer_device())
    # print_model_summary(model)

    data_loaders: dict[Literal["train", "val", "test"], DataLoader] = {
        name: scans_to_slices_data_loader(ds)
        for name, ds in zip(
            ("train", "val", "test"),
            get_train_val_test_scans_datasets(),
            strict=True,
        )
    }
    defeat_max_num_iters = (
        NUM_EPOCHS * max(len(data_loaders[split]) for split in data_loaders) + 1
    )
    log_after_iters = int(defeat_max_num_iters / 20)

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
        validate_after_iters=2 * log_after_iters,
        log_after_iters=log_after_iters,
        tensorboard_formatter=BraTS2020MRISlicesDataset.get_tensorboard_formatter(),
    )
    trainer.fit()


if __name__ == "__main__":
    main()
