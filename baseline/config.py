import os
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import torch

from baseline import BASELINE_FOLDER, LOGS_FOLDER
from data import (
    BRATS_2020_TRAINING_FOLDER,
    BRATS_2020_TRAINING_VALIDATION_DATASET_FOLDER,
    BRATS_2020_VALIDATION_FOLDER,
)

PathLike: TypeAlias = str | os.PathLike


@dataclass
class GlobalConfig:
    root_dir: PathLike = BRATS_2020_TRAINING_VALIDATION_DATASET_FOLDER
    train_root_dir: PathLike = BRATS_2020_TRAINING_FOLDER
    test_root_dir: PathLike = BRATS_2020_VALIDATION_FOLDER
    path_to_csv: PathLike = BASELINE_FOLDER / "train_data.csv"
    pretrained_model_path: PathLike | None = BASELINE_FOLDER / "last_epoch_model.pth"
    train_logs_path: PathLike | None = LOGS_FOLDER / "unet/train_log.csv"
    ae_pretrained_model_path: PathLike | None = (
        BASELINE_FOLDER / "autoencoder_best_model.pth"
    )
    tab_data: PathLike = (
        LOGS_FOLDER / "data/df_with_voxel_stats_and_latent_features.csv"
    )
    seed: int = 55


def seed_everything(seed: int):
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


config = GlobalConfig()
seed_everything(config.seed)
