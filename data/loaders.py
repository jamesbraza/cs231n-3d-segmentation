import itertools
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import BRATS_2020_TRAINING_FOLDER, BRATS_2020_VALIDATION_FOLDER


class BraTS2020Dataset(Dataset):
    """Map-style dataset for BraTS 2020."""

    TARGET_COLUMN = "BraTS_2020_subject_ID"
    NONMASK_EXTENSIONS = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
    MASK_EXTENSION = "_seg.nii"

    def __init__(
        self,
        data_folder_path: os.PathLike | str,
        mapping_csv_name: str,
        device: torch.device | None = None,
        train: bool = True,
    ):
        self._data_folder_path = data_folder_path
        self._names = pd.read_csv(
            os.path.join(data_folder_path, mapping_csv_name),
            usecols=[self.TARGET_COLUMN],
            dtype=str,
        )
        self._device = device
        self.train = train

    def __len__(self) -> int:
        return len(self._names)

    def get_full_path(self, index: int, extension: str) -> str:
        image_folder = os.path.join(
            self._data_folder_path,
            self._names[self.TARGET_COLUMN][index],
        )
        return os.path.join(image_folder, os.path.basename(image_folder) + extension)

    WT, TC, ET = 1, 2, 4

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Get (images, masks) if training, otherwise (images,)."""
        raw_imgs = (
            np.asarray(nib.load(path).dataobj)
            for path in (
                self.get_full_path(index, extension)
                for extension in self.NONMASK_EXTENSIONS
            )
        )
        # Normalize to be in [0, 1]
        img = np.stack((img - img.min()) / (img.max() - img.min()) for img in raw_imgs)
        image_tensor = torch.as_tensor(
            # N x W x H x C to N x C x H x W
            np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1)),
            dtype=torch.get_default_dtype(),  # Match pytorch-3dunet internals
            device=self._device,
        )
        if not self.train:
            return (image_tensor,)
        mask = np.asarray(
            nib.load(self.get_full_path(index, self.MASK_EXTENSION)).dataobj,
        )
        wt = mask >= self.WT
        tc = np.logical_and(mask != self.TC, mask >= self.WT)
        et = mask >= self.ET
        return image_tensor, torch.as_tensor(
            # N x W x H x C to N x C x H x W
            np.moveaxis(np.stack((wt, tc, et)), (0, 1, 2, 3), (0, 3, 2, 1)),
            dtype=torch.get_default_dtype(),  # Match pytorch-3dunet internals
            device=self._device,
        )


TRAIN_DS_KWARGS = {
    "data_folder_path": BRATS_2020_TRAINING_FOLDER,
    "mapping_csv_name": "name_mapping.csv",
}
VAL_DS_KWARGS = {
    "data_folder_path": BRATS_2020_VALIDATION_FOLDER,
    "mapping_csv_name": "name_mapping_validation_data.csv",
}


def main() -> None:
    train_ds = BraTS2020Dataset(**TRAIN_DS_KWARGS)
    val_ds = BraTS2020Dataset(**VAL_DS_KWARGS)
    for images, targets in itertools.chain(train_ds, val_ds):  # noqa: B007
        _ = 0


if __name__ == "__main__":
    main()
