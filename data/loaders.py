import os
from enum import IntEnum

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset

from data import BRATS_2020_TRAINING_FOLDER, BRATS_2020_VALIDATION_FOLDER


class BraTS2020Classes(IntEnum):
    """
    Classes found in the BraTS 2020 dataset.

    SEE: https://www.med.upenn.edu/cbica/brats2020/data.html
    """

    NON_TUMOR = 0
    NON_ENHANCING_TUMOR_CORE = 1  # Aka NCR/NET (neuroendocrine tumor)
    # Swelling around the tumor
    PERITUMORAL_EDEMA = 2  # Aka ED
    # Gadolinium enhancing
    GD_ENHANCING_TUMOR = 4  # Aka ET

    @classmethod
    def to_whole_tumor(cls, mask: npt.NDArray[int]) -> np.ndarray:
        """Convert a mask to whole tumor (WT)."""
        return mask >= cls.NON_TUMOR.value

    @classmethod
    def to_tumor_core(cls, mask: npt.NDArray[int]) -> np.ndarray:
        """Convert a mask to tumor core (TC)."""
        return np.logical_or(
            mask == cls.NON_ENHANCING_TUMOR_CORE.value,
            mask == cls.GD_ENHANCING_TUMOR.value,
        )

    @classmethod
    def to_enhancing_tumor(cls, mask: npt.NDArray[int]) -> np.ndarray:
        """Convert a mask to enhancing tumor (ET)."""
        return mask == cls.GD_ENHANCING_TUMOR.value


class BraTS2020Dataset(Dataset):
    """Map-style dataset for BraTS 2020."""

    TARGET_COLUMN = "BraTS_2020_subject_ID"
    # flair = T2-weighted Fluid Attenuated Inversion Recovery (T2-FLAIR)
    # t1 = native T1-weighted (T1)
    # t1ce = post-contrast T1-weighted (T1Gd)
    # t2 = T2-weighted (T2)
    NONMASK_EXTENSIONS = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
    MASK_EXTENSION = "_seg.nii"

    def __init__(
        self,
        data_folder_path: os.PathLike | str,
        mapping_csv_name: str,
        device: torch.device | None = None,
        train: bool = True,
        skip_slices: int = 0,
    ):
        """
        Initialize.

        Args:
            data_folder_path: Path to the BraTS 2020 dataset.
            mapping_csv_name: Name of the name mapping CSV file.
            device: Optional torch device to use, default is None (CPU).
            train: Set True (default) for training data (images and mask), set
                False for test data (only images).
            skip_slices: Symmetric count of MRI slices to exclude, since the
                first few and last few slices usually are empty or nearly-empty.
        """
        self.data_folder_path = data_folder_path
        self._names = pd.read_csv(
            os.path.join(data_folder_path, mapping_csv_name),
            usecols=[self.TARGET_COLUMN],
            dtype=str,
        )
        self.device = device
        self.train = train
        self.skip_slices = skip_slices

    def __len__(self) -> int:
        return len(self._names)

    def get_full_path(self, index: int, extension: str) -> str:
        image_folder = os.path.join(
            self.data_folder_path,
            self._names[self.TARGET_COLUMN][index],
        )
        return os.path.join(image_folder, os.path.basename(image_folder) + extension)

    def _load_nii_with_slicing(self, path: str) -> np.ndarray:
        """Load in a .nii file taking into account the skip slices."""
        if self.skip_slices <= 0:
            raw_img = nib.load(path).dataobj
        else:
            raw_img = nib.load(path).dataobj[:, :, self.skip_slices : -self.skip_slices]
        return np.asarray(raw_img)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Get (images, masks) if training, otherwise (images,)."""
        raw_imgs = (
            self._load_nii_with_slicing(path)
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
            device=self.device,
        )
        if not self.train:
            return (image_tensor,)
        mask = self._load_nii_with_slicing(
            path=self.get_full_path(index, self.MASK_EXTENSION),
        )
        wt = BraTS2020Classes.to_whole_tumor(mask)
        tc = BraTS2020Classes.to_tumor_core(mask)
        et = BraTS2020Classes.to_enhancing_tumor(mask)
        return image_tensor, torch.as_tensor(
            # N x W x H x C to N x C x H x W
            np.moveaxis(np.stack((wt, tc, et)), (0, 1, 2, 3), (0, 3, 2, 1)),
            dtype=torch.get_default_dtype(),  # Match pytorch-3dunet internals
            device=self.device,
        )


# Has labels
TRAIN_VAL_DS_KWARGS = {
    "data_folder_path": BRATS_2020_TRAINING_FOLDER,
    "mapping_csv_name": "name_mapping.csv",
}
# Has no labels
TEST_DS_KWARGS = {
    "data_folder_path": BRATS_2020_VALIDATION_FOLDER,
    "mapping_csv_name": "name_mapping_validation_data.csv",
    "train": False,
}


def main() -> None:
    train_ds = BraTS2020Dataset(**TRAIN_VAL_DS_KWARGS)
    for images, targets in train_ds:  # noqa: B007
        _ = 0
    test_ds = BraTS2020Dataset(**TEST_DS_KWARGS)
    for images in test_ds:  # noqa: B007
        _ = 0


if __name__ == "__main__":
    main()
