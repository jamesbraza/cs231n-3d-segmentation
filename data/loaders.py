from __future__ import annotations

import os
import re
from enum import IntEnum

import nibabel as nib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from tqdm import tqdm

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
        return mask > cls.NON_TUMOR.value

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


class BraTS2020MRIScansDataset(Dataset):
    """Map-style dataset for BraTS 2020 MRI scans."""

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

    def get_image_folder(self, index: int) -> str:
        return os.path.join(
            self.data_folder_path,
            self._names[self.TARGET_COLUMN][index],
        )

    def get_full_path(self, index: int, extension: str) -> str:
        image_folder = self.get_image_folder(index)
        return os.path.join(image_folder, os.path.basename(image_folder) + extension)

    def _load_nii_with_slicing(self, path: str) -> np.ndarray:
        """Load in a .nii file taking into account the skip slices."""
        if self.skip_slices <= 0:
            raw_img = nib.load(path).dataobj
        else:
            raw_img = nib.load(path).dataobj[:, :, self.skip_slices : -self.skip_slices]
        return np.asarray(raw_img)

    def __getitem__(self, index: int | slice) -> tuple[torch.Tensor, ...]:
        """Get (images, masks) if training, otherwise (images,)."""
        if isinstance(index, slice):
            raise NotImplementedError("Dataset slicing is unimplemented.")
        raw_imgs = (
            self._load_nii_with_slicing(path)
            for path in (
                self.get_full_path(index, extension)
                for extension in self.NONMASK_EXTENSIONS
            )
        )
        try:
            # Normalize to be in [0, 1]
            img = np.stack(
                (img - img.min()) / (img.max() - img.min()) for img in raw_imgs
            )
        except KeyError as exc:
            raise IndexError(f"Index {index} is not in the dataset.") from exc
        image_tensor = torch.as_tensor(
            # N x W x H x C to N x C x H x W
            np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1)),
            dtype=torch.get_default_dtype(),  # Match pytorch-3dunet internals
            device=self.device,
        )
        if not self.train:
            return (image_tensor,)
        try:
            # Normal case
            mask = self._load_nii_with_slicing(
                path=self.get_full_path(index, self.MASK_EXTENSION),
            )
        except FileNotFoundError:
            # Exceptional case for training data's BraTS20_Training_355
            image_folder = self.get_image_folder(index)
            files = [
                i
                for i in os.listdir(image_folder)
                if os.path.isfile(os.path.join(image_folder, i))
            ]
            seg_files = [re.match(".*seg.*.nii", f, re.IGNORECASE) for f in files]
            mask = self._load_nii_with_slicing(
                path=os.path.join(image_folder, next(filter(None, seg_files)).string),
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


class BraTS2020MRISlicesDataset(IterableDataset):
    """
    Iterable-style dataset for BraTS 2020 MRI scan slices.

    This implementation memory-efficiently supports batching, across MRIs.
    """

    def __init__(
        self,
        scans_ds: BraTS2020MRIScansDataset,
        slices_per_mri: int | None = None,
        insert_z_dim: bool = True,
    ):
        """
        Initialize.

        Args:
            scans_ds: Dataset of MRI scans to wrap.
            slices_per_mri: Slices per MRI to use, leave as default of None to
                infer from the 0th MRI scan.
            insert_z_dim: Set True (default) to add a placeholder C dimension,
                for compatibility with pytorch-3dunet's UNet2D training.
        """
        self._scans_ds = scans_ds
        if slices_per_mri is None:  # Infer
            slices_per_mri = scans_ds[0][0].shape[1]
        self._slices_per_mri: int = slices_per_mri
        # Coordinate (scan, slice) of next slice to read
        self._coordinate: tuple[int, int] = 0, 0
        self.insert_z_dim = insert_z_dim

    _current_scans: tuple[torch.Tensor, ...]

    def __len__(self) -> int:
        return len(self._scans_ds) * self._slices_per_mri

    def __iter__(self) -> BraTS2020MRISlicesDataset:
        return self

    def __next__(self) -> tuple[torch.Tensor, ...]:
        scan_index, slice_index = self._coordinate
        if scan_index >= len(self._scans_ds):
            raise StopIteration
        if slice_index == 0:  # Fetch a new MRI scan
            self._current_scans = self._scans_ds[self._coordinate[0]]
        slices = tuple(s[:, slice_index] for s in self._current_scans)
        if self.insert_z_dim:
            slices = tuple(s.unsqueeze(dim=-3) for s in slices)
        if slice_index + 1 == self._slices_per_mri:
            self._coordinate = scan_index + 1, 0
        else:
            self._coordinate = scan_index, slice_index + 1
        return slices


def split_train_val(
    ds: Dataset,
    batch_size: int,
    fraction: float = 0.9,
) -> tuple[Subset, Subset]:
    """Split the input dataset into two based on a batch size and fraction."""
    num_train = int(len(ds) / batch_size * fraction)
    return tuple(
        torch.utils.data.random_split(
            dataset=ds,
            lengths=(num_train, len(ds) - num_train),
        ),
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


def play_scans_ds() -> None:
    train_ds = BraTS2020MRIScansDataset(**TRAIN_VAL_DS_KWARGS)
    for images, targets in tqdm(train_ds, desc="training dataset"):  # noqa: B007
        _ = 0  # Debug here
    _ = 0  # Debug here
    test_ds = BraTS2020MRIScansDataset(**TEST_DS_KWARGS)
    for images in tqdm(test_ds, desc="test dataset"):  # noqa: B007
        _ = 0  # Debug here


def play_slices_ds() -> None:
    train_scans_ds, val_scans_ds = split_train_val(
        BraTS2020MRIScansDataset(**TRAIN_VAL_DS_KWARGS),
        batch_size=1,
    )
    train_slices_ds = BraTS2020MRISlicesDataset(scans_ds=train_scans_ds)
    data_loader = DataLoader(train_slices_ds, batch_size=32)
    for images, targets in tqdm(data_loader, desc="training dataset"):  # noqa: B007
        _ = 0  # Debug here
    _ = 0  # Debug here


if __name__ == "__main__":
    play_scans_ds()
