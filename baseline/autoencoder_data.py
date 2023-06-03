import os
from collections.abc import Callable

import nibabel as nib
import numpy as np
import pandas as pd
from albumentations import Compose
from torch.utils.data import DataLoader, Dataset


def get_augmentations(phase):
    list_transforms = []

    return Compose(list_transforms)


def get_dataloader(
    dataset: Callable[[pd.DataFrame, str], Dataset],
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Returns: dataloader for the model training."""
    read_df = pd.read_csv(path_to_csv)

    train_df = read_df.loc[read_df["fold"] != fold].reset_index(drop=True)
    val_df = read_df.loc[read_df["fold"] == fold].reset_index(drop=True)

    dataset = dataset(train_df if phase == "train" else val_df, phase)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )


class AutoEncoderDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test"):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Brats20ID"]
        root_path = self.df.loc[self.df["Brats20ID"] == id_]["path"].to_numpy()[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)

            img = self.normalize(img)
            images.append(img.astype(np.float32))
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        return {
            "Id": id_,
            "data": img,
            "label": img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        return np.asarray(data.dataobj)

    def normalize(self, data: np.ndarray, mean=0.0, std=1.0):  # noqa: ARG002
        """Normilize image value between 0 and 1."""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)


def main() -> None:
    dataloader = get_dataloader(
        AutoEncoderDataset,
        "train_data.csv",
        phase="valid",
        fold=0,
    )
    print(len(dataloader))

    data = next(iter(dataloader))
    print(data["Id"], data["data"].shape, data["label"].shape)


if __name__ == "__main__":
    main()
