import os
from collections.abc import Callable

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from albumentations import Compose
from skimage.transform import resize
from skimage.util import montage
from torch.utils.data import DataLoader, Dataset


def get_augmentations(phase):
    return Compose([])


class BratsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase: str = "test", is_resize: bool = False):
        self.df = df
        self.phase = phase
        self.augmentations = get_augmentations(phase)
        self.data_types = ["_flair.nii", "_t1.nii", "_t1ce.nii", "_t2.nii"]
        self.is_resize = is_resize

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, "Brats20ID"]
        root_path = self.df.loc[self.df["Brats20ID"] == id_]["path"].to_numpy()[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)  # .transpose(2, 0, 1)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_path = os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)

            augmented = self.augmentations(
                image=img.astype(np.float32),
                mask=mask.astype(np.float32),
            )

            img = augmented["image"]
            mask = augmented["mask"]

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }

        return {
            "Id": id_,
            "image": img,
        }

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


def get_dataloader(
    dataset: Callable[[pd.DataFrame, str], Dataset],
    path_to_csv: str,
    phase: str,
    fold: int = 0,
    batch_size: int = 1,
    num_workers: int = 4,
):
    """Get a dataloader for the model training."""
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


def main() -> None:
    dataloader = get_dataloader(
        dataset=BratsDataset,
        path_to_csv="train_data.csv",
        phase="valid",
        fold=0,
    )
    print(len(dataloader))
    data = next(iter(dataloader))
    print(data["Id"], data["image"].shape, data["mask"].shape)

    img_tensor = data["image"].squeeze()[0].cpu().detach().numpy()
    mask_tensor = data["mask"].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor))

    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image, cmap="bone")
    ax.imshow(np.ma.masked_where(mask is False, mask), cmap="cool", alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main()
