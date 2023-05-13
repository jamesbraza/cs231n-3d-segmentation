import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from skimage.transform import resize
from skimage.util import montage
from tqdm import tqdm


class Image3dToGIF3d:
    """
    Displaying 3D images in 3d axes.

    Parameters:
        img_dim: shape of cube for resizing.
        figsize: figure size for plotting in inches.
    """

    def __init__(
        self,
        img_dim: tuple = (55, 55, 55),
        figsize: tuple = (15, 10),
        binary: bool = False,
        normalizing: bool = True,
    ):
        self.img_dim = img_dim
        print(img_dim)
        self.figsize = figsize
        self.binary = binary
        self.normalizing = normalizing

    def _explode(self, data: np.ndarray):
        """Compute an array twice as large in each dimension, with an extra space
        between each voxel."""
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded

    def _expand_coordinates(self, indices: np.ndarray):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z

    def _normalize(self, arr: np.ndarray):
        """Normilize image value between 0 and 1."""
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)

    def _scale_by(self, arr: np.ndarray, factor: int):
        """
        Scale 3d Image to factor.

        Args:
            arr: 3d image for scaling.
            factor: factor for scaling.
        """
        mean = np.mean(arr)
        return (arr - mean) * factor + mean

    def get_transformed_data(self, data: np.ndarray):
        """Transform data: normalization, scaling, resizing."""
        if self.binary:
            resized_data = resize(data, self.img_dim, preserve_range=True)
            return np.clip(resized_data.astype(np.uint8), 0, 1).astype(np.float32)

        norm_data = np.clip(self._normalize(data) - 0.1, 0, 1) ** 0.4
        scaled_data = np.clip(self._scale_by(norm_data, 2) - 0.1, 0, 1)
        resized_data = resize(scaled_data, self.img_dim, preserve_range=True)

        return resized_data

    def plot_cube(
        self,
        cube,
        title: str = "",
        init_angle: int = 0,
        make_gif: bool = False,
        path_to_save: str = "filename.gif",
    ):
        """
        Plot 3d data.

        Args:
            cube: 3d data
            title: title for figure.
            init_angle: angle for image plot (from 0-360).
            make_gif: if True create gif from every 5th frames from 3d image plot.
            path_to_save: path to save GIF file.
        """
        if self.binary:
            facecolors = cm.winter(cube)
            print("binary")
        else:
            if self.normalizing:
                cube = self._normalize(cube)
            facecolors = cm.gist_stern(cube)
            print("not binary")

        facecolors[:, :, :, -1] = cube
        facecolors = self._explode(facecolors)

        filled = facecolors[:, :, :, -1] != 0
        x, y, z = self._expand_coordinates(np.indices(np.array(filled.shape) + 1))

        with plt.style.context("dark_background"):
            fig = plt.figure(figsize=self.figsize)
            ax = fig.gca(projection="3d")

            ax.view_init(30, init_angle)
            ax.set_xlim(right=self.img_dim[0] * 2)
            ax.set_ylim(top=self.img_dim[1] * 2)
            ax.set_zlim(top=self.img_dim[2] * 2)
            ax.set_title(title, fontsize=18, y=1.05)

            ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)

            if make_gif:
                images = []
                for angle in tqdm(range(0, 360, 5)):
                    ax.view_init(30, angle)
                    fname = str(angle) + ".png"

                    plt.savefig(fname, dpi=120, format="png", bbox_inches="tight")
                    images.append(imageio.imread(fname))
                    # os.remove(fname)
                imageio.mimsave(path_to_save, images)
                plt.close()

            else:
                plt.show()


def compute_results(model, dataloader, threshold=0.33) -> dict[str, list]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {"Id": [], "image": [], "GT": [], "Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data["Id"], data["image"], data["mask"]
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits)

            predictions = (probs >= threshold).float()
            predictions = predictions.cpu()
            targets = targets.cpu()

            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)

            # only 5 pars
            if i > 5:
                return results
        return results


class ShowResult:
    def mask_preprocessing(self, mask):
        """Test."""
        mask = mask.squeeze().cpu().detach().numpy()
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        mask_WT = np.rot90(montage(mask[0]))
        mask_TC = np.rot90(montage(mask[1]))
        mask_ET = np.rot90(montage(mask[2]))

        return mask_WT, mask_TC, mask_ET

    def image_preprocessing(self, image):
        """Return image flair as mask for overlapping gt and predictions."""
        image = image.squeeze().cpu().detach().numpy()
        image = np.moveaxis(image, (0, 1, 2, 3), (0, 3, 2, 1))
        return np.rot90(montage(image[0]))

    def plot(self, image, ground_truth, prediction):
        image = self.image_preprocessing(image)
        gt_mask_WT, gt_mask_TC, gt_mask_ET = self.mask_preprocessing(ground_truth)
        pr_mask_WT, pr_mask_TC, pr_mask_ET = self.mask_preprocessing(prediction)

        fig, axes = plt.subplots(1, 2, figsize=(35, 30))

        [ax.axis("off") for ax in axes]
        axes[0].set_title("Ground Truth", fontsize=35, weight="bold")
        axes[0].imshow(image, cmap="bone")
        axes[0].imshow(
            np.ma.masked_where(gt_mask_WT is False, gt_mask_WT),
            cmap="cool_r",
            alpha=0.6,
        )
        axes[0].imshow(
            np.ma.masked_where(gt_mask_TC is False, gt_mask_TC),
            cmap="autumn_r",
            alpha=0.6,
        )
        axes[0].imshow(
            np.ma.masked_where(gt_mask_ET is False, gt_mask_ET),
            cmap="autumn",
            alpha=0.6,
        )

        axes[1].set_title("Prediction", fontsize=35, weight="bold")
        axes[1].imshow(image, cmap="bone")
        axes[1].imshow(
            np.ma.masked_where(pr_mask_WT is False, pr_mask_WT),
            cmap="cool_r",
            alpha=0.6,
        )
        axes[1].imshow(
            np.ma.masked_where(pr_mask_TC is False, pr_mask_TC),
            cmap="autumn_r",
            alpha=0.6,
        )
        axes[1].imshow(
            np.ma.masked_where(pr_mask_ET is False, pr_mask_ET),
            cmap="autumn",
            alpha=0.6,
        )

        plt.tight_layout()

        plt.show()


# show_result = ShowResult()
# show_result.plot(data['image'], data['mask'], data['mask'])


def merging_two_gif(path1: str, path2: str, name_to_save: str):
    """
    Merge GIFs side by side.

    Args:
        path1: path to gif with ground truth.
        path2: path to gif with prediction.
        name_to_save: name for saving new GIF.
    """
    # https://stackoverflow.com/questions/51517685/combine-several-gif-horizontally-python
    # Create reader object for the gif
    gif1 = imageio.get_reader(path1)
    gif2 = imageio.get_reader(path2)

    # If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length())

    # Create writer object
    new_gif = imageio.get_writer(name_to_save)

    for _frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        # here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()
    new_gif.close()


# merging_two_gif('BraTS20_Training_001_flair_3d.gif',
#                'BraTS20_Training_001_flair_3d.gif',
#                'result.gif')


def get_all_csv_file(root: str) -> list:
    """Extract all unique ids from file names."""
    ids = []
    for dirname, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if path.endswith(".csv"):
                ids.append(path)
    ids = list(set(filter(None, ids)))
    print(f"Extracted {len(ids)} csv files.")
    return ids


# csv_paths = get_all_csv_file(
#     "../input/brats20-dataset-training-validation/BraTS2020_TrainingData",
# )
