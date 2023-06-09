import itertools
import os
from collections.abc import Iterable
from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rich.status
import torch
from matplotlib import colors
from pytorch3dunet.unet3d.model import AbstractUNet, UNet3D
from pytorch3dunet.unet3d.utils import load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.loaders import ET, MASK_COUNT, TC, WT
from unet_zoo import CHECKPOINTS_FOLDER, IMAGES_FOLDER
from unet_zoo.metrics import MeanIoU
from unet_zoo.train import (
    BATCH_SIZE,
    INITIAL_CONV_OUT_CHANNELS,
    NUM_GROUPS,
    NUM_SCANS_PER_EXAMPLE,
    get_train_val_test_scans_datasets,
)
from unet_zoo.utils import get_mask_middle, infer_device

LAST_MODEL = CHECKPOINTS_FOLDER / "last_checkpoint.pytorch"
BEST_MODEL = CHECKPOINTS_FOLDER / "best_checkpoint.pytorch"
THRESHOLD = 0.25  # Tuned parameter
DEVICE = infer_device()


def make_summary_slice_plot(
    images: torch.Tensor,
    actual_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    scan_id: int | None = None,
    slice_dim: int = 0,
) -> matplotlib.figure.Figure:
    """Create a summary plot depicting images, targets, and predictions."""
    # Take off GPU, if not already there
    images, actual_masks, pred_masks = (
        images.cpu(),
        actual_masks.cpu(),
        pred_masks.cpu(),
    )
    fig = plt.figure(figsize=(20, 10))
    axes: list[matplotlib.axes.Axes] = []
    wt_mask_middle = get_mask_middle(mask=actual_masks[WT], middle_dim=slice_dim)
    element_dim = wt_mask_middle, slice_dim
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    # 1. Plot source MRIs
    # NOTE: order matches BraTS2020MRIScansDataset.NONMASK_EXTENSIONS
    for i, title in enumerate(("FLAIR", "T1", "T1 contrast", "T2")):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        ax_img = ax.imshow(np.take(images[i], *element_dim), cmap="bone")
        ax.set_title(title, fontsize=18, weight="bold", y=-0.2)
        fig.colorbar(ax_img)

    # 2. Plot segmentations
    axes_masks = (fig.add_subplot(gs[1, :2]), actual_masks), (
        fig.add_subplot(gs[1, 2:]),
        pred_masks,
    )
    for title, (ax, _) in zip(
        ["Target Mask", "Predicted Mask"],
        axes_masks,
        strict=True,
    ):
        ax.set_title(title, fontsize=18, weight="bold", y=-0.15)

    all_seg_ax_imgs: list[list[matplotlib.image.AxesImage]] = [
        [
            ax.imshow(np.take(mask[WT], *element_dim), cmap="summer")
            for ax, mask in axes_masks
        ],
        [
            ax.imshow(
                np.ma.masked_where(
                    ~np.take(mask[TC], *element_dim).bool(),
                    np.take(mask[TC], *element_dim),
                ),
                cmap="rainbow",
                alpha=0.6,
            )
            for ax, mask in axes_masks
            if mask.max() > 0
        ],
        [
            ax.imshow(
                np.ma.masked_where(
                    ~np.take(mask[ET], *element_dim).bool(),
                    np.take(mask[ET], *element_dim),
                ),
                cmap="winter",
                alpha=0.6,
            )
            for ax, mask in axes_masks
            if mask.max() > 0
        ],
    ]
    patches = [
        mpatches.Patch(
            color=all_seg_ax_imgs[0][0].cmap(all_seg_ax_imgs[0][0].norm(value=0)),
            label="Healthy",
        ),
        *(
            mpatches.Patch(
                color=seg_ax_imgs[0].cmap(seg_ax_imgs[0].norm(value=1)),
                label=label,
            )
            for label, seg_ax_imgs in zip(
                [
                    "Non-Enhancing\nTumor Core",
                    "Peritumoral Edema",
                    "Gd-Enhancing\nTumor",
                ],
                all_seg_ax_imgs,
                strict=True,
            )
            if len(seg_ax_imgs) > 0
        ),
    ]
    axes_masks[1][0].legend(
        handles=patches,
        bbox_to_anchor=(1.35, 0.2, 0.5, 0.5),
        loc="upper right",
        fontsize="xx-large",
        title_fontsize=18,
        framealpha=0.0,
    )

    # 3. Wrap up with display of title
    for ax_ in (*axes, *(ax for ax, _ in axes_masks)):
        ax_.set_axis_off()
    scan_id = "" if scan_id is None else f"{scan_id} "
    plt.suptitle(
        f"MRI Scan {scan_id}at Slice {wt_mask_middle}",
        fontsize=20,
        weight="bold",
    )
    return fig


def make_summary_plots(
    model: AbstractUNet,
    threshold: float | torch.Tensor = THRESHOLD,
) -> None:
    """Make plots of the predicted and actual masks from three angles."""
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    model.eval()
    test_ds = get_train_val_test_scans_datasets()[2]
    for ex_i, (images, targets) in enumerate(DataLoader(test_ds)):
        with torch.no_grad():
            preds = model(images)
        for mask_i in range(MASK_COUNT):
            summary_fig = make_summary_slice_plot(
                images=images[0],
                actual_masks=targets[0],
                pred_masks=preds[0] >= threshold,
                slice_dim=mask_i,
            )
            summary_fig.savefig(
                IMAGES_FOLDER / f"unet3d_inference_ex{ex_i}_angle{mask_i}.png",
            )
            plt.close(summary_fig)
        _ = 0  # Debug here


KNOWN_APPEALING_ELEV_AZIM_ROLL = (
    (60, 0, 90),
    (60, 210, 300),
    (120, 150, 240),
    (240, 180, 270),
    (300, 150, 300),
)


def _plot_images_masks(
    ex_i: int,
    image: np.ndarray,
    masks: np.ndarray,
    name: str = "predictions",
    threshold: float | torch.Tensor | None = None,
    elev_azim_rolls: Iterable[
        tuple[int | None, int | None, int | None]
    ] = KNOWN_APPEALING_ELEV_AZIM_ROLL,
) -> None:
    """Save a figure of the input image and masks."""
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    with rich.status.Status(f"plotting example {ex_i}'s WT mask"):
        wt_mask = masks[WT] >= threshold if threshold is not None else masks[WT]
        facecolors = np.empty((*wt_mask.shape, 4), dtype=object)
        facecolors[wt_mask] = colors.to_rgba("green", alpha=0.5)
        ax.voxels(wt_mask, facecolors=facecolors)
        del wt_mask
    with rich.status.Status(f"plotting example {ex_i}'s TC mask"):
        tc_mask = masks[TC] >= threshold if threshold is not None else masks[TC]
        facecolors = np.empty((*tc_mask.shape, 4), dtype=object)
        facecolors[tc_mask] = colors.to_rgba("blue", alpha=0.75)
        ax.voxels(tc_mask, facecolors=facecolors)
        del tc_mask
    with rich.status.Status(f"plotting example {ex_i}'s ET mask"):
        et_mask = masks[ET] >= threshold if threshold is not None else masks[ET]
        facecolors = np.empty(et_mask.shape, dtype=object)
        facecolors[et_mask] = "purple"
        ax.voxels(et_mask, facecolors=facecolors)
        del et_mask, masks
    with rich.status.Status(f"plotting example {ex_i}'s images"):
        facecolors = np.empty((*image.shape, 4), dtype=object)
        facecolors[image] = colors.to_rgba("grey", alpha=0.25)
        ax.voxels(image, facecolors=facecolors)
        del image
    plt.tight_layout()
    # After iterating over all 30°, these combinations looked good
    for elev_azim_roll in elev_azim_rolls:
        destination = IMAGES_FOLDER / f"unet3d_{name}_ex{ex_i}_ear{elev_azim_roll}.png"
        with rich.status.Status(
            f"saving 3D visualization of example {ex_i} to {destination}",
        ):
            ax.view_init(*elev_azim_roll)
            fig.canvas.draw()
            fig.savefig(destination, transparent=True)
    plt.close(fig)


def make_3d_visualization(
    model: AbstractUNet,
    threshold: float | torch.Tensor = THRESHOLD,
) -> None:
    """Save 3-D visualizations test dataset's labels and predictions."""
    os.makedirs(IMAGES_FOLDER, exist_ok=True)
    model.eval()
    test_ds = get_train_val_test_scans_datasets()[2]
    for ex_i, (images, targets) in enumerate(DataLoader(test_ds)):
        with (
            rich.status.Status(f"running prediction on example {ex_i}"),
            torch.no_grad(),
        ):
            preds = model(images)
        # Unbatch, convert to bool, take off GPU (if not already there)
        compressed_image = (images[0].max(dim=0)[0] > 0).cpu()
        binary_targets = targets[0].bool().cpu()
        binary_preds = (preds[0] >= threshold).cpu()
        del images, targets, preds

        kwargs = {"ex_i": ex_i, "image": compressed_image}
        _plot_images_masks(masks=binary_preds, threshold=threshold, **kwargs)
        del binary_preds
        _plot_images_masks(masks=binary_targets, name="labels", **kwargs)
        del binary_targets, compressed_image
        _ = 0  # Debug here


def sweep_thresholds(
    model: AbstractUNet,
    min_max: tuple[float, float] = (0.05, 0.95),
    num: int = 19,
    multi_channel: bool = True,
    mean_iou_binarize: bool = False,
    save_filename: str | None = None,
) -> dict[tuple[float, float, float], float]:
    """Sweep through possible binary thresholds to maximize IoU."""
    model.eval()
    calc_iou = MeanIoU(binarize=mean_iou_binarize)
    threshold_to_mean_iou: dict[tuple[float, float, float], float] = {}
    val_ds = get_train_val_test_scans_datasets()[1]
    if multi_channel:
        iterator = itertools.product(
            np.linspace(*min_max, num),
            np.linspace(*min_max, num),
            np.linspace(*min_max, num),
        )
    else:
        iterator = np.linspace(*min_max, num)
    for thresholds in tqdm(iterator, desc="trying thresholds"):
        ious: list[float] = []
        for images, targets in tqdm(
            DataLoader(val_ds, batch_size=BATCH_SIZE),
            desc="compiling ious",
        ):
            with torch.no_grad():
                preds = model(images)
            thresholds_tensor = torch.as_tensor(thresholds, device=DEVICE)
            if multi_channel:
                thresholds_tensor = thresholds_tensor.reshape(
                    3,
                    *(1,) * len(preds.shape[2:]),
                )
            ious.append(
                calc_iou(input=preds >= thresholds_tensor, target=targets),
            )
        threshold_to_mean_iou[thresholds] = np.mean(ious).item()

    if save_filename is not None:
        # This could be done with less lines, but that requires more thought
        storage_wt_tc_et, mean_ious = ([], [], []), []
        for threshold_tuples, mean_iou in threshold_to_mean_iou.items():
            mean_ious.append(mean_iou)
            for storage, threshold in zip(
                storage_wt_tc_et,
                threshold_tuples,
                strict=True,
            ):
                storage.append(threshold)

        fig, ax = plt.subplots()
        if multi_channel:
            for x, label in zip(storage_wt_tc_et, ["WT", "TC", "ET"], strict=True):
                ax.scatter(x, mean_ious, label=label)
            ax.legend()
        else:
            ax.scatter(*zip(*list(threshold_to_mean_iou.items()), strict=True))
        ax.set_xlim(left=max(min_max[0] - 0.05, 0), right=min(min_max[1] + 0.05, 1.0))
        ax.set_xlabel("Binary threshold")
        ax.set_ylabel("Intersection over Union (IoU)")
        ax.set_title("Discerning Best Binary Threshold")
        fig.savefig(save_filename)
        plt.close(fig)

    return threshold_to_mean_iou


def main() -> None:
    model = UNet3D(
        in_channels=NUM_SCANS_PER_EXAMPLE,
        out_channels=MASK_COUNT,
        final_sigmoid=True,
        f_maps=INITIAL_CONV_OUT_CHANNELS,
        num_groups=NUM_GROUPS,
    ).to(device=DEVICE)
    state_dict: dict[str, Any] = load_checkpoint(BEST_MODEL, model)  # noqa: F841

    make_3d_visualization(model)


if __name__ == "__main__":
    main()
