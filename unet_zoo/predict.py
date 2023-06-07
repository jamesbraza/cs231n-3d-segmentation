from typing import Any

import matplotlib.axes
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3dunet.unet3d.metrics import MeanIoU
from pytorch3dunet.unet3d.model import AbstractUNet, UNet3D
from pytorch3dunet.unet3d.utils import load_checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm

from unet_zoo import CHECKPOINTS_FOLDER
from unet_zoo.train import (
    BATCH_SIZE,
    INITIAL_CONV_OUT_CHANNELS,
    MASK_COUNT,
    NUM_GROUPS,
    NUM_SCANS_PER_EXAMPLE,
    get_train_val_scans_datasets,
)
from unet_zoo.utils import get_arbitrary_element, get_mask_middle, infer_device

LAST_MODEL = CHECKPOINTS_FOLDER / "last_checkpoint.pytorch"
BEST_MODEL = CHECKPOINTS_FOLDER / "best_checkpoint.pytorch"
THRESHOLD = 0.33


def make_summary_plot(
    images: torch.Tensor,
    actual_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    scan_id: int | None = None,
    slice_dim: int = 0,
) -> matplotlib.figure.Figure:
    """Create a summary plot depicting images, targets, and predictions."""
    fig = plt.figure(figsize=(20, 10))
    axes: list[matplotlib.axes.Axes] = []
    wt_mask_middle = get_mask_middle(mask=actual_masks[0], middle_dim=slice_dim)
    element_dim = wt_mask_middle, slice_dim
    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

    # 1. Plot source MRIs
    # NOTE: order matches BraTS2020MRIScansDataset.NONMASK_EXTENSIONS
    for i, title in enumerate(("FLAIR", "T1", "T1 contrast", "T2")):
        ax = fig.add_subplot(gs[0, i])
        axes.append(ax)
        ax_img = ax.imshow(
            get_arbitrary_element(images[i], *element_dim),
            cmap="bone",
        )
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
            ax.imshow(get_arbitrary_element(mask[0], *element_dim), cmap="summer")
            for ax, mask in axes_masks
        ],
        [
            ax.imshow(
                np.ma.masked_where(
                    ~get_arbitrary_element(mask[1], *element_dim).bool(),
                    get_arbitrary_element(mask[1], *element_dim),
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
                    ~get_arbitrary_element(mask[2], *element_dim).bool(),
                    get_arbitrary_element(mask[2], *element_dim),
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
                    "GD-Enhancing\nTumor",
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


def make_summary_plots(model: AbstractUNet) -> None:
    model.eval()
    val_ds = get_train_val_scans_datasets()[1]
    for images, targets in DataLoader(val_ds, batch_size=BATCH_SIZE):
        with torch.no_grad():
            preds = model(images)[0] > THRESHOLD
        for i in range(MASK_COUNT):
            summary_fig = make_summary_plot(  # noqa: F841
                images=images[0],
                actual_masks=targets[0],
                pred_masks=preds,
                slice_dim=i,
            )
        _ = 0  # Debug here


def sweep_thresholds(
    model: AbstractUNet,
    min_max: tuple[float, float] = (0.05, 0.95),
    num: int = 19,
) -> dict[float, float]:
    """Sweep through possible binary thresholds to maximize IoU."""
    model.eval()
    calc_iou = MeanIoU()
    threshold_to_mean_iou: dict[float, float] = {}
    val_ds = get_train_val_scans_datasets()[1]
    for threshold in tqdm(np.linspace(*min_max, num), desc="trying thresholds"):
        ious: list[float] = []
        for images, targets in tqdm(
            DataLoader(val_ds, batch_size=BATCH_SIZE),
            desc="compiling ious",
        ):
            with torch.no_grad():
                preds = model(images) >= threshold
            ious.append(calc_iou(preds, targets))
        threshold_to_mean_iou[threshold] = np.mean(ious)
    return threshold_to_mean_iou


def main() -> None:
    model = UNet3D(
        in_channels=NUM_SCANS_PER_EXAMPLE,
        out_channels=MASK_COUNT,
        final_sigmoid=True,
        f_maps=INITIAL_CONV_OUT_CHANNELS,
        num_groups=NUM_GROUPS,
    ).to(device=infer_device())
    state_dict: dict[str, Any] = load_checkpoint(BEST_MODEL, model)  # noqa: F841

    print(sweep_thresholds(model))


if __name__ == "__main__":
    main()
