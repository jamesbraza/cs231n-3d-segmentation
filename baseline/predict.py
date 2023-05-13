import contextlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from baseline.data_load import BratsDataset, get_dataloader
from baseline.model import UNet3d
from baseline.model_utils import compute_scores_per_classes


def main() -> None:
    model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
    with contextlib.suppress(AssertionError):  # torch isn't compiled with CUDA enabled
        model = model.to("cuda")

    val_dataloader = get_dataloader(
        BratsDataset,
        "train_data.csv",
        phase="valid",
        fold=0,
    )
    print(len(val_dataloader))

    model.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model,
        val_dataloader,
        ["WT", "TC", "ET"],
    )
    dice_df = pd.DataFrame(dice_scores_per_classes)
    dice_df.columns = ["WT dice", "TC dice", "ET dice"]

    iou_df = pd.DataFrame(iou_scores_per_classes)
    iou_df.columns = ["WT jaccard", "TC jaccard", "ET jaccard"]
    val_metics_df = pd.concat([dice_df, iou_df], axis=1, sort=True)
    val_metics_df = val_metics_df.loc[
        :,
        ["WT dice", "WT jaccard", "TC dice", "TC jaccard", "ET dice", "ET jaccard"],
    ]
    val_metics_df.sample(5)
    colors = ["#35FCFF", "#FF355A", "#96C503", "#C5035B", "#28B463", "#35FFAF"]
    palette = sns.color_palette(colors, 6)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        x=val_metics_df.mean().index,
        y=val_metics_df.mean(),
        palette=palette,
        ax=ax,
    )
    ax.set_xticklabels(val_metics_df.columns, fontsize=14, rotation=15)
    ax.set_title("Dice and Jaccard Coefficients from Validation", fontsize=20)

    for idx, p in enumerate(ax.patches):
        percentage = "{:.1f}%".format(100 * val_metics_df.mean().to_numpy()[idx])
        x = p.get_x() + p.get_width() / 2 - 0.15
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x, y), fontsize=15, fontweight="bold")

    fig.savefig(
        "result1.png",
        format="png",
        pad_inches=0.2,
        transparent=False,
        bbox_inches="tight",
    )
    fig.savefig(
        "result1.svg",
        format="svg",
        pad_inches=0.2,
        transparent=False,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
