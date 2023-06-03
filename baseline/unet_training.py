import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import Image, clear_output, display
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from baseline import BASELINE_FOLDER
from baseline.config import GlobalConfig
from baseline.config import config as default_config
from baseline.model_utils import BCEDiceLoss, Meter, compute_scores_per_classes
from baseline.training_utils import (
    Image3dToGIF3d,
    ShowResult,
    compute_results,
    merging_two_gif,
)
from baseline.unet_data import BratsDataset, get_dataloader
from baseline.unet_model import UNet3d


class Trainer:
    """
    Factory for training process.

    Args:
        display_plot: if True - plot train history after each epoch.
        net: neural network for mask prediction.
        criterion: factory for calculating objective loss.
        optimizer: optimizer for weights updating.
        phases: list with train and validation phases.
        dataloaders: dict with data loaders for train and val phases.
        path_to_csv: path to csv file.
        meter: factory for storing and updating metrics.
        batch_size: data batch size for one step weights updating.
        num_epochs: num weights updation for all data.
        accumulation_steps: the number of steps after which the optimization step can be taken
                    (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
        lr: learning rate for optimizer.
        scheduler: scheduler for control learning rate.
        losses: dict for storing lists with losses for each phase.
        jaccard_scores: dict for storing lists with jaccard scores for each phase.
        dice_scores: dict for storing lists with dice scores for each phase.
    """

    def __init__(
        self,
        net: nn.Module,
        dataset: torch.utils.data.Dataset,
        criterion: nn.Module,
        lr: float,
        accumulation_steps: int,
        batch_size: int,
        fold: int,
        num_epochs: int,
        path_to_csv: str,
        display_plot: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", self.device)
        self.display_plot = display_plot
        self.net = net
        self.net = self.net.to(self.device)
        self.criterion = criterion
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=2,
            verbose=True,
        )
        self.accumulation_steps = accumulation_steps // batch_size
        self.phases = ["train", "val"]
        self.num_epochs = num_epochs

        self.dataloaders = {
            phase: get_dataloader(
                dataset=dataset,
                path_to_csv=path_to_csv,
                phase=phase,
                fold=fold,
                batch_size=batch_size,
                num_workers=4,
            )
            for phase in self.phases
        }
        self.best_loss = float("inf")
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.jaccard_scores = {phase: [] for phase in self.phases}

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        meter = Meter()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch["image"], data_batch["mask"]
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            meter.update(logits.detach().cpu(), targets.detach().cpu())

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        epoch_dice, epoch_iou = meter.get_metrics()

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)
        self.jaccard_scores[phase].append(epoch_iou)

        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)
            if self.display_plot:
                self._plot_train_history()

            if val_loss < self.best_loss:
                print(f"\n{'# ' *20}\nSaved new checkpoint\n{'# ' *20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "best_model.pth")
            print()
        self._save_train_history()

    def _plot_train_history(self):
        data = [self.losses, self.dice_scores, self.jaccard_scores]
        colors = ["deepskyblue", "crimson"]
        labels = [
            f"""
            train loss {self.losses['train'][-1]}
            val loss {self.losses['val'][-1]}
            """,
            f"""
            train dice score {self.dice_scores['train'][-1]}
            val dice score {self.dice_scores['val'][-1]}
            """,
            f"""
            train jaccard score {self.jaccard_scores['train'][-1]}
            val jaccard score {self.jaccard_scores['val'][-1]}
            """,
        ]

        clear_output(wait=True)
        with plt.style.context("seaborn-dark-palette"):
            fig, axes = plt.subplots(3, 1, figsize=(8, 10))
            for i, ax in enumerate(axes):
                ax.plot(data[i]["val"], c=colors[0], label="val")
                ax.plot(data[i]["train"], c=colors[-1], label="train")
                ax.set_title(labels[i])
                ax.legend(loc="upper right")

            plt.tight_layout()
            plt.show()

    def load_pretrained_model(self, state_path: str):
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        self.net.load_state_dict(torch.load(state_path, map_location=map_location))
        print("Pretrained model loaded")

    def _save_train_history(self):
        """Write model weights and training logs to files."""
        torch.save(self.net.state_dict(), "last_epoch_model.pth")

        logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
        log_names_ = ["_loss", "_dice", "_jaccard"]
        logs = [logs_[i][key] for i in list(range(len(logs_))) for key in logs_[i]]
        log_names = [
            key + log_names_[i] for i in list(range(len(logs_))) for key in logs_[i]
        ]
        pd.DataFrame(dict(zip(log_names, logs, strict=True))).to_csv(
            "train_log.csv",
            index=False,
        )


def quantify_inference_time(
    model,
    dataloader: DataLoader,
    save_path: str = BASELINE_FOLDER / "inference_histogram",
) -> None:
    model.eval()
    with torch.no_grad():
        dts: list[float] = []
        for data in tqdm(dataloader, desc="inferences"):
            tic = time.perf_counter()
            probs = torch.sigmoid(model(data["image"]))  # noqa: F841
            toc = time.perf_counter()
            dts.append(toc - tic)

    print(f"All delta times: {dts}.")
    counts, bins = np.histogram(np.array(dts), bins="auto")
    plt.hist(bins[:-1], bins, weights=counts, rwidth=0.9)
    plt.xlabel("Inference Time (sec)")
    plt.ylabel("Count")
    plt.savefig(f"{save_path}.png", format="png", bbox_inches="tight")


def visualize_results(model, dataloader: DataLoader, save_gif: bool = True) -> None:
    model.eval()
    results = compute_results(model, dataloader, 0.33)
    for id_, img, gt, prediction in zip(
        results["Id"][4:],
        results["image"][4:],
        results["GT"][4:],
        results["Prediction"][4:],
        strict=True,
    ):
        print(id_)
        show_result = ShowResult()
        show_result.plot(img, gt, prediction)

        # Ground truth
        gt = gt.squeeze().cpu().detach().numpy()
        gt = np.moveaxis(gt, (0, 1, 2, 3), (0, 3, 2, 1))
        wt, tc, et = gt
        print(wt.shape, tc.shape, et.shape)
        gt = wt + tc + et
        gt = np.clip(gt, 0, 1)
        print(gt.shape)

        title = "Ground Truth_" + id_[0]
        filename1 = title + "_3d"

        data_to_3dgif = Image3dToGIF3d(
            img_dim=(120, 120, 78),
            binary=True,
            normalizing=False,
        )
        transformed_data = data_to_3dgif.get_transformed_data(gt)
        data_to_3dgif.plot_cube(
            transformed_data,
            title=title,
            make_gif=save_gif,
            path_to_save=filename1,
        )

        # Prediction
        prediction = prediction.squeeze().cpu().detach().numpy()
        prediction = np.moveaxis(prediction, (0, 1, 2, 3), (0, 3, 2, 1))
        wt, tc, et = prediction
        print(wt.shape, tc.shape, et.shape)
        prediction = wt + tc + et
        prediction = np.clip(prediction, 0, 1)
        print(prediction.shape)

        title = "Prediction_" + id_[0]
        filename2 = title + "_3d"

        data_to_3dgif = Image3dToGIF3d(
            img_dim=(120, 120, 78),
            binary=True,
            normalizing=False,
        )
        transformed_data = data_to_3dgif.get_transformed_data(prediction)
        data_to_3dgif.plot_cube(
            transformed_data,
            title=title,
            make_gif=save_gif,
            path_to_save=filename2,
        )

        if save_gif:
            merging_two_gif(f"{filename1}.gif", f"{filename2}.gif", "result.gif")
            display(Image("result.gif", format="png"))
        break


def visualize_metrics(model: UNet3d, dataloader: DataLoader) -> None:
    model.eval()
    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(
        model,
        dataloader,
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


def visualize_post_training(model: UNet3d) -> None:
    val_dataloader = get_dataloader(
        BratsDataset,
        BASELINE_FOLDER / "train_data.csv",
        phase="valid",
        fold=0,
    )
    print(len(val_dataloader))

    quantify_inference_time(model, val_dataloader)
    visualize_metrics(model, val_dataloader)
    visualize_results(model, val_dataloader)


def main(config: GlobalConfig | None = None, skip_training: bool = False) -> None:
    if config is None:
        config = GlobalConfig(pretrained_model_path=None)
    model = UNet3d(in_channels=4, n_classes=3, n_channels=24)
    if torch.cuda.is_available():
        model = model.to("cuda")
    trainer = Trainer(
        net=model,
        dataset=BratsDataset,
        criterion=BCEDiceLoss(),
        lr=5e-4,
        accumulation_steps=4,
        batch_size=1,
        fold=0,
        num_epochs=1,
        path_to_csv=config.path_to_csv,
    )
    if config.pretrained_model_path is not None:
        trainer.load_pretrained_model(config.pretrained_model_path)

        if config.train_logs_path is not None:
            train_logs = pd.read_csv(config.train_logs_path)
            trainer.losses["train"] = train_logs.loc[:, "train_loss"].to_list()
            trainer.losses["val"] = train_logs.loc[:, "val_loss"].to_list()
            trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
            trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
            trainer.jaccard_scores["train"] = train_logs.loc[
                :,
                "train_jaccard",
            ].to_list()
            trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()

    if not skip_training:
        trainer.run()
    visualize_post_training(model)


if __name__ == "__main__":
    main(default_config)
