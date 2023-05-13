import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Image, display
from skimage.util import montage
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from baseline.autoencoder_data import AutoEncoderDataset
from baseline.autoencoder_model import AutoEncoder
from baseline.config import GlobalConfig
from baseline.config import config as default_config
from baseline.training_utils import (
    Image3dToGIF3d,
    merging_two_gif,
)
from baseline.unet_data import get_dataloader


class Trainer:
    def __init__(
        self,
        net: nn.Module,
        criterion: nn.Module,
        lr: float,
        accumulation_steps: int,
        batch_size: int,
        fold: int,
        num_epochs: int,
        path_to_csv: str,
        dataset: torch.utils.data.Dataset,
    ):
        """Initialization."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", self.device)
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

    def _compute_loss_and_outputs(self, images: torch.Tensor, targets: torch.Tensor):
        images = images.to(self.device)
        targets = targets.to(self.device)
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        return loss, logits

    def _do_epoch(self, epoch: int, phase: str):
        print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

        self.net.train() if phase == "train" else self.net.eval()
        dataloader = self.dataloaders[phase]
        total_batches = len(dataloader)
        running_loss = 0.0
        self.optimizer.zero_grad()
        for itr, data_batch in enumerate(dataloader):
            images, targets = data_batch["data"], data_batch["label"]
            loss, logits = self._compute_loss_and_outputs(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches

        self.losses[phase].append(epoch_loss)
        print(f"Loss | {self.losses[phase][-1]}")

        return epoch_loss

    def run(self):
        for epoch in range(self.num_epochs):
            self._do_epoch(epoch, "train")
            with torch.no_grad():
                val_loss = self._do_epoch(epoch, "val")
                self.scheduler.step(val_loss)

            if val_loss < self.best_loss:
                print(f"\n{'#' * 20}\nSaved new checkpoint\n{'#' * 20}\n")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "autoencoder_best_model.pth")
            print()
        self._save_train_history()

    def load_predtrain_model(self, state_path: str):
        self.net.load_state_dict(torch.load(state_path))
        print("Predtrain model loaded")

    def _save_train_history(self):
        """Writing model weights and training logs to files."""
        torch.save(self.net.state_dict(), "autoencoder_last_epoch_model.pth")


def visualize_post_training(model: AutoEncoder) -> None:
    dataloader = get_dataloader(
        AutoEncoderDataset,
        "train_data.csv",
        phase="val",
        fold=0,
    )
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            id_, imgs, targets = data["Id"], data["data"], data["label"]
            imgs, targets = imgs.to("cuda"), targets.to("cuda")
            output = model(imgs)
            output = output.cpu()

            imgs = imgs.squeeze().cpu().numpy()
            imgs = np.moveaxis(imgs, (0, 1, 2, 3), (0, 3, 2, 1))
            print(imgs.shape)

            gt_flair, gt_t1, gt_t1ce, gt_t2 = imgs
            print(gt_flair.shape, gt_t1.shape, gt_t1ce.shape, gt_t2.shape)
            plt.figure(figsize=(15, 10))
            plt.imshow(np.rot90(montage(gt_flair)), cmap="bone")

            title = "AE_Ground_Truth_" + id_[0]
            filename1 = title + "_3d.gif"

            data_to_3dgif = Image3dToGIF3d(
                img_dim=(55, 55, 55),
                binary=False,
                normalizing=False,
            )
            transformed_data = data_to_3dgif.get_transformed_data(gt_flair)
            # transformed_data = np.rot90(transformed_data)
            data_to_3dgif.plot_cube(
                transformed_data[:38, :47, :35],
                title=title,
                make_gif=True,
                path_to_save=filename1,
            )

            output = output.squeeze().numpy()
            output = np.moveaxis(output, (0, 1, 2, 3), (0, 3, 2, 1))
            print(output.shape)

            pr_flair, pr_t1, pr_t1ce, pr_t2 = output
            print(pr_flair.shape, pr_t1.shape, pr_t1ce.shape, pr_t2.shape)

            plt.figure(figsize=(15, 10))
            pr_flair1 = pr_flair.copy()
            pr_flair1[pr_flair1 < 1e-7] = 0  # remove artifacts.
            plt.imshow(np.rot90(montage(pr_flair1)), cmap="bone")

            title = "AE_Prediction_" + id_[0]
            filename2 = title + "_3d.gif"

            data_to_3dgif = Image3dToGIF3d(
                img_dim=(55, 55, 55),
                binary=False,
                normalizing=False,
            )
            transformed_data = data_to_3dgif.get_transformed_data(pr_flair1)
            # transformed_data = np.rot90(transformed_data)
            data_to_3dgif.plot_cube(
                transformed_data[:38, :47, :35],
                title=title,
                make_gif=True,
                path_to_save=filename2,
            )

            merging_two_gif(filename1, filename2, "AE_result.gif")
            display(Image("AE_result.gif", format="png"))
            break


def main(config: GlobalConfig | None = None) -> None:
    if config is None:
        config = GlobalConfig(pretrained_model_path=None)
    model = AutoEncoder()
    if torch.cuda.is_available():
        model = model.to("cuda")
    trainer = Trainer(
        net=model,
        dataset=AutoEncoderDataset,
        criterion=nn.MSELoss(),
        lr=5e-4,
        accumulation_steps=4,
        batch_size=1,
        fold=0,
        num_epochs=1,
        path_to_csv=config.path_to_csv,
    )

    if config.ae_pretrained_model_path is not None:
        trainer.load_predtrain_model(config.ae_pretrained_model_path)

    trainer.run()
    visualize_post_training(model)


if __name__ == "__main__":
    main(config=default_config)
