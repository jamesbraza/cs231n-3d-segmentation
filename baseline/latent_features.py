import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from tqdm import tqdm

from baseline.autoencoder_model import AutoEncoder
from baseline.config import config
from baseline.data_preprocess import all_df
from baseline.unet_model import UNet3d


class LatentFeaturesGenerator:
    def __init__(self, autoencoder, device: str = "cuda"):
        self.autoencoder = autoencoder.to(device)
        self.device = device

    def __call__(self, img):
        with torch.no_grad():
            img = torch.FloatTensor(img).unsqueeze(0).to(self.device)
            return (
                self.autoencoder.encode(img, return_partials=False)
                .squeeze(0)
                .cpu()
                .numpy()
            )


class FeaturesGenerator:
    def __init__(self, df, autoencoder):
        self.df = df
        self.df_voxel_stats = pd.DataFrame()
        self.latent_feature_generator = LatentFeaturesGenerator(autoencoder)

    def _read_file(self, file_path):
        data = nib.load(file_path)
        return np.asarray(data.dataobj).astype(np.float32)

    def _normalize(self, data: np.ndarray):
        """Normilize image value between 0 and 1."""
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def _create_features(self, Brats20ID):  # noqa: N803
        features = {}
        images = []
        # vOXEL STATS
        for data_type in ["_t1.nii", "_t2.nii", "_flair.nii", "_t1ce.nii"]:
            # data path
            root_path = self.df.loc[self.df["Brats20ID"] == Brats20ID][
                "path"
            ].to_numpy()[0]
            file_path = os.path.join(root_path, Brats20ID + data_type)

            # flatten 3d array
            img_data = self._read_file(file_path)
            data = img_data.reshape(-1)

            # create features
            data_mean = data.mean()
            data_std = data.std()
            intensive_data = data[data > data_mean]
            more_intensive_data = data[data > data_mean + data_std]
            non_intensive_data = data[data < data_mean]

            data_skew = stats.skew(data)
            data_kurtosis = stats.kurtosis(data)
            intensive_skew = stats.skew(intensive_data)
            non_intensive_skew = stats.skew(non_intensive_data)

            data_diff = np.diff(data)

            # write new features in df
            features["Brats20ID"] = Brats20ID
            features[f"{data_type}_skew"] = (data_skew,)
            features[f"{data_type}_kurtosis"] = (data_kurtosis,)
            features[f"{data_type}_diff_skew"] = (stats.skew(data_diff),)
            features[f"{data_type}_intensive_dist"] = (intensive_data.shape[0],)
            features[f"{data_type}_intensive_skew"] = (intensive_skew,)
            features[f"{data_type}_non_intensive_dist"] = (non_intensive_data.shape[0],)
            features[f"{data_type}_non_intensive_skew"] = (non_intensive_skew,)
            # features[f"{data_type}_intensive_non_intensive_mean_ratio"] = (
            #     intensive_data.mean() / non_intensive_data.mean(),
            # )
            # features[f"{data_type}_intensive_non_intensive_std_ratio"] = (
            #     intensive_data.std() / non_intensive_data.std(),
            # )
            features[f"{data_type}_data_intensive_skew_difference"] = (
                data_skew - intensive_skew,
            )
            features[f"{data_type}_data_non_intensive_skew_difference"] = (
                data_skew - non_intensive_skew,
            )
            features[f"{data_type}_more_intensive_dist"] = (
                more_intensive_data.shape[0],
            )

            parts = 15
            for p, part in enumerate(np.array_split(data, parts)):
                features[f"{data_type}_part{p}_mean"] = part.mean()

            # Latent Features
            img = self._normalize(img_data)
            images.append(img.astype(np.float32))

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        latent_features = self.latent_feature_generator(img)

        for i, lf in enumerate(latent_features):
            features[f"latent_f{i}"] = lf

        return pd.DataFrame(features)

    def run(self):
        for _, row in tqdm(self.df.iterrows()):
            ID = row["Brats20ID"]

            df_features = self._create_features(ID)

            self.df_voxel_stats = pd.concat([self.df_voxel_stats, df_features], axis=0)

        self.df_voxel_stats.reset_index(inplace=True, drop=True)  # noqa: PD002
        self.df_voxel_stats = self.df_voxel_stats.merge(
            self.df[["Brats20ID", "Age", "Survival_days"]],
            on="Brats20ID",
            how="left",
        )


def save_latent_features(model: AutoEncoder | UNet3d) -> None:
    model.eval()
    fg = FeaturesGenerator(all_df, model)
    fg.run()
    fg.df_voxel_stats.to_csv("df_with_voxel_stats_and_latent_features.csv", index=False)
    print(fg.df_voxel_stats)


def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))


NUM_FOLDS = 7


def visualize_latent_features(num_folds: int = NUM_FOLDS) -> None:
    df = pd.read_csv("df_with_voxel_stats_and_latent_features.csv")  # noqa: PD901

    df["is_train"] = 0
    df["is_train"].loc[df["Survival_days"].notna()] = 1

    df["SD"] = df["Survival_days"].str.extract(r"(\d+[.\d]*)")
    df["SD"] = df["SD"].astype("float64")
    df["Age"] = df["Age"].astype("float64")
    df.sample(5)
    test_df = df[df["is_train"] is not True].copy()
    df = df[df["is_train"] is True].copy()  # noqa: PD901
    print("train ->", df.shape, "test ->", test_df.shape)
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.countplot(df["Age"].apply(lambda x: np.round(x, 0)), ax=ax, palette="Dark2")

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_ylabel("number of unique (rounded) ages", fontsize=20)
    ax.set_xlabel("unique (rounded) ages", fontsize=20)
    ax.set_title(
        "Distribution of rounded Ages in data",
        fontsize=25,
        y=1.05,
        fontweight="bold",
    )
    fig, ax = plt.subplots(figsize=(20, 10))
    k = 10
    sns.countplot(
        df["SD"].apply(lambda x: int(k * round(float(x) / k))),
        ax=ax,
        palette="Dark2",
    )  # base * round(float(x)/base)

    ax.set_xticks(ax.get_xticks()[::2])
    ax.set_ylabel(
        "number of unique (rounding to the nearest {k}) Survival_days",
        fontsize=15,
    )
    ax.set_xlabel(f"unique (rounding to the nearest {k}) Survival_days", fontsize=17)
    ax.set_title(
        "Distribution of rounded Survival_days in data",
        fontsize=25,
        y=1.05,
        fontweight="bold",
    )

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config.seed)

    features = list(df.columns[1:-4])

    overal_score = 0
    for target, c, w in [("Age", 100, 0.5), ("SD", 5, 0.5)]:
        y_oof = np.zeros(df.shape[0])
        y_test = np.zeros((test_df.shape[0], num_folds))

        for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):
            train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
            train_df = train_df[train_df[target].notna()]

            model = SVR(C=c, cache_size=3000.0)
            model.fit(train_df[features], train_df[target])

            y_oof[val_ind] = model.predict(val_df[features])
            y_test[:, f] = model.predict(test_df[features])

        df[f"pred_{target}"] = y_oof
        test_df[target] = y_test.mean(axis=1)
        score = metric(
            df[df[target].notna()][target].values,
            df[df[target].notna()][f"pred_{target}"].values,
        )
        overal_score += w * score
        print(target, np.round(score, 4))
        print()

    print("Overall score:", np.round(overal_score, 4))
