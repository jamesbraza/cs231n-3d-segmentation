import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from baseline.config import config
from data import BRATS_2020_TRAINING_FOLDER

survival_info_df = pd.read_csv(BRATS_2020_TRAINING_FOLDER / "survival_info.csv")
name_mapping_df = pd.read_csv(BRATS_2020_TRAINING_FOLDER / "name_mapping.csv")
name_mapping_df = name_mapping_df.rename({"BraTS_2020_subject_ID": "Brats20ID"}, axis=1)

all_df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

paths = []
for _, row in all_df.iterrows():
    id_ = row["Brats20ID"]
    phase = id_.split("_")[-2]

    if phase == "Training":
        path = os.path.join(config.train_root_dir, id_)
    else:
        path = os.path.join(config.test_root_dir, id_)
    paths.append(path)

all_df["path"] = paths

# split data on train, test, split
# train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=69, shuffle=True)
# train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

train_data = all_df.loc[all_df["Age"].notna()].reset_index(drop=True)
train_data["Age_rank"] = train_data["Age"] // 10 * 10
train_data = train_data.loc[
    train_data["Brats20ID"] != "BraTS20_Training_355"
].reset_index(drop=True)

skf = StratifiedKFold(n_splits=7, random_state=config.seed, shuffle=True)
for i, (_train_index, val_index) in enumerate(
    skf.split(train_data, train_data["Age_rank"]),
):
    train_data.loc[val_index, "fold"] = i

train_df = train_data.loc[train_data["fold"] != 0].reset_index(drop=True)
val_df = train_data.loc[train_data["fold"] == 0].reset_index(drop=True)
test_df = all_df.loc[~all_df["Age"].notna()].reset_index(drop=True)
print(
    "train_df ->",
    train_df.shape,
    "val_df ->",
    val_df.shape,
    "test_df ->",
    test_df.shape,
)
train_data.to_csv("train_data.csv", index=False)
