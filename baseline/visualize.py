import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from IPython.display import YouTubeVideo

from data import BRATS_2020_TRAINING_FOLDER

TRAINING_001 = BRATS_2020_TRAINING_FOLDER / "BraTS20_Training_001"

sample_filename = TRAINING_001 / "BraTS20_Training_001_flair.nii"
sample_filename_mask = TRAINING_001 / "BraTS20_Training_001_seg.nii"

sample_img = nib.load(sample_filename)
sample_img = np.asanyarray(sample_img.dataobj)
sample_img = np.rot90(sample_img)
sample_mask = nib.load(sample_filename_mask)
sample_mask = np.asanyarray(sample_mask.dataobj)
sample_mask = np.rot90(sample_mask)
print("img shape ->", sample_img.shape)
print("mask shape ->", sample_mask.shape)

sample_filename2 = TRAINING_001 / "BraTS20_Training_001_t1.nii"
sample_img2 = nib.load(sample_filename2)
sample_img2 = np.asanyarray(sample_img2.dataobj)
sample_img2 = np.rot90(sample_img2)

sample_filename3 = TRAINING_001 / "BraTS20_Training_001_t2.nii"
sample_img3 = nib.load(sample_filename3)
sample_img3 = np.asanyarray(sample_img3.dataobj)
sample_img3 = np.rot90(sample_img3)

sample_filename4 = TRAINING_001 / "BraTS20_Training_001_t1ce.nii"
sample_img4 = nib.load(sample_filename4)
sample_img4 = np.asanyarray(sample_img4.dataobj)
sample_img4 = np.rot90(sample_img4)

mask_WT = sample_mask.copy()
mask_WT[mask_WT == 1] = 1
mask_WT[mask_WT == 2] = 1
mask_WT[mask_WT == 4] = 1

mask_TC = sample_mask.copy()
mask_TC[mask_TC == 1] = 1
mask_TC[mask_TC == 2] = 0
mask_TC[mask_TC == 4] = 1

mask_ET = sample_mask.copy()
mask_ET[mask_ET == 1] = 0
mask_ET[mask_ET == 2] = 0
mask_ET[mask_ET == 4] = 1
# https://matplotlib.org/3.3.2/gallery/images_contours_and_fields/plot_streamplot.html#sphx-glr-gallery-images-contours-and-fields-plot-streamplot-py
# https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
fig = plt.figure(figsize=(20, 10))

gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])

#  Varying density along a streamline
ax0 = fig.add_subplot(gs[0, 0])
flair = ax0.imshow(sample_img[:, :, 65], cmap="bone")
ax0.set_title("FLAIR", fontsize=18, weight="bold", y=-0.2)
fig.colorbar(flair)

#  Varying density along a streamline
ax1 = fig.add_subplot(gs[0, 1])
t1 = ax1.imshow(sample_img2[:, :, 65], cmap="bone")
ax1.set_title("T1", fontsize=18, weight="bold", y=-0.2)
fig.colorbar(t1)

#  Varying density along a streamline
ax2 = fig.add_subplot(gs[0, 2])
t2 = ax2.imshow(sample_img3[:, :, 65], cmap="bone")
ax2.set_title("T2", fontsize=18, weight="bold", y=-0.2)
fig.colorbar(t2)

#  Varying density along a streamline
ax3 = fig.add_subplot(gs[0, 3])
t1ce = ax3.imshow(sample_img4[:, :, 65], cmap="bone")
ax3.set_title("T1 contrast", fontsize=18, weight="bold", y=-0.2)
fig.colorbar(t1ce)

#  Varying density along a streamline
ax4 = fig.add_subplot(gs[1, 1:3])

# ax4.imshow(
#     np.ma.masked_where(mask_WT[:, :, 65] == False, mask_WT[:, :, 65]),
#     cmap="summer",
#     alpha=0.6,
# )
l1 = ax4.imshow(
    mask_WT[:, :, 65],
    cmap="summer",
)
l2 = ax4.imshow(
    np.ma.masked_where(mask_TC[:, :, 65] is False, mask_TC[:, :, 65]),
    cmap="rainbow",
    alpha=0.6,
)
l3 = ax4.imshow(
    np.ma.masked_where(mask_ET[:, :, 65] is False, mask_ET[:, :, 65]),
    cmap="winter",
    alpha=0.6,
)
ax4.set_title("", fontsize=20, weight="bold", y=-0.1)

_ = [ax.set_axis_off() for ax in [ax0, ax1, ax2, ax3, ax4]]

colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]
labels = ["Non-Enhancing Tumor Core", "Peritumoral Edema ", "Gd-Enhancing Tumor"]
patches = [
    mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))
]
# put those patched as legend-handles into the legend
plt.legend(
    handles=patches,
    bbox_to_anchor=(1.1, 0.65),
    loc=2,
    borderaxespad=0.4,
    fontsize="xx-large",
    title="Mask Classes",
    title_fontsize=18,
    edgecolor="black",
    facecolor="#c5c6c7",
)
plt.suptitle("Multimodal Scans - Data", fontsize=20, weight="bold")

fig.savefig(
    "data_sample.png",
    format="png",
    pad_inches=0.2,
    transparent=False,
    bbox_inches="tight",
)
fig.savefig(
    "data_sample.svg",
    format="svg",
    pad_inches=0.2,
    transparent=False,
    bbox_inches="tight",
)


def youtube_embed() -> None:
    """Play https://youtu.be/nrmizEvG8aM when run from IPython."""
    YouTubeVideo("nrmizEvG8aM", width=600, height=400)
