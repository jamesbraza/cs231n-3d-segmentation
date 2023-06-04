# cs231n-3d-segmentation

[Stanford CS231N Deep Learning for Computer Vision][1]: Class Project

## Dataset

We used the [BraTS2020 Dataset (Training + Validation)][5] dataset from Kaggle.
All iterations of the BraTS challenge can be found [here][3].

Here's how to easily download the dataset with the [Kaggle API][4]:

```console
kaggle datasets download -p data/brats2020-training-validation-data --unzip awsaf49/brats20-dataset-training-validation
```

## Development

Development began in spring 2023 using Python 3.11.

### AWS

AWS granted access to G and VT instances, and at the time,
AWS's Deep Learning AMI supported G3, P3, P3dn, P4d, P4de, G5, G4dn instances.
Thus, the AMI used was
Deep Learning AMI GPU PyTorch 2.0.0 (Ubuntu 20.04) 20230530 ([release notes][6])
with instance type `g4dn.2xlarge` and 60 GiB of gp3 storage.

Step 1: check GPU is present.

```shell
nvcc --version  # cuda_11.8
nvidia-smi  # NVIDIA Tesla T4
sudo apt update && sudo apt upgrade -y
sudo apt install -y ubuntu-drivers-common alsa-utils
ubuntu-drivers devices  # Drivers: nvidia-driver-525, nvidia-driver-525-server
```

Step 2: install and configure Python 3.10.

```shell
python3 --version  # 3.8.10
sudo apt update && sudo apt upgrade -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.10 python3.10-dev python3.10-venv
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2
python3 --version  # 3.10.11
```

Step 3: `git clone` and install requirements into a `venv`.

```shell
git clone https://github.com/jamesbraza/cs231n-3d-segmentation.git
cd cs231n-3d-segmentation
python3 -m venv venv
source venv/bin/activate
python -m pip install --no-cache-dir --progress-bar off -r requirements.txt
```

Step 4: download BraTS 2020 dataset using the Kaggle API.

```shell
# Starting from non-VM
export SEG01=111.222.333.444
scp -pr ~/.kaggle/ ubuntu@$SEG01:~/.kaggle/
ssh ubuntu@$SEG01
cd cs231n-3d-segmentation
source venv/bin/activate
kaggle datasets download -p data/brats2020-training-validation-data \
    --unzip awsaf49/brats20-dataset-training-validation
```

[1]: http://cs231n.stanford.edu/
[3]: https://www.med.upenn.edu/cbica/brats/
[4]: https://github.com/Kaggle/kaggle-api
[5]: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
[6]: https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-pytorch-2-0-ubuntu-20-04/
