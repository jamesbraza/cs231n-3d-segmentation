import torch


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")  # Current CUDA device
    return torch.device("cpu")
