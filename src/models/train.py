import torch
import torchaudio
from torch import _nn
from torch.utils.data import DataLoader
import src.data.makedataset as makedataset
from cnn import CNN


if __name__ == "__main__":
