import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dataset_maker import GtzanDataset
from cnn import VGG
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import math

# Multi-GPU support
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

TIMESTAMP = datetime.datetime.now().strftime("%m-%d-%Y--%H:%M")


WRITER = SummaryWriter()


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            gpu_id: int,
            save_every: int) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_data = train_data,
        self.val_data = val_data,
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        # Get  Batch Size from dataloader
        b_sz = self.train_data[0].batch_size
        print(
            f"[GPU {self.gpu_id}] Epoch {epoch} | Batch Size {b_sz} | Steps {len(self.train_data)}")
        for i, (source, targets) in enumerate(self.train_data[0]):
            self._run_batch(source.to(self.gpu_id), targets.to(self.gpu_id))

    def _save_checkpoint(self, epoch):
        checkpoint = self.model.state_dict()
        # Save the model
        model_name = self.model.get_model_name()

        # Get path to MGR/src/results directory
        model_filename = f"{model_name}_{TIMESTAMP}_{self.gpu_id}.pth"
        model_path = os.path.join(
            os.getcwd(),
            "saved_models",
            model_filename)
        torch.save(checkpoint, model_path)
        print(f"Saved checkpoint for epoch {epoch}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

# Load Train Objects
# The ingredients for training


def load_train_objs():

    # Load YOUR dataset
    train_set = GtzanDataset(ANNOTATIONS_FILE_LOCAL, GENRES_DIR_LOCAL)

    # load YOUR model
    model = VGG(VGG_type="VGG16")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    return train_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def main(device, total_epochs, save_every):
    dataset, model, optimizer, criterion = load_train_objs()
    train_data = prepare_dataloader(dataset, 32, True)

    # Get Batch Size from dataloader
    print(train_data.batch_size)

    trainer = Trainer(model, train_data, None, optimizer,
                      criterion, device, save_every)
    trainer.train(total_epochs)


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    device = 0  # Shorthand for cuda:0
    main(device, total_epochs, save_every)
