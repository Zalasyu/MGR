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
from tqdm import tqdm

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


def ddp_setup(rank, world_size):
    """
    Initialize the distributed environment.
    Args:
        rank (_type_): A unique identifier for each process.
        world_size (_type_): Total number of processes in a group
    """

    # MASTER b/c it coordinates the communications between the other processes
    # IP address of the machine that is running the rank 0
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # nccl is a communication backend that is used for multi-GPU training
    init_process_group("nccl", rank=rank, world_size=world_size)


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

        # Wrap the model with DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        # Get  Batch Size from dataloader
        b_sz = len(next(iter(self.train_data))[0])
        print(
            f"[GPU {self.gpu_id}] Epoch {epoch} | Batch Size {b_sz} | Steps {len(self.train_data)}")
        for i, (source, targets) in enumerate(self.train_data):
            self._run_batch(source.to(self.gpu_id), targets.to(self.gpu_id))

    def _save_checkpoint(self, epoch):
        # We need to access module since it was wrapped with DDP
        checkpoint = self.model.module.state_dict()
        # Save the model
        model_name = self.model.module.get_model_name()

        # Get path to MGR/src/results directory
        model_filename = f"{model_name}_{TIMESTAMP}_Epoch{epoch}.pth"
        model_path = os.path.join(
            os.getcwd(),
            "saved_models",
            model_filename)
        torch.save(checkpoint, model_path)
        print(f"Saved checkpoint for epoch {epoch}")

    def train(self, max_epochs: int):
        for epoch in tqdm(range(max_epochs)):
            self._run_epoch(epoch)

            # Save only from master process since all other nodes will have the same model
            if epoch % self.save_every == 0 and self.gpu_id == 0:
                self._save_checkpoint(epoch)

# Load Train Objects
# The ingredients for training


def load_train_objs():

    # Load YOUR dataset
    train_set = GtzanDataset(ANNOTATIONS_FILE_CLOUD, GENRES_DIR_CLOUD)

    # load YOUR model
    # Types of VGG available: VGG11, VGG13, VGG16, VGG19
    model = VGG(VGG_type="VGG16")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    return train_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:

    # Create a DistributedSampler to handle distributing the dataset across nodes
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, worldsize: int, total_epochs, save_every):
    ddp_setup(rank, worldsize)
    dataset, model, optimizer, criterion = load_train_objs()
    train_data = prepare_dataloader(dataset, 32)
    trainer = Trainer(model, train_data, None, optimizer,
                      criterion, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    world_size = torch.cuda.device_count()  # Shorthand for cuda:0
    mp.spawn(main, args=(world_size, total_epochs, save_every),
             nprocs=world_size)
