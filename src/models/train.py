import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_maker import GtzanDataset
from cnn import ConvoNetwork, VGG16
import matplotlib.pyplot as plt
import librosa.display
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

# Get Time Stamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Get the system information
sysinfo = torch.cuda.get_device_properties(
    0) if torch.cuda.is_available() else "CPU"


# Create a tensorboard writer
writer = SummaryWriter("runs/" + timestamp + "_" + sysinfo.name)

BATCH_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 0.0001
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE

# TODO: Implement way to visualize training stage
# TODO: Implement metrics
# TODO: Implement dynamic model generation
# TODO: Implement Cross Validation


def initialize_weights(m):
    """
    Initialize the weights randomly for the model

    Args:
        m (nn.Module): The model to initialize
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    """
    Train the model for one epoch

    Args:
        model (nn.Module): The model to train (CNN)
        data_loader (DataLoader): The data loader to use for training
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
        device (_type_): _description_
    """
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(data_loader):
        # Every data instance is a input and a label
        inputs, labels = data

        # Move the data to the device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients for every batch
        optimizer.zero_grad()

        # Make a predictions for this batch
        outputs = model(inputs)

        # Calculate the loss and backpropagate
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Update the weights (Learning)
        optimizer.step()

        # Gather data for reporting
        running_loss += loss.item()
        print(f"Batch {i+1}")
        if i % BATCH_SIZE == BATCH_SIZE - 1:
            last_loss = running_loss / BATCH_SIZE
            print(f"Batch {i+1} loss: {last_loss}")
            running_loss = 0.0

    return last_loss


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        t0 = time.perf_counter()
        print(f"Epoch {i+1}")
        model.train(True)
        last_loss = train_one_epoch(
            model, data_loader, loss_fn, optimizer, device)

        # We do not need gradients on to do reporting
        model.train(False)
        tb_x = i * len(data_loader.dataset) + i + 1
        writer.add_scalar("Loss/train", last_loss, tb_x)
        t1 = time.perf_counter()
        print(f"Epoch {i+1} took {t1-t0:.2f} seconds")
        print(" ------------------- ")
    writer.flush()

    print("Training finished")


def get_batch_create_img_grid(data_loader):
    """
    Extract one batch of data from the data loader

    Args:
        data_loader (DataLoader): The data loader to extract from
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # Get image size with torchvision with first image
    img_size = torchvision.transforms.functional.get_image_size(images[0])
    print(f"Image size: {img_size}")

    # Create a grid of images
    img_grid = torchvision.utils.make_grid(images)
    return img_grid


def visualize_model(model, data_loader):
    """
    Use Tensorboard to examine the data flow within the model.

    Args:
        model (nn.Module): The model to visualize
        data_loader (DataLoader): The data loader to use for visualization
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    writer.add_graph(model, images)
    writer.flush()


if __name__ == "__main__":

    ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

    ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
    GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the data
    print("Loading data...")
    print("NOTE: Genre dictionary is hardcoded in dataset_maker.py")
    print("Converted genres string labels to integer labels")
    print("Here is the genre dictionary: ")
    print("{'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}")
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_LOCAL,
                         genres_dir=GENRES_DIR_LOCAL, device=device)
    print("Data loaded")
    print("Size of dataset: ", len(gtzan))
    print("-------------------")

    print("-------------------")
    print("Splitting data into train, validation, and test sets")
    # Split the data into training, testing, and validation sets
    train_count = int(len(gtzan) * TRAINING_PERCENTAGE)  # 80% of data
    val_count = int(len(gtzan) * VALIDATION_PERCENTAGE)  # 20% of the data
    test_count = len(gtzan) - train_count - val_count  # 10% of the data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        gtzan, [train_count, val_count, test_count])
    print("Data split")
    print("-------------------")

    print("Creating data loader...")
    # Create a dataloader for the training, testing, and validation sets
    training_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE)

    val_data_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE)

    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE)
    print("Train Data Type", type(training_data_loader))

    print("Data loader created")
    print("-------------------")

    # Display a random sample from the dataset
    img_grid = get_batch_create_img_grid(training_data_loader)
    writer.add_image("Random Mel Spectrograms", img_grid)
    writer.flush()

    print("Creating model...")

    # Construct the model
    cnn = VGG16().to(device)
    print("Model created")
    print(cnn)
    print("-------------------")

    # Instantiate Loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Visualize the model
    visualize_model(cnn, training_data_loader)

    # Train the model
    train(cnn, training_data_loader, loss_fn, optimizer, device, EPOCHS)

    # Test the model

    # Validate models

    # Save the model
    # TODO: Save the model in results folder
    torch.save(cnn.state_dict(), "CNN.pth")
