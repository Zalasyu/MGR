import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_maker import GtzanDataset
from cnn import VGG
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime

# TODO: Implement way to visualize training stage
# TODO: Implement metrics
# TODO: Implement a way to dynamically create initial weights for model
# TODO: Implement Cross Validation


# Get Time Stamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Get the system information
sysinfo = torch.cuda.get_device_properties(
    0) if torch.cuda.is_available() else "CPU"


# Create a tensorboard writer
writer = SummaryWriter("runs/" + timestamp + "_" + sysinfo.name)

# Device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS
# number of data samples propagated through the network before parameters are updated
BATCH_SIZE = 25
EPOCHS = 1  # Number of times to iterate over the dataset
# How much to update the model parameters at each batch/epoch.
# NOTE: Smaller learning rate means slow learning speed, but more stable
LEARNING_RATE = 0.0001

# Data Spliting COnfiguration
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE

print("Creating model...")
# Construct the model
MODEL = VGG().to(DEVICE)
print("Model created")
print(MODEL)
print("-------------------")


def train_one_epoch(data_loader, loss_fn, optimizer):
    """
    Train the model for one epoch

    Args:
        data_loader (DataLoader): The data loader to use for training
        loss_fn (_type_): _description_
        optimizer (_type_): _description_
    """
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(data_loader):
        # Every data instance is a input and a label
        inputs, labels = data

        # Move the data to the device
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Zero the parameter gradients for every batch
        optimizer.zero_grad()

        # Make a predictions for this batch
        outputs = MODEL(inputs)

        # Calculate the loss and backpropagate
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Update the weights (Learning)
        optimizer.step()

        # Gather data for reporting
        running_loss += loss.item()
        # print(f"Batch {i+1} loss: {loss.item()}")
        if i % BATCH_SIZE == BATCH_SIZE - 1:
            last_loss = running_loss / BATCH_SIZE
            print(f"Batch {i+1} loss: {last_loss}")

            tb_x = i * len(data_loader) + i + 1

            writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def train_it_baby(train_dataloader, test_dataloader, loss_fn, optimizer):
    """
    Train and Report

    Args:
        model (_type_): _description_
        train_dataloader (_type_): _description_
        test_dataloader (_type_): _description_
        optimizer (_type_): _description_
        loss_fn (_type_): _description_
        epochs (_type_): _description_
    """
    epochs_num = 0

    # Validation loss
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):

        MODEL.train(True)
        avg_loss = train_one_epoch(train_dataloader, loss_fn, optimizer)
        MODEL.train(False)

        # Validation loss
        print("Validating...")
        running_vloss = 0.0
        for i, vdata in enumerate(test_dataloader):
            vinputs, vlabels = vdata

            # Send data to device
            vinputs, vlabels = vinputs.to(DEVICE), vlabels.to(DEVICE)

            # Make a predictions for this batch
            vloss = MODEL(vinputs)
            vloss = loss_fn(vloss, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        # LOGGING
        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars("Training vs Validation Loss",
                           {"Training Loss": avg_loss, "Validation Loss": avg_vloss}, epochs_num + 1)
        writer.flush()

        # If the validation loss is better than the best validation loss, save the model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_class_name = MODEL.__class__.__name__
            model_path = f"../results/{model_class_name}_{timestamp}_{sysinfo.name}_{epochs_num}.pth"
            torch.save(MODEL.state_dict(), model_path)
            print("Saved best model")
            print(f"Model saved to {model_path}")

        epochs_num += 1


def train(data_loader, loss_fn, optimizer):
    for i in range(EPOCHS):
        t0 = time.perf_counter()
        print(f"Epoch {i+1}")
        MODEL.train(True)
        last_loss = train_one_epoch(data_loader, loss_fn, optimizer)

        # We do not need gradients on to do reporting
        MODEL.train(False)
        tb_x = i * len(data_loader.dataset) + i + 1
        writer.add_scalar("Loss/train", last_loss, tb_x)
        t1 = time.perf_counter()
        print(f"Epoch {i+1} took {t1-t0:.2f} seconds")
        print(" ------------------- ")
    writer.flush()

    print("Training finished")


def test(data_loader, loss_fn):
    """
    Test the model

    Args:
        data_loader (DataLoader): The data loader to use for testing
        loss_fn (_type_): _description_
    """
    # We do not need gradients on to do reporting
    MODEL.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(data_loader):
        # Every data instance is a input and a label
        inputs, labels = data

        # Move the data to the device
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Make a predictions for this batch
        outputs = MODEL(inputs)

        # Calculate the loss and backpropagate
        loss = loss_fn(outputs, labels)

        # Gather data for reporting
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / (i + 1)
    accuracy = 100.0*correct / total

    print(f"Test Loss: {avg_loss}")
    print(f"Accuracy: {accuracy}")

    # LOGGING
    writer.add_scalar("Loss/test", avg_loss, 1)
    writer.flush()

    print("Testing finished")


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
    writer.add_image("Random Mel Spectrograms", img_grid)
    writer.flush()


def visualize_model(data_loader):
    """
    Use Tensorboard to examine the data flow within the model.

    Args:
        model (nn.Module): The model to visualize
        data_loader (DataLoader): The data loader to use for visualization
    """
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    writer.add_graph(MODEL, images)
    writer.flush()


if __name__ == "__main__":

    ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

    ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
    GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

    # Free GPU cache
    print(torch.cuda.memory_summary())

    # Load the data
    print("Loading data...")
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_CLOUD,
                         genres_dir=GENRES_DIR_CLOUD, device=DEVICE)
    print("Data loaded")
    print("Size of dataset: ", len(gtzan))
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

    print("Creating data loaders...")
    # Create a dataloader for the training, testing, and validation sets
    training_data_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_data_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Data loaders created")
    print("-------------------")

    print("SANITY CHECK")
    print("Reporting to Tensorboard Random Mel Spectrogram Images from Training Set")
    # Display a random sample from the dataset
    get_batch_create_img_grid(training_data_loader)
    print("-------------------")

    # Instantiate Loss function and Optimizer
    # CrossEntropyLoss is used for classification
    loss_fn = nn.CrossEntropyLoss()

    # Adam is a popular optimizer for deep learning
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)

    # Visualize the model
    visualize_model(training_data_loader)

    print("Training model...")
    # Train the model
    t_start = time.perf_counter()

    # train_it_baby(training_data_loader, test_data_loader, loss_fn, optimizer)
    train(training_data_loader, loss_fn, optimizer)

    t_end = time.perf_counter()
    print(f"Training took {t_end - t_start} seconds")
    print("Model trained")
    print("-------------------")

    # Test the model
    print("Testing model...")
    t_start = time.perf_counter()
    test(test_data_loader, loss_fn)
    print("Model tested")
    print("-------------------")

    # Save the model
    model_class_name = MODEL.__class__.__name__

    # Get working directory
    wdir = os.getcwd()
    print(f"Working directory: {wdir}")

    # Get path to MGR/src/results directory
    model_filename = f"{model_class_name}_{timestamp}_{sysinfo.name}.pt"
    model_path = os.path.join("../results" + model_filename)

    torch.save(MODEL.state_dict(), model_path)
    print("Saved best model")
    print(f"Model saved to {model_path}")
    print("-------------------")
