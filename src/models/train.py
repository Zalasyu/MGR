import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from dataset_maker import GtzanDataset
from cnn import ConvoNetwork

BATCH_SIZE = 125
EPOCHS = 10
LEARNING_RATE = 0.001


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

    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs)
        print(targets)

        # 1. Calculate loss
        predictions = model(inputs)
        print("predictions: ", predictions)
        loss = loss_fn(predictions, targets)

        # 2. Backpropagation
        optimizer.zero_grad()  # At each batch we reset the gradients to zero
        loss.backward()  # Calculate the gradients (backpropagation)
        optimizer.step()  # Update the weights

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print(" ------------------- ")

    print("Training finished")


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
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_CLOUD,
                         genres_dir=GENRES_DIR_CLOUD, device=device)
    print("Data loaded")
    print(gtzan)

    print("Creating data loader...")
    # Create a dataloader for the training data
    training_data_loader = DataLoader(
        gtzan, batch_size=BATCH_SIZE)
    print("Data loader created")

    print("Creating model...")
    # Construct the model
    cnn = ConvoNetwork().to(device)
    print("Model created")
    print(cnn)

    # Instantiate Loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(cnn, training_data_loader, loss_fn, optimizer, device, EPOCHS)

    # Save the model
    torch.save(cnn.state_dict(), "CNN.pth")
