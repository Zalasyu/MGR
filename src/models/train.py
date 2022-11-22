import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_maker import GtzanDataset
from cnn import ConvoNetwork
import matplotlib.pyplot as plt
import librosa.display
import time

BATCH_SIZE = 125
EPOCHS = 10
LEARNING_RATE = 0.0001
VALIDATION_PERCENTAGE = 0.2
TEST_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE


def initialize_weights(m):
    """
    Initialize the weights randomly for the model

    Args:
        m (nn.Module): The model to initialize
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def display_training_progress(loss, accuracy, batch_size, n_batches):
    """
    Show the training progress

    Args:
        loss (float): The loss of the model
        accuracy (float): The accuracy of the model
        batch_size (int): The batch size
        n_batches (int): The number of batches
    """
    print(
        f"Loss: {loss.item():.4f} "
        f"[{batch_size * (n_batches + 1)}/{n_batches * batch_size} "
        f"({100. * (n_batches + 1) / n_batches:.0f}%)]\t"
        f"Accuracy: {accuracy / batch_size:.3f}"
    )


def calculate_accuracy(model, data_loader, device):
    """
    Calculate the accuracy of the model

    Args:
        model (nn.Module): The model to calculate accuracy for
        data_loader (DataLoader): The data loader to use for calculating accuracy
        device (_type_): _description_

    Returns:
        float: The accuracy of the model
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return 100 * correct / total


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
        print("Input: ", len(inputs), inputs)
        print("Target: ", len(targets), targets)

        # 1. Calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # 2. Backpropagation
        optimizer.zero_grad()  # When training, each batch we reset the gradients to zero
        loss.backward()        # Calculate the gradients (backpropagation)
        optimizer.step()       # Update the weights (Actual optimization step)

    # 3. Calculate accuracy
    accuracy = calculate_accuracy(model, data_loader, device)

    print(f"Loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        t0 = time.perf_counter()
        print(f"Epoch {i+1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        t1 = time.perf_counter()
        print(f"Epoch {i+1} took {t1-t0:.2f} seconds")
        print(" ------------------- ")

    print("Training finished")


def plot_melspecgrams(data_loader: DataLoader):
    """
    Display a mel spectrogram

    Args:
        data_loader (DataLoader): The data loader to use for displaying the spectrogram
    """
    fig, axs = plt.subplots()
    axs.set_title("Mel Spectrogram")
    axs.set_ylabel("Frequency")
    axs.set_xlabel("Time")
    # Display the mel spectrogram
    imgs, labels = next(iter(training_data_loader))
    print("train_features.shape: ", imgs.shape)
    img = imgs[0]
    img = img.cpu().numpy()
    print(img.shape)
    # Drop the channel dimension
    img = img[0, :, :]
    print(img.shape)
    img = axs.imshow(librosa.power_to_db(img), origin="lower", aspect="auto")
    fig.colorbar(img, ax=axs, format="%+2.0f dB")
    plt.show()


if __name__ == "__main__":

    ANNOTATIONS_FILE_LOCAL = "data/features_30_sec.csv"
    GENRES_DIR_LOCAL =  "data/interim/wav"

    #ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
    #GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

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


    print("Data loader created")
    print("-------------------")

    # Display a random sample from the dataset
    plot_melspecgrams(training_data_loader)

    print("Creating model...")

    # Construct the model
    cnn = ConvoNetwork().to(device)
    print("Model created")
    print(cnn)
    print("-------------------")

    # Instantiate Loss function and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(cnn, training_data_loader, loss_fn, optimizer, device, EPOCHS)

    # Test the model

    # Validate models

    # Save the model
    # TODO: Save the model in results folder
    torch.save(cnn.state_dict(), "CNN.pth")
