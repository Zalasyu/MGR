from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    """
    xxx
    """

    # 1. Initialize parameters/variables
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    VALIDATION_PERCENTAGE = 0.2        # 20%, Percent of full dataset for validating
    TEST_PERCENTAGE = 0.1              # 10%, Percent of full dataset for testing
    TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE - TEST_PERCENTAGE   # 70%, Percent of full dataset for training
    CLASSES = 10    # How many genres of music we are using
    BATCH_SIZE = 3  # How many spectrographs it uses per pass
    EPOCHS = 3      # How many times it runs through all of the data
    LEARNING_RATE = 0.0001   # How fast it learns
    DATA_PATH = 'data/interim/wav'  # Path to folder containing sub-folders for each genre


    train_and_save = True  # Set to false if you already completd training and saved model (.pth)
    if train_and_save:
        # 2. Construct tensor array from gitzan_dataset
        genre_dict = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
                            "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}
        music_training_data = MusicTrainingData(genre_dict, DEVICE)
        gitzan_dataset = music_training_data.make_training_data(DATA_PATH)


        # 3. Split dataset into 3 non-overlapping sets: training set, validation set, testing set
        train_count = int(len(gitzan_dataset) * TRAINING_PERCENTAGE)  # 80% of data
        val_count = int(len(gitzan_dataset) * VALIDATION_PERCENTAGE)  # 20% of the data
        test_count = len(gitzan_dataset) - train_count - val_count  # 10% of the data
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            gitzan_dataset, [train_count, val_count, test_count])
        print(f"Dataset sizes: Full={len(gitzan_dataset)}, Training={len(train_dataset)}, Validation={len(val_dataset)}, Testing={len(test_dataset)}")


        # 4. Create Dataloader for each set
        training_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        validation_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


        # 5. Train model
        model = Model(BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_PERCENTAGE, "PLACEHOLDER", CLASSES, DEVICE)

    
        # 6. Save model (.pth)
