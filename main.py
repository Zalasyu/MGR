from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    """# Instructions: 
    1) 
    2) 

   """

    # 1. Initialize parameters/variables
    device = "cuda" if torch.cuda.is_available() else "cpu"
    validation_percentage = 0.2        # 20%, Percent of full dataset for validating
    test_percentage = 0.1              # 10%, Percent of full dataset for testing
    training_precentage = 1 - validation_percentage - test_percentage   # 70%, Percent of full dataset for training
    classes = 10    # How many genres of music we are using
    batch_size = 3  # How many spectrographs it uses per pass
    epochs = 3      # How many times it runs through all of the data
    learning_rate = 0.0001   # How fast it learns
    data_path = 'data/interim/wav'  # Path to folder containing sub-folders for each genre


    # 2. Construct tensor array from gitzan_dataset
    genre_dict = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
                           "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}
    music_training_data = MusicTrainingData(genre_dict, device)
    gitzan_dataset = music_training_data.make_training_data(data_path)


    # 3. Split dataset into 3 non-overlapping sets: training set, validation set, testing set
    train_count = int(len(gitzan_dataset) * training_precentage)  # 80% of data
    val_count = int(len(gitzan_dataset) * validation_percentage)  # 20% of the data
    test_count = len(gitzan_dataset) - train_count - val_count  # 10% of the data
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        gitzan_dataset, [train_count, val_count, test_count])
    print(f"Dataset sizes: Full={len(gitzan_dataset)}, Training={len(train_dataset)}, Validation={len(val_dataset)}, Testing={len(test_dataset)}")


    # 4. Create Dataloader for each set
    training_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    validation_data_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)


    # 5. Training...