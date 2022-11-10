from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model
import time

if __name__ == "__main__":
    """xxxx"""
    
   
    t0 = time.perf_counter()                # Used for timing
    music_training_data = MusicTrainingData()
    data_path = 'data/raw'
    output_path = 'data/processed'
    music_training_data.make_training_data(data_path, output_path)
    t1 = time.perf_counter()
    print(f"Time(s) to build dataset: {round(t1-t0, 2)}")   # Used for timing

    # Tweak following value appropriately
    # Classes are how many genres of music we are using
    classes = 8
    # Batch size is how many spectrographs it uses per pass
    batch_size = 3
    # Epochs are how many times it runs through all of the data
    # more epochs will make our model fit the training data more
    epochs = 3
    # Learning rate is how fast it learns
    learning_rate = 0.0001
    # Validation sample size
    validation_percent = 0.1
    # Training data path
    data_path = "data/processed/training_data.npy"

    model = Model(batch_size, epochs, learning_rate, validation_percent, data_path, classes)

    t0 = time.perf_counter()  # Used for timing
    model.train_model(classes)
    t1 = time.perf_counter()
    print(f"Time(s) to train model: {round(t1-t0, 2)}")   # Used for timing
