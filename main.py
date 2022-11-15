from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model
import time   # Used for timimg

if __name__ == "__main__":
    """Main driver for utilization of the machine model for the 
    Music Genre Classifier program."""
    
    # 1. Construct dataset (skip if you already have a .npy dataset)
    make_dataset = True   # Set to True to skip step
    if make_dataset:
        print("Initiating dataset construction.... ")
        t0 = time.perf_counter()
        music_training_data = MusicTrainingData()
        data_path = 'data/raw'
        output_path = 'data/processed'
        music_training_data.make_training_data(data_path, output_path)
        t1 = time.perf_counter()
        print(f"Dataset construction completed in {round(t1-t0, 2)} seconds.")  # Used for timing


    # 2. Specify parameters (Tweak following value appropriately)
<<<<<<< HEAD
    classes = 10              # Number of genres in dataset
    batch_size = 5            # Slice of data that will be passed into model at a time
    epochs = 10               # Specifies number of runs through dataset
=======
    # classes = 10              # Number of genres in dataset NOT USED IN MODEL
    batch_size = 3            # Slice of data that will be passed into model at a time
    epochs = 100              # Specifies number of runs through dataset
>>>>>>> dd74b95e865d625b5e68d3e851fbd2f2d06a36a9
    learning_rate = 0.0001    # Rate of optimization (How fast it learns)
    validation_percent = 0.1  # Percent of sliced dataset that will be used for validating/testing
    data_path = "data/processed/training_data.npy"   # Path to dataset
    dict_path = "data/processed/genre_dict.txt"      # Path to genre dictionary

    # 3. Call model
    model = Model(batch_size, epochs, learning_rate, validation_percent, data_path, dict_path)
    print("Initiating model training.... ")
    t0 = time.perf_counter()  # Used for timing
    model.train_model()
    t1 = time.perf_counter()
    print(f"Model training completed in {round(t1-t0, 2)} seconds.")   # Used for timing
