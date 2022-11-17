from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model
import torch
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

    quit()
    # 2. Initialize parameters (Tweak following value appropriately)
    classes = 10              # Number of genres in dataset
    batch_size = 3            # Slice of data that will be passed into model at a time
    epochs = 100               # Specifies number of runs through dataset
    learning_rate = 0.0001    # Rate of optimization (How fast it learns)
    validation_percent = 0.1  # Percent of sliced dataset that will be used for validating/testing
    data_path = "data/processed/training_data.npy"   # Path to dataset
    dict_path = "data/processed/genre_dict.txt"      # Path to genre dictionary
    model_path = 'src/models/mgr_model.pth'
    model = Model(batch_size, epochs, learning_rate, validation_percent, data_path, dict_path)

    # 3. Train model
    train_model = True
    if train_model:
        # Train model
        print("Initiating model training.... ")
        t0 = time.perf_counter()  # Used for timing
        model.train_model()
        t1 = time.perf_counter()
        print(f"Model training completed in {round(t1-t0, 2)} seconds.")   # Used for timing

