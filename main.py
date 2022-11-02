from src.data.makedataset import MusicTrainingData
from src.models.train_model import Model

if __name__ == "__main__":
    """# Instructions: We have 2 sections, 
    1) creating the spectograms and numpy array
    2) building the model
    The model's variables below can be adjusted to suit the user."""
    
    # uncomment below to create spectrograms and numpy array 
   
    music_training_data = MusicTrainingData()
    data_path = 'data/raw'
    output_path = 'data/processed'
    music_training_data.make_training_data(data_path, output_path)


    # uncomment below to build model

    # classes are how many genres of music we are using
    classes = 8
    # batch size is how many spectrographs it uses per pass
    batch_size = 3
    # epochs are how many times it runs through all of the data
    # more epochs will make our model fit the training data more
    epochs = 3
    # learning rate is how fast it learns
    learning_rate = 0.0001
    # validation sample size
    validation_percent = 0.1
    # training data path
    training_data = "data/processed/training_data.npy"

    model = Model(batch_size, epochs, learning_rate, validation_percent, training_data)
    model.train_model(classes)
