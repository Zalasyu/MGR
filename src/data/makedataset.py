import os
import numpy as np
from src.data.PrepareInput import PrepareAudio


class MusicTrainingData:
    """
    The MusicTrainingData class generates a training dataset
    to be used on the neural network. This will output a NumPy
    array file (.npy) that contains a mel spectrograph and a
    label for the spectrograph as a one-hot vector to signal
    the correct genre.
    """

    def __init__(self):
        """
        Constructor method to create blank training data list
        and genre dictionary.
        """
        self.training_data = []
        self.genre_dict = {}

    def create_genre_dictionary(self, path):
        """
        Creates a dictionary using the names of the objects at a
        specified path. The objects should be directories that
        contain data for a single music genre (ex. Rap, Classical).
        """
        self.genre_dict = {}
        genre_count = 0
        for g in os.listdir(path):
            self.genre_dict[g] = genre_count
            genre_count += 1

    def get_genre_dictionary(self):
        """
        Returns genre dictionary
        """
        return self.genre_dict

    def make_training_data(self, data_path, output_path):
        """
        Creates numpy array of mel spectrograph and genre label using
        the genre dictionary to create a one-hot vector. Processes
        all files within genre-labeled directories at a specified path.
        """
        self.create_genre_dictionary(data_path)
        genre_count = len(self.genre_dict)
        mel_spectrogram = PrepareAudio()
        # Iterate through all genres
        for genre in self.genre_dict:
            # For each file in a genre
            for f in os.listdir(os.path.join(data_path, genre)):
                # Use Librosa to create a spectrograph - Midhun's code
                img = mel_spectrogram.start(os.path.join(data_path, genre, f))
                os.path.join(data_path, genre, f)
                # Add image and label to training data            
                self.training_data.append([np.array(img), np.eye(genre_count)[self.genre_dict[genre]]])
        # uncomment to look at lengths
        # for i in self.training_data:
        #     print(len(i[0][0][0]))

        # Shuffle and save dataset to designated output path
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)