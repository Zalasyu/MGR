import os
import numpy as np


class MusicTrainingData:
    """
    The MusicTrainingData class generates a training dataset
    to be used on the neural network. This will output a NumPy
    array file (.npy) that contains a mel spectrograph and a
    label for the spectrograph as a one-hot vector to signal
    the correct genre.
    """

    training_data = []
    genre_dict = {}

    def __create_genre_dictionary(self, path):
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

    def make_training_data(self, path):
        """
        Creates numpy array of mel spectrograph and genre label using
        the genre dictionary to create a one-hot vector. Processes
        all files within genre-labeled directories at a specified path.
        """
        self.__create_genre_dictionary(path)
        genre_count = len(self.genre_dict)
        # Iterate through all genres
        for genre in self.genre_dict:
            # For each file in a genre
            for f in os.listdir(path + "/" + genre):
                # Use Librosa to create a spectrograph - Midhun's code
                img = []
                # Add image and label to training data
                self.training_data.append([np.array(img), np.eye(genre_count)[self.genre_dict[genre]]])

        # Shuffle and save dataset
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)