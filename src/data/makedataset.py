import os
import torch
from src.data.PrepareInput import PrepareAudio
from torch.utils.data import Dataset


class MusicTrainingData(Dataset):
    """
    The MusicTrainingData class generates a training dataset
    to be used on the neural network. This will output a NumPy
    array file (.npy) that contains a mel spectrograph and a
    label for the spectrograph as a one-hot vector to signal
    the correct genre.
    """

    def __init__(self, genre_dict, device):
        """
        Constructor method to create blank training data list
        and genre dictionary.
        """
        self.device = device
        self.training_data = []
        self.genre_dict = genre_dict

    def get_genre_dictionary(self):
        """
        Returns genre dictionary
        """
        return self.genre_dict

    def make_training_data(self, data_path):
        """
        Creates numpy array of mel spectrograph and genre label using
        the genre dictionary to create a one-hot vector. Processes
        all files within genre-labeled directories at a specified path.
        """
        
        mel_spectrogram = PrepareAudio(self.device)
        # Iterate through all genres
        for genre in self.genre_dict:
            # For each file in a genre
            for f in os.listdir(os.path.join(data_path, genre)):
                # Use Librosa to create a spectrograph - Midhun's code
                img = mel_spectrogram.start(os.path.join(data_path, genre, f))
                # Add image and label to training data
                label = self.genre_dict[genre]   # Returns int that correlates with genre
                self.training_data.append([img, label])

        # Return array of tensors in format: [[signal(tensor), label(int)], [...], ...]
        return self.training_data

        


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    genre_dict = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
     "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}
    x = MusicTrainingData(genre_dict, device)

    data_path = 'data/interim/wav'
    data = x.make_training_data(data_path)
    print(len(data))