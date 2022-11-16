import os
import numpy as np
from src.data.PrepareInput import PrepareAudio
from multiprocessing import Pool
from tqdm import tqdm
import time


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
                # Add image and label to training data
                label = np.eye(genre_count)[self.genre_dict[genre]]
                self.training_data.append([img, list(label)])

        # Shuffle and save dataset to designated output path
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)


class MusicTrainingDataAdvanced(MusicTrainingData):
    """
    Extract, Transform, Load (ETL) pipeline
    Extract: Extract audio files from genre directories
    Transform: Create mel spectrograms and one-hot vectors
    Load: Save data to output directory

    Advanced version uses multiprocessing to speed up ETL process
    """

    def etl_one_audio_file(self, genre: str, file: str, data_path: str, output_path: str):
        """ Process one audio file. Calls the transform and load functions.

        Args:
            genre (str): Name of genre directory
            file (str): Name of audio file
            data_path (str): Path to genre directory
            output_path (str): Path to output directory
        """
        start_t = time.perf_counter()
        mel_gen = PrepareAudio()

        # Transform
        # Create Mel Spectrogram
        mel_img = mel_gen.start(os.path.join(data_path, genre, file))

        # Create one-hot vector
        label = np.eye(len(self.genre_dict))[self.genre_dict[genre]]
        self.training_data.append([mel_img, list(label)])
        stop_t = time.perf_counter()
        return file, stop_t - start_t

    def _process_genre(self, genre: str, data_path: str, output_path: str):
        """ Process audio files in a genre directory
        This is a CPU-Bound task, so we use multiprocessing to speed up the process

        Args:
            genre (str): Name of genre directory
            data_path (str): Path to genre directory
            output_path (str): Path to output directory
        """

        audio_files = os.listdir(os.path.join(data_path, genre))

        # Create a pool of processes
        # The number of processes is the number of CPU cores
        # This is a CPU-Bound task
        print("Starting ETL process for genre: {}".format(genre))
        with Pool() as pool:
            # pool.starmap() will call the function with multiple arguments
            # starmap: Pass multiple arguments to the function
            # Here we create a list of tuples, where each tuple contains the arguments for one function call
            results = pool.starmap(
                self.etl_one_audio_file,
                [(genre, file, data_path, output_path)
                 for file in audio_files])

            # Waits for all processes to finish before continuing,
            # even though some processes may be done before others

            for filename, duration in results:
                print(f"{filename} took {duration:.2f} seconds")

    def make_training_data(self, data_path: str, output_path: str):
        """
        Creates numpy array of mel spectrograph and genre label 
        using the genre dictionary to create a one-hot vector. 

        Args:
            data_path (str)): _description_
            output_path (str): _description_
        """
        # Create genre dictionary
        self.create_genre_dictionary(data_path)

        # Get all genre directories
        genres = os.listdir(data_path)

        print("Starting ETL process...")
        # Process each genre directory
        for genre in genres:
            self._process_genre(genre, data_path, output_path)

        # Load data to output directory
        self.save_training_data(output_path)

    def save_training_data(self, output_path: str):
        """ Save training data to output directory

        Args:
            output_path (str): Path to output directory
        """
        print("Saving training data...")
        # Shuffle and save dataset to designated output path
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)
