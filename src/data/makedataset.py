import os
import numpy as np
from src.data.PrepareInput import PrepareAudio
from multiprocessing import Pool
import multiprocessing as mpr
import time
import torchaudio
import torch


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

    def __init__(self):
        """
        Constructor method to create blank training data list
        and genre dictionary.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.training_data = []
        self.genre_dict = {}
        self.NUM_SAMPLES = 44100 * 30  # 30 seconds
        self.SAMPLE_RATE = 44100  # 44.1 kHz
        self.N_FFT = 1024  # Number of samples per frame
        self.HOP_LENGTH = 512  # Number of samples between successive frames
        self.N_MELS = 64  # Number of mel bands

    def _right_pad(self, signal):
        """
        Right pad signal if necessary

        Args:
            signal (Tensor): audio signal

        Returns:
            signal (Tensor): audio signal
        """
        # signal -> Tensor -> (num_channels, num_samples) -> (1, num_samples)
        if signal.shape[1] < self.NUM_SAMPLES:

            # Pad the signal to the desired length
            # [1, 1, 1] -> [1, 1, 1, 0, 0, 0]
            signal = torch.nn.functional.pad(
                signal, (0, self.NUM_SAMPLES - signal.shape[1]), "constant", 0)

        return signal

    def _cut(self, signal):
        """
        Cut signal if necessary

        Args:
            signal (Tensor): audio signal

        Returns:
            signal (Tensor): audio signal
        """
        # signal -> Tensor -> (num_channels, num_samples) -> (1, num_samples)
        if signal.shape[1] > self.NUM_SAMPLES:

            # Cut the signal to the desired length
            # via slicing [start:stop:step] -> [start:stop] (In this case)
            signal = signal[:, :self.NUM_SAMPLES]

        return signal

    def _resample(self, signal, sr):
        """
        Resamples the signal to the specified sample rate

        Args:
            signal (Tensor): audio signal
            sr (int): sample rate

        Returns:
            signal (Tensor): audio signal
        """
        if sr != self.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(  # Resample to 44.1 kHz
                orig_freq=sr, new_freq=self.SAMPLE_RATE)
            signal = resampler(signal)
        return signal

    def _uniformize_to_mono(self, signal):
        """
        Uniformizes the signal to mono

        Args:
            signal (Tensor): audio signal

        Returns:
            signal (Tensor): audio signal
        """
        # If the signal is stereo, convert it to mono
        if signal.shape[0] > 1:

            # Mean the signal across the channels
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _generate_mel_spectrogram(self, signal, sr):
        """
        Creates a mel spectrogram from an audio signal

        Parameters
                signal: audio signal
                sr: sample rate

        Returns
                mel_img: mel spectrogram
        """
        mel_generator = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE, n_fft=self.N_FFT, hop_length=self.HOP_LENGTH, n_mels=self.N_MELS)

        # MelSpectrogram is a class that can be called like a function.
        mel_img = mel_generator(signal)
        return mel_img

    def _enable_gpu(self):
        """
        Enables GPU if available
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _append_data(self, data):
        """
        Appends data to training data list

        Parameters
                data: tuple of (img, label)
        """
        print("Size of training data before: {}".format(len(self.training_data)))
        self.training_data.append(data)
        print("Size of training data after: {}".format(len(self.training_data)))

    def etl_one_audio_file(self, genre: str, filename: str, data_path: str):
        """ Process one audio file. Calls the transform and load functions.

        Args:
            genre (str): Name of genre directory
            file (str): Name of audio file
            data_path (str): Path to genre directory
        """
        start_t = time.perf_counter()

        # Extract
        signal, sr = torchaudio.load(os.path.join(data_path, genre, filename))

        # Preprocess audio data
        signal = self._resample(signal, sr)
        signal = self._uniformize_to_mono(signal)
        signal = self._right_pad(signal)
        signal = self._cut(signal)

        # Transform
        # Create Mel Spectrogram
        mel_img = self._generate_mel_spectrogram(signal, sr)

        # Create one-hot vector:
        label = np.eye(len(self.genre_dict))[self.genre_dict[genre]]

        # Create a tuple of the data
        data = (mel_img, list(label))

        stop_t = time.perf_counter()
        print(
            f"Finished processing {filename} in {stop_t - start_t:.2f} seconds")

        return filename, data

    def _process_genre(self, genre: str, data_path: str):
        """ Process audio files in a genre directory
        This is a CPU-Bound task, so we use multiprocessing to speed up the process

        Args:
            genre (str): Name of genre directory
            data_path (str): Path to genre directory
        """

        audio_files = os.listdir(os.path.join(data_path, genre))

        # Create a pool of processes
        # The number of processes is the number of CPU cores
        # This is a CPU-Bound task
        print("Starting ETL process for genre: {}".format(genre))

        # Enable GPU
        # self._enable_gpu()

        core_count = mpr.cpu_count() - 1

        print("Number of cores: {}".format(core_count))
        with Pool(processes=core_count) as pool:
            # pool.starmap() will call the function with multiple arguments
            # starmap: Pass multiple arguments to the function
            # Here we create a list of tuples, where each tuple contains the arguments for one function call
            results = pool.starmap(
                self.etl_one_audio_file,
                [(genre, filename, data_path)
                 for filename in audio_files])

            # Waits for all processes to finish before continuing,
            # even though some processes may be done before others

            for filename, data in results:
                print(f"Saving {filename} to training data")
                self._append_data(data)
                print("Finished processing {}".format(filename))

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
            self._process_genre(genre, data_path)

        # Load data to output directory
        self.save_training_data(output_path)

    def save_training_data(self, output_path: str):
        """ Save training data to output directory

        Args:
            output_path (str): Path to output directory
        """
        print("Saving training data...")
        print("Size of Training data:", len(self.training_data))
        # Shuffle and save dataset to designated output path
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)
        self.analyze_training_dataset()

    def analyze_training_dataset(self):
        """ Analyze training dataset
        """
        print("Analyzing training dataset...")
        print("Size of Training data:", len(self.training_data))
        print("Type of Training data:", type(self.training_data))
        print("Shape of Training data:", self.training_data[0][0].shape)
        print("Shape of Training data:", self.training_data[0][1].shape)
        print("First element of Training data:", self.training_data[0][0])


def main():

    data_path = 'data'
    output_path = 'output'

    # Create training data object
    music_training_data = MusicTrainingDataAdvanced()

    # Create training data
    music_training_data.make_training_data(
        data_path='../../data/raw',
        output_path='../../data/processed')

    # Get genre dictionary
    genre_dict = music_training_data.get_genre_dictionary()

    # Print genre dictionary
    print(genre_dict)


if __name__ == "__main__":
    main()
