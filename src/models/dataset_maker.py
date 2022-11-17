
import os
import numpy as np
from multiprocessing import Pool
import multiprocessing as mpr
import time
import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd


class GtzanDataset(Dataset):
    """
    Extract, Transform, Load (ETL) pipeline
    Extract: Extract audio files from genre directories
    Transform: Create mel spectrograms and one-hot vectors
    Load: Save data to output directory

    Advanced version uses multiprocessing to speed up ETL process
    """

    def __init__(self, annotations_file, genres_dir, device):
        """
        Constructor method to create blank training data list
        and genre dictionary.
        """
        self.device = device

        self.genres_dir = genres_dir
        self.annotations = pd.read_csv(annotations_file)

        self.genre_dict = {"blues": 0, "classical": 1, "country": 2, "disco": 3,
                           "hiphop": 4, "jazz": 5, "metal": 6, "pop": 7, "reggae": 8, "rock": 9}

        self.NUM_SAMPLES = 44100 * 30  # 30 seconds
        self.SAMPLE_RATE = 44100  # 44.1 kHz
        self.N_FFT = 1024  # Number of samples per frame
        self.HOP_LENGTH = 512  # Number of samples between successive frames
        self.N_MELS = 64  # Number of mel bands

    def _get_audio_file_path(self, audio_id):
        """Get the path to the audio file

        Args:
            audio_id (str): ID of the audio file

        Returns:
            str: Path to the audio file
        """
        genre = self.annotations.iloc[audio_id, 59]
        path = os.path.join(self.genres_dir, genre,
                            self.annotations.iloc[audio_id, 0])
        return path

    def _get_audio_label(self, audio_id):
        """Get the label of the audio file

        Args:
            audio_id (str): ID of the audio file

        Returns:
            str: Label of the audio file
        """
        return self.annotations.iloc[audio_id, 59]

    def __len__(self):
        return len(self.annotations)

    def _convert_str_label_to_one_hot(self, label: str):
        """
        Convert string label to one-hot vector

        Args:
            label (str): The genre of the audio file
        """
        return self.genre_dict[label]

    def __getitem__(self, idx):
        """
        a_list[1] -> a_list.__getitem__(1)
        Loading the waveform and the corresponding label at the given index

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        audio_file_path = self._get_audio_file_path(idx)
        label = self._get_audio_label(idx)
        signal, sr = torchaudio.load(audio_file_path)

        # Load signal to device: CPU or GPU
        signal = signal.to(self.device)

        signal = self._resample(signal, sr)
        signal = self._uniformize_to_mono(signal)
        signal = self._cut(signal)
        signal = self._right_pad(signal)

        # Generate mel spectrogram but on GPU or CPU
        signal = self._generate_mel_spectrogram(signal, sr)

        converted_label = self._convert_str_label_to_one_hot(label)
        return signal, converted_label

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
                orig_freq=sr, new_freq=self.SAMPLE_RATE).to(self.device)
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
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS).to(self.device)

        # MelSpectrogram is a class that can be called like a function.
        mel_img = mel_generator(signal)
        return mel_img

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
        print("First element of Training data:", self.training_data[0][0])


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR = "/home/zalasyu/Documents/467-CS/Data/genres_original"

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    gtzan = GtzanDataset(ANNOTATIONS_FILE, GENRES_DIR, device)
    print("Type of gtzan:", type(gtzan))

    print("Length of dataset:", len(gtzan))

    signal, label = gtzan[0]
    print("Signal:", signal)
    print("Label:", label)
