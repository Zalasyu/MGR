import os
import numpy as np
# from src.data.PrepareInput import PrepareAudio
import json
from collections import deque
from tqdm import tqdm
import os.path                      # For managing directory files
import torch                        # Main ML framework
import torchaudio                   # Main audio processing framework
import matplotlib.pyplot as plt     # For displaying the graph
import librosa                      # For audio processing
import librosa.display              # For displaying spectrogram
import numpy as np                  # For utilizing a numpy array
from pydub import AudioSegment      # Audio format conversion


class PrepareAudio:
    """The PrepareAudio class processes audio inputs for the purpose of
    generating a mel spectrograph that will ultimately serve as the
    input for a machine learning model."""

    def __init__(self, spectrogram_length=2586, min_song_duration=29):
        """Constructor method for PrepareAudio class.
        No values except self.n_mels needs to be tweaked.
        Optional argument: spectrogram_length designates the length of the img for use in the training model.
        """
        # Paths for interim data
        self.DATA_PATH = 'data'
        self.INTERIM_DATA_PATH = os.path.join(self.DATA_PATH, 'interim')
        self.WAV_DATA_PATH = os.path.join(self.INTERIM_DATA_PATH, 'wav')
        self.PNG_DATA_PATH = os.path.join(self.INTERIM_DATA_PATH, 'png')

        # List of accepted file types (Feel free to add to this as you please)
        self.accepted_file_types = ['.mp3', '.mp4', '.au', '.wav']

        # For the purpose of naming the output file. Updated after start().
        self.file_name = None

        # Sample rate (Samples per second). Default = 44.1kHz aka 441000hz
        self.sr = 44100

        # Size of Fast Fourier Transform (FFT). Also used for window length
        self.n_fft = 2048

        # Step or stride between windows. The amount we are transforming
        # ...each fft to the right. Should be < than n_fft
        self.hop_length = 512

        # Number of mel bands (Your mel spectrogram willvary depending on
        # ...what value you specify here. Use power of 2: 32, 64 or 128)
        self.n_mels = 64

        # Length of the image array that is output for the model
        self.spectrogram_length = spectrogram_length

        # Minimum duration (in seconds) of a song clip for processing
        self.min_song_duration = min_song_duration

        # Transformer object used for wavefrom signal -> mel spectrogram
        self.transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

    def start(self, path):
        """Main driver function for the PrepareAudio class.
        *No other methods need to be directly called.
        Args: Path = path to audio file.
        Return: None
        """

        # 1. Check if file exists. If it does not exist,
        # return False and an error message
        if not self.check_file_exists(path):
            return False, "The specified file does not exist. Please try again."

        # 2. Check if the file is accepted
        if not self.check_file_type(path):
            return False, "The specified file is not accepted. Please try again."

        # Check that the file meets the minimum required song length
        if librosa.get_duration(filename=path) < self.min_song_duration:
            return False, f'The specified file must be of at least {self.min_song_duration} seconds in duration.'

        # 3. Convert audio file to .wav and save to /interim/wav directory (Does so
        # ...for ALL file types, even .wav for the purpose of uniformity)
        path = self.convert_to_wav(path)

        # 4. Get signal and sample rate of choosen audio file
        waveform, sr = torchaudio.load(path)  # type: ignore

        # self.plot_graph(signal, sr, 'Waveform')             # Optional
        # self.plot_graph(signal, sr, 'Vanilla Spectrogram')  # Optional

        # 5. Resample audio signal to match desired sample rate
        # ...if it doesnt already match
        if sr != self.sr:
            waveform = self.resample_signal(waveform, sr)

        # 6. Perform mel transformation on signal
        mel_spectrogram = self.transformer(waveform)
        # self.plot_melspectrogram(mel_spectrogram[0])  # Optional

        # 7. Save image to directory (This img will be the
        # ...output of this program and input of ML model)
        self.generate_melspec_png(mel_spectrogram[0])
        # Reshape the image array to match size of spectrogram_length
        spectrogram_aray = self.reshape_array(mel_spectrogram[0])
        return spectrogram_aray

    def reshape_array(self, mel_spectrogram):
        """Reshapes the mel spectrogram array to fit the specified spectrogram length.
        Normalizes the values in the array.
        Takes in original mel spectrogram array.
        Returns new array.
        """
        mel_spectrogram = np.array(mel_spectrogram)         # Convert to numpy array
        if len(mel_spectrogram[0]) > self.spectrogram_length:
            # Spectrogram too long, shorten it to the specified length
            mel_spectrogram = mel_spectrogram[:, :self.spectrogram_length]
        elif len(mel_spectrogram[0]) < self.spectrogram_length:
            # Spectrogram is too short, pad it with zeros
            pad_diff = self.spectrogram_length - len(mel_spectrogram[0])
            mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, pad_diff)))
        # Normalize the array to values between 0 and 1
        mel_spectrogram = mel_spectrogram / np.linalg.norm(mel_spectrogram)
        return mel_spectrogram.tolist()

    def check_file_exists(self, path):
        """Checks if the specified file exists.
        Args: Path = path to audio file.
        Return: True if file exists, False otherwise.
        """

        if os.path.exists(path):
            return True
        else:
            return False

    def check_file_type(self, path):
        """Check if the file type is acceptable.
        Args: Path = path to audio file.
        Return: True if file is accepted, False otherwise.
        """
        file_ext = os.path.splitext(path)[1]  # Extract file extension
        if file_ext in self.accepted_file_types:
            return True
        else:
            return False

    def convert_to_wav(self, path):
        """Converts the input audio file to .wav format and places
        in directory data/interim/wav/*.wav. Note, the file name is unchanged
        and only the format is converted.
        Args: Path = path to audio file.
        Return: Path to converted file
        """
        # Check if file is already a .wav extension
        root, ext = os.path.splitext(path)
        if ext == '.wav':
            # Already a wav file
            return path
        # Extract name of file from path variable
        path_parts = os.path.split(path)
        file_name_with_ext = path_parts[len(path_parts)-1]
        idx = len(file_name_with_ext) - file_name_with_ext.rindex('.')
        self.file_name = file_name_with_ext[:len(file_name_with_ext)-idx]
        path_to_wav = f'{self.WAV_DATA_PATH}/{self.file_name}.wav'

        # Create a wav/ folder if it doesn't already exist
        if not os.path.isdir(self.DATA_PATH):
            os.mkdir(self.DATA_PATH)
        if not os.path.isdir(self.INTERIM_DATA_PATH):
            os.mkdir(self.INTERIM_DATA_PATH)  # type: ignore
        if not os.path.isdir(self.WAV_DATA_PATH):
            os.mkdir(self.WAV_DATA_PATH)

        # Convert input file to .wav and save to wav/ directory
        AudioSegment.from_file(path).export(
            path_to_wav, format='wav')

        # Return path to converted audio file
        return path_to_wav

    def resample_signal(self, signal, sr):
        """Resamples the passed audio signal to a desired sample rate
        before returning to caller.
        Args: signal = waveform signal, sr = sample rate of signal.
        Return: signal with new sample rate.
        """

        sr_modifier = torchaudio.transforms.Resample(sr, self.sr)
        new_signal = sr_modifier(signal)
        return new_signal

    def plot_graph(self, signal, sr, title):
        """Displays the graph that you pass in (raw waveform or vanilla
        spectrogram). This function is only for visualization purposes
        and is not vital for processsing the audio file.
        Args: signal = waveform signal, sr = sample rate of signal.
        Return: None
        """

        signal = signal.numpy()

        num_channels, num_frames = signal.shape
        time_axis = torch.arange(0, num_frames) / sr

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            if title == "Waveform":
                axes[c].plot(time_axis, signal[c], linewidth=1)  # type: ignore
                axes[c].grid(True)  # type: ignore
            else:
                axes[c].specgram(signal[c], Fs=sr)  # type: ignore
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c+1}')  # type: ignore
        figure.suptitle(title)
        plt.show(block=True)

    def plot_melspectrogram(self, melspec):
        """Displays the mel spectrogram. This function is only for
        visualization purposes and is not vital for processsing the
        audio file.
        Args: melspec = mel spectrogram signal.
        Return: None
        """

        # Convert the power value to a decibel value
        melspec_db = librosa.power_to_db(melspec, ref=np.max)  # type: ignore

        librosa.display.specshow(
            melspec_db, sr=self.sr, hop_length=self.hop_length)

        plt.title("Mel spectrogram")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()

    def generate_melspec_png(self, melspec):
        """Saves the mel spectrogram as a .png to directory: interim/png/*_ms.png.
        Note, the output file name is identical to the input file name
        except with a '_ms" appended to the end.
        Args: melspec = mel spectrogram signal.
        Return: None
        """

        # Convert the power value to a decibel value
        melspec_db = librosa.power_to_db(melspec, ref=np.max)  # type: ignore
        librosa.display.specshow(
            melspec_db, sr=self.sr, hop_length=self.hop_length)
        plt.tight_layout()

        # Create interim/png/ directory if it doesnt already exist
        if not os.path.isdir(self.PNG_DATA_PATH):
            os.makedirs(self.PNG_DATA_PATH)  # type: ignore

        # Save mel spectrogram as .png to said directory
        # plt.savefig(f"{self.PNG_DATA_PATH}/{self.file_name}_ms.png")


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
        self.training_data = deque()
        self.genre_dict = {}

    def create_genre_dictionary(self, data_path, output_path):
        """
        Creates a dictionary using the names of the objects at a
        specified path. The objects should be directories that
        contain data for a single music genre (ex. Rap, Classical).
        """
        self.genre_dict = {}
        genre_count = 0
        for g in os.listdir(data_path):
            self.genre_dict[g] = genre_count
            genre_count += 1
        # Save dictionary to output path
        self.save_genre_dict(output_path)

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
        self.create_genre_dictionary(data_path, output_path)
        genre_count = len(self.genre_dict)
        mel_spectrogram = PrepareAudio()
        # Iterate through all genres
        for genre in self.genre_dict:
            # For each file in a genre
            for f in (pbar := tqdm(os.listdir(os.path.join(data_path, genre)))):
                # Progress bar description
                pbar.set_description(f"Processing {genre}")
                # Use Librosa to create a spectrograph - Midhun's code
                img = mel_spectrogram.start(os.path.join(data_path, genre, f))
                # Add image and label to training data
                label = np.eye(genre_count)[self.genre_dict[genre]]
                self.training_data.append([img, list(label)])

        # Shuffle and save dataset to designated output path
        self.training_data = np.array(self.training_data)
        np.random.shuffle(self.training_data)
        np.save(os.path.join(output_path, 'training_data.npy'), self.training_data)

    def save_genre_dict(self, output_path):
        """
        Saves the genre dictionary to a specified output path.
        """
        file_path = os.path.join(output_path, 'genre_dict.txt')
        with open(file_path, 'w') as dict_file:
            dict_file.write(json.dumps(self.genre_dict))
