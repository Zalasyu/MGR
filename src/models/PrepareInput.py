""" This module prepares the audio file for further processing. """

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

    def __init__(self, device):
        """Constructor method for PrepareAudio class.
        No values except self.n_mels needs to be tweaked.
        Optional argument: spectrogram_length designates the length of the img for use in the training model.
        """

        # Device to store/run dataset
        self.device = device

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

        # For reshaping array
        self.NUM_SAMPLES = 44100 * 30  # 30 seconds

        # Transformer object used for wavefrom signal -> mel spectrogram
        self.transformer = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)

    def start(self, path):
        """Main driver function for the PrepareAudio class.
        *No other methods need to be directly called.
        Args: Path = path to a specific .wav audio file.
        Return: None
        """

        # 1. Check if file exists. If it does not exist,
        # ...print message and exit return
        if not self.check_file_exists(path):
            print("The specified file does not exist. Please try again.")
            return False

        # 2. Check if the file is accepted
        if not self.check_file_type(path):
            print("The specified file is not accepted. Please try again.")
            return False

        # 3. Converts audio file to .wav and save to /interim/wav directory
        # path = self.convert_to_wav(path)

        # 4. Get signal and sample rate of choosen audio file. 
        signal_tensor, sr = torchaudio.load(path)  # type: ignore

        # 5. Resample audio signal to match desired sample rate
        # ...if it doesnt already match
        if sr != self.sr:
            signal_tensor = self.resample_signal(signal_tensor, sr)
        
        # 6. Convert signal to mono (For uniformity)
        signal_tensor = self.uniformize_to_mono(signal_tensor)
        
        # 7. Reshape the image array to match size of spectrogram_length
        signal_tensor = self.reshape_array(signal_tensor).to(self.device)

        # 6. Perform mel transformation on signal
        mel_spectrogram = self.transformer(signal_tensor)
        # self.plot_melspectrogram(mel_spectrogram[0])  # Optional

        # 7. Save image to directory *Not vital to dataset creation, only for visualization
        #self.generate_melspec_png(mel_spectrogram[0])

        return mel_spectrogram

    def reshape_array(self, ms_tensor):
        """Reshapes the mel spectrogram tesnsor to fit the specified spectrogram length.
        Takes in original mel spectrogram tensor.
        Returns reshaped tenso.
        """

        # Spectrogram too long, shorten it to the specified length
        if ms_tensor.shape[1] > self.NUM_SAMPLES:
            # Cut the tensor to the desired length
            # via slicing [start:stop:step] -> [start:stop] (In this case)
            ms_tensor = ms_tensor[:, :self.NUM_SAMPLES]

        # Spectrogram is too short, pad it with zeros
        elif ms_tensor.shape[1] < self.NUM_SAMPLES:
            # Pad the tensor to the desired length [1, 1, 1] -> [1, 1, 1, 0, 0, 0]
            ms_tensor = torch.nn.functional.pad(
                ms_tensor, (0, self.NUM_SAMPLES - ms_tensor.shape[1]), "constant", 0)

        return ms_tensor     

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

    def uniformize_to_mono(self, ms_tensor):
        """
        Uniformizes the signal to mono
        Args: signal (Tensor)= audio signal
        Returns: signal (Tensor): audio signal
        """

        # If the signal is stereo, convert it to mono
        if ms_tensor.shape[0] > 1:

            # Mean the signal across the channels
            signal = torch.mean(ms_tensor, dim=0, keepdim=True)
        return ms_tensor

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
        plt.savefig(f"{self.PNG_DATA_PATH}/{self.file_name}_ms.png")


if __name__ == "__main__":
    # Instructions: Specify a valid path to desired audio file
    # for the 'file' variable and the class with take care of the rest!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_prepper = PrepareAudio(device)
    file = 'data/interim/wav/blues/blues.00000.wav'
    ms_tensor = audio_prepper.start(file)
    print(ms_tensor)
