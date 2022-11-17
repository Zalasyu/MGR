import train
from dataset_maker import GtzanDataset
from cnn import ConvoNetwork
import torch
import torchaudio

class_mapping = ["blues", "classical", "country", "disco",
                 "hiphop", "jazz", "metal", "pop", "reggae", "rock"]


def predict(model, input, target, class_mapping):
    """Predicts the genre of a song using the model"""
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    ANNOTATIONS_FILE_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/features_30_sec.csv"
    GENRES_DIR_CLOUD = "/nfs/stak/users/moldovaa/hpc-share/Data/genres_original"

    cnn = ConvoNetwork()
    state_dict = torch.load("CNN.pth")
    print("state_dict: ", state_dict)
    cnn.load_state_dict(state_dict)
    print("Model loaded")
    print(cnn)

    # LOAD gtzan dataset
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_CLOUD,
                         genres_dir=GENRES_DIR_CLOUD, device="cpu")

    # Get a song from the dataset for inference
    input_song, target = gtzan[0][0], gtzan[0][1]
    print("input_song: ", input_song)
    print("target: ", target)
    input_song.unsqueeze_(0)  # Add a batch dimension

    print("The state of the model is: " .format(cnn.training()))

    # Make prediction
    predicted, expected = predict(cnn, input_song, target, class_mapping)

    print(f"Predicted: {predicted}, Expected: {expected}")
