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

    ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"

    cnn = ConvoNetwork()
    state_dict = torch.load("CNN.pth")
    print("state_dict: ", state_dict)
    cnn.load_state_dict(state_dict)
    print("Model loaded")
    print(cnn)

    # LOAD gtzan dataset
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_LOCAL,
                         genres_dir=GENRES_DIR_LOCAL, device="cpu")

    # Make Predictions
    for i in range(10):
        input, target = gtzan[i]
        predicted, expected = predict(cnn, input, target, class_mapping)
        print("Predicted: ", predicted, "Expected: ", expected)
