import train
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
    cnn = ConvoNetwork()
    state_dict = torch.load("CNN.pth")
    cnn = cnn.load_state_dict(state_dict)

    # LOAD gtzan dataset
    gtzan = train.GtzanDataset(annotations_file=train.ANNOTATIONS_FILE_CLOUD,
                               genres_dir=train.GENRES_DIR_CLOUD, device="cpu")

    # Get a song from the dataset for inference
    input, target = gtzan[0][0], gtzan[0][1]
    input.unsqueeze_(0)  # Add a batch dimension

    # Make prediction
    predicted, expected = predict(cnn, input, target, class_mapping)
    print(f"Predicted: {predicted}, Expected: {expected}")
