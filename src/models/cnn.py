from torch import nn

# Define the CNN architecture
# VGG Type architecture with 3 convolutional layers and 2 fully connected layers
# The first convolutional layer has 32 filters, the second has 64, and the third has 128
# The first fully connected layer has 512 nodes, and the second has 10 nodes (one for each class)
# The output of the last fully connected layer is passed through a softmax activation function
# The output of the softmax activation function is the probability of the input belonging to each of the 10 classes
# The class with the highest probability is the predicted class
# The 10 classes are: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
# The input to the model is a 128x128x1 spectrogram
# The output of the model is a 10x1 vector of probabilities
# The model is trained using the Adam optimizer and the cross entropy loss function


class ConvoNetwork(nn.Module):
    """Simple VGG-like network for Music classification"""

    def __init__(self):
        super().__init__()
        # 4 convolutional blocks / flatten / linear / softmax

        # Convolutional block 1: 16 filters, 3x3 kernel, stride 1, padding 2
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
