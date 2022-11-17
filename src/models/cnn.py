from torch import nn
from torchsummary import summary

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
        # TODO: Quick and Dirty, need to make this more dynamic
        # TODO: Can dynamically create the layers
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
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Flatten the output of the convolutional layers
        self.flatten = nn.Flatten()

        # Linear layer 1: 512 nodes
        # 128 = Output channels from the last convolutional layer
        self.linear = nn.Linear(128 * 5 * 163, 10)

        # Softmax activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        """Forward pass through the network

        Args:
            input_data (_type_): _description_
        """
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # The output of the convolutional layers is flattened
        x = self.flatten(x)
        print(x.shape)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
