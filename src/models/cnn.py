from torch import nn
from torchsummary import summary


class ConvoNetwork(nn.Module):
    """Simple VGG16-like architecture"""

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
        # 5 = Pooling factor of the last convolutional layer
        # 128 *5 * 163 = 1049600
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
        print("After convolutions: ", x.shape)

        # The output of the convolutional layers is flattened
        x = self.flatten(x)
        print("Flattened: ", x.shape)
        logits = self.linear(x)
        print("After linear: ", logits.shape)
        predictions = self.softmax(logits)
        print("After softmax: ", predictions.shape)
        print("Predictions: ", predictions)
        return predictions


class VGG16(nn.Module):
    """VGG16 architecture"""

    def __init__(self):
        super().__init__()


class AlexNet(nn.Module):
    """AlexNet architecture"""

    def __init__(self):
        super().__init__()


class GoogleNet(nn.Module):
    """GoogleNet architecture"""

    def __init__(self):
        super().__init__()


class ResNet(nn.Module):
    """ResNet architecture"""

    def __init__(self):
        super().__init__()


class ZFNet(nn.Module):
    """ZFNet architecture"""

    def __init__(self):
        super().__init__()
