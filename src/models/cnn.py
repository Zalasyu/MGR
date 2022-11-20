from torch import nn
from torchsummary import summary
VGG_types = {
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
}


# TODO: Implement a way to choose the type of VGG
# TODO: Implement a way to choose the activation function
class VGG(nn.Module):
    """
    VGG16 architecture
    It has 16 weight layers
    Uses 3x3 kernels
    Padding of 1
    Stride of 1
    The number of features (Image resolution) always stays the same with VGG!
    VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
        'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    Then flatten and 4096*4096*4096 Linear layers
    """

    def __init__(self, in_channels=1, num_classes=10, img_height=64, img_width=2584):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types["VGG16"])
        self.name_of_model = "VGG16"

        self.height_out_after_conv = img_height // 2**5
        self.width_out_after_conv = img_width // 2**5

        self.fcs = nn.Sequential(
            nn.Linear(512*self.height_out_after_conv *
                      self.width_out_after_conv, 4096),
            nn.ReLU(),
            # Dropout Layer (This is a regularization technique)
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):

        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # Flatten the output of the convolutional layers
        x = x.reshape(x.shape[0], -1)

        # Pass the flattened output to the fully connected layers
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        # Programmatically create the convolutional layers
        # Use the architecture list to create the layers
        for x in architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1)
                    ),
                    nn.BatchNorm2d(x),  # Batch Normalization Layer
                    nn.ReLU()  # Activation Function
                ]
                in_channels = x
            elif x == 'M':
                # Pooling Layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        # Create Sequential Container and *layers unpacks the list
        # So the list is passed as arguments to the Sequential container
        # and the Sequential container persists the order of the layers
        return nn.Sequential(*layers)


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
