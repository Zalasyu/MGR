from torch import nn
import torch
from torchsummary import summary
from src.data.dataset_maker import GtzanDataset


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
    A Dynamic VGG model that can be used to create any VGG model
    You can make a VGG11, VGG13, VGG16, VGG19
    VGG11: 11 layers
    VGG13: 13 layers
    VGG16: 16 layers
    VGG19: 19 layers


    """

    def __init__(self, in_channels=1, num_classes=10, img_height=128, img_width=2584, VGG_type="VGG16"):
        """
        Initialize the VGG model

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 1.
            num_classes (int, optional): _description_. Defaults to 10.
            img_height (int, optional): _description_. Defaults to 64.
            img_width (int, optional): _description_. Defaults to 2584.
            VGG_type (str, optional): _description_. Defaults to "VGG16".
        """
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv_layers(VGG_types[VGG_type])
        self.name_of_model = VGG_type

        self.max_pool_count = self._get_number_of_max_pools(
            VGG_types[VGG_type])

        self.height_out_after_conv = img_height // 2**self.max_pool_count
        self.width_out_after_conv = img_width // 2**self.max_pool_count

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

    def get_model_name(self):
        return self.name_of_model

    def _get_number_of_max_pools(self, architecture):
        return architecture.count('M')

    def forward(self, x):

        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # Flatten the output of the convolutional layers
        x = nn.Flatten()(x)

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


if __name__ == """__main__""":
    ANNOTATIONS_FILE_LOCAL = "/home/zalasyu/Documents/467-CS/Data/features_30_sec.csv"
    GENRES_DIR_LOCAL = "/home/zalasyu/Documents/467-CS/Data/genres_original"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG(VGG_type="VGG16").to(device)
    print(model)
    gtzan = GtzanDataset(annotations_file=ANNOTATIONS_FILE_LOCAL,
                         genres_dir=GENRES_DIR_LOCAL)

    train_dataloader = torch.utils.data.DataLoader(
        gtzan, batch_size=1, shuffle=True)

    # Get one batch of data
    input_tensor = next(iter(train_dataloader))[0].to(device)
    print(input_tensor.shape)

    summary(model, input_size=(1, 64, 2584))
