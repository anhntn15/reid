import torch
from torch import nn
import torch.nn.functional as F


class ConvReAE(torch.nn.Module):
    def __init__(self, img_num_channel=3):
        """
        init encoder & decoder module for the model
        :param img_num_channel: define input_channel for encoder's 1st layer and output_channel for decoder's last layer
        this param can be adjusted according to type of input image (1 for gray image, 3 for color image)
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_num_channel, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, img_num_channel, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvMaxReAE(torch.nn.Module):
    def __init__(self, img_num_channel=3):
        """
        init encoder & decoder module for the model
        :param img_num_channel: define input_channel for encoder's 1st layer and output_channel for decoder's last layer
        this param can be adjusted according to type of input image (1 for gray image, 3 for color image)
        """
        super().__init__()

        self.conv1 = nn.Conv2d(img_num_channel, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 7)
        self.pool = nn.MaxPool2d(2, return_indices=True)

        self.un_pool = nn.MaxUnpool2d(2)
        self.t_conv1 = nn.ConvTranspose2d(64, 32, 7)
        self.t_conv2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(16, img_num_channel, 3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x = F.relu(self.conv1(x))
        size1 = x.size()
        x, idx1 = self.pool(x)

        x = F.relu(self.conv2(x))
        size2 = x.size()
        x, idx2 = self.pool(x)

        x = F.relu(self.conv3(x))
        size3 = x.size()
        x, idx3 = self.pool(x)

        # decoder
        x = self.un_pool(x, idx3, output_size=size3)
        x = F.relu(self.t_conv1(x))

        x = self.un_pool(x, idx2, output_size=size2)
        x = self.sigmoid(self.t_conv2(x))

        x = self.un_pool(x, idx1, output_size=size1)
        x = self.sigmoid(self.t_conv3(x))

        return x
