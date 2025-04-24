import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x))


class Encoder(nn.Module):
    """
    Encoder network: embeds binary message into an image.
    """

    def __init__(self, msg_length, image_size=(256, 256)):
        super(Encoder, self).__init__()
        self.msg_length = msg_length
        C = 3 + msg_length
        self.net = nn.Sequential(
            nn.Conv2d(C, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, image, msg):
        # image: [B,3,H,W], msg: [B,msg_length]
        B, _, H, W = image.shape
        # expand message to spatial map
        msg_map = msg.view(B, self.msg_length, 1, 1).expand(-1, -1, H, W)
        x = torch.cat([image, msg_map], dim=1)
        return self.net(x)
