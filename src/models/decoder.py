import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder network: extracts binary message from an image.
    """

    def __init__(self, msg_length):
        super(Decoder, self).__init__()
        self.msg_length = msg_length
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64, msg_length), nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B,3,H,W]
        feat = self.features(x)
        return self.classifier(feat)
