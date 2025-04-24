import torch
from models.encoder import Encoder


def test_encoder_output_shape_and_range():
    B, C, H, W = 2, 3, 64, 64
    msg_length = 10
    img = torch.rand(B, C, H, W)
    msg = (torch.rand(B, msg_length) > 0.5).float()
    encoder = Encoder(msg_length=msg_length)
    out = encoder(img, msg)
    assert out.shape == (B, 3, H, W)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
