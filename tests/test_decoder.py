import torch
from models.decoder import Decoder


def test_decoder_output_shape_and_range():
    B, C, H, W = 2, 3, 64, 64
    msg_length = 10
    img = torch.rand(B, C, H, W)
    decoder = Decoder(msg_length=msg_length)
    out = decoder(img)
    assert out.shape == (B, msg_length)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
