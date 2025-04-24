import torch
from models.discriminator import Discriminator


def test_discriminator_output_shape_and_range():
    B, C, H, W = 2, 3, 64, 64
    img = torch.rand(B, C, H, W)
    disc = Discriminator()
    out = disc(img)
    # Discriminator outputs a value per sample
    assert out.shape == (B,)
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0
