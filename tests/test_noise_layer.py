import torch
from noise.mixed_noise_layer import MixedNoiseLayer


def test_noise_layer_shape_and_ops():
    B, C, H, W = 2, 3, 64, 64
    x = torch.rand(B, C, H, W)
    noise_layer = MixedNoiseLayer(image_size=(H, W))
    noised, ops = noise_layer(x)
    assert noised.shape == (B, C, H, W)
    assert isinstance(ops, list) and len(ops) == B
    for op in ops:
        assert op in [
            "identity",
            "real_jpeg",
            "diff_jpeg",
            "blur",
            "noise",
            "scale_crop",
            "deepfake",
        ]
    assert noised.min().item() >= 0.0 and noised.max().item() <= 1.0
