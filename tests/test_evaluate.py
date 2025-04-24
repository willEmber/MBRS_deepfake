import torch
import numpy as np
from utils.evaluate import compute_psnr, compute_ssim, compute_ber


def test_psnr_ssim_identical():
    img = torch.rand(3, 64, 64)
    # identical images: PSNR should be infinite or very large, SSIM == 1.0
    psnr = compute_psnr(img, img)
    assert psnr == float("inf") or psnr > 100
    assert np.isclose(compute_ssim(img, img), 1.0)


def test_ber():
    msg = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    pred = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    assert compute_ber(msg, pred, threshold=0.5) == 1.0
    pred2 = msg.clone()
    assert compute_ber(msg, pred2, threshold=0.5) == 0.0
