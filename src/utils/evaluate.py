import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(img1, img2):
    # img1, img2: torch.Tensor [C,H,W] or numpy uint8/float [H,W,C]
    arr1 = img1.detach().cpu().numpy()
    arr2 = img2.detach().cpu().numpy()
    # if [C,H,W], transpose to [H,W,C]
    if arr1.ndim == 3:
        arr1 = np.transpose(arr1, (1, 2, 0))
        arr2 = np.transpose(arr2, (1, 2, 0))
    return peak_signal_noise_ratio(arr1, arr2, data_range=1.0)


def compute_ssim(img1, img2):
    arr1 = img1.detach().cpu().numpy()
    arr2 = img2.detach().cpu().numpy()
    if arr1.ndim == 3:
        arr1 = np.transpose(arr1, (1, 2, 0))
        arr2 = np.transpose(arr2, (1, 2, 0))
    # convert to grayscale for SSIM
    if arr1.shape[2] == 3:
        arr1 = np.dot(arr1[..., :3], [0.2989, 0.5870, 0.1140])
        arr2 = np.dot(arr2[..., :3], [0.2989, 0.5870, 0.1140])
    return structural_similarity(arr1, arr2, data_range=1.0)


def compute_ber(msg, pred, threshold=0.5):
    # msg, pred: torch.Tensor of shape [msg_length]
    pred_bin = (pred > threshold).float()
    return (msg != pred_bin).float().mean().item()


def evaluate(encoder, decoder, noise_layer, dataloader, device, threshold=0.5):
    """
    Evaluate robustness and fragility.
    Returns: (avg_psnr, avg_ssim, robust_ber, fragility_ber)
    """
    robust_ops = ["identity", "real_jpeg", "blur", "noise", "scale_crop"]
    robust_bers = []
    frag_bers = []
    psnrs = []
    ssims = []
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            B = images.size(0)
            msg = (torch.rand(B, encoder.msg_length, device=device) > 0.5).float()

            encoded = encoder(images, msg)
            # compute PSNR/SSIM on encoded
            for orig, enc in zip(images, encoded):
                psnrs.append(compute_psnr(orig, enc))
                ssims.append(compute_ssim(orig, enc))

            # robust BER
            for i in range(B):
                img = images[i : i + 1]
                m = msg[i]
                # sample until benign op
                while True:
                    noised, ops = noise_layer(img)
                    if ops[0] in robust_ops:
                        break
                p = decoder(noised)[0]
                robust_bers.append(compute_ber(m, p, threshold))
            # frag BER
            for i in range(B):
                img = images[i : i + 1]
                m = msg[i]
                # sample until deepfake op
                while True:
                    noised, ops = noise_layer(img)
                    if ops[0] == "deepfake":
                        break
                p = decoder(noised)[0]
                frag_bers.append(compute_ber(m, p, threshold))
    return np.mean(psnrs), np.mean(ssims), np.mean(robust_bers), np.mean(frag_bers)
