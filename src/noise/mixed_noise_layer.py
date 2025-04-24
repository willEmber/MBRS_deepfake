import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io
import random
from torchvision.transforms import GaussianBlur as TVGaussianBlur


class MixedNoiseLayer(nn.Module):
    """
    Mixed Noise/Distortion Layer for MBRS-DF.
    Randomly applies one of: identity, real JPEG, differentiable JPEG, Gaussian blur,
    Gaussian noise, scaling & cropping.
    """

    def __init__(self, jpeg_quality=(30, 70), image_size=(256, 256)):
        super(MixedNoiseLayer, self).__init__()
        self.jpeg_quality = jpeg_quality
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()
        # Other distortions
        self.blur = TVGaussianBlur(kernel_size=(3, 3), sigma=(1.0, 1.0))
        # diff_jpeg support removed due to import issues
        self.image_size = image_size
        # operation probabilities: identity, real_jpeg, blur, noise, scale_crop, deepfake
        self.ops = ["identity", "real_jpeg", "blur", "noise", "scale_crop", "deepfake"]
        self.ops_probs = [0.2, 0.2, 0.2, 0.2, 0.15, 0.05]

    def forward(self, x):
        # x: [B, C, H, W], values in [0,1]
        out = []
        ops_list = []
        for img in x:
            # sample operation with specified probabilities
            op = random.choices(self.ops, weights=self.ops_probs, k=1)[0]
            ops_list.append(op)
            if op == "identity":
                img_noised = img
            elif op == "real_jpeg":
                # Real JPEG compression via PIL
                pil = self.to_pil(img.cpu().clamp(0, 1))
                buffer = io.BytesIO()
                q = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
                pil.save(buffer, format="JPEG", quality=q)
                buffer.seek(0)
                dec = Image.open(buffer)
                img_noised = self.to_tensor(dec).to(img.device)
            elif op == "blur":
                # Gaussian blur
                img_noised = self.blur(img)
            elif op == "noise":
                noise = torch.randn_like(img) * 0.01
                img_noised = (img + noise).clamp(0, 1)
            elif op == "scale_crop":
                # Random scaling and crop/pad to original size
                scale = random.uniform(0.9, 1.1)
                H, W = self.image_size
                th, tw = int(H * scale), int(W * scale)
                # resize
                img_resized = torch.nn.functional.interpolate(
                    img.unsqueeze(0),
                    size=(th, tw),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                # center crop or pad
                if scale >= 1.0:
                    top = (th - H) // 2
                    left = (tw - W) // 2
                    img_noised = img_resized[:, top : top + H, left : left + W]
                else:
                    pad_h = H - th
                    pad_w = W - tw
                    pad_top = pad_h // 2
                    pad_left = pad_w // 2
                    pad_bottom = pad_h - pad_top
                    pad_right = pad_w - pad_left
                    img_noised = torch.nn.functional.pad(
                        img_resized.unsqueeze(0),
                        (pad_left, pad_right, pad_top, pad_bottom),
                        mode="reflect",
                    )[0]
            elif op == "deepfake":
                # Regional strong destruction (simulate Deepfake region replacement)
                img_noised = img.clone()
                h, w = self.image_size
                mh, mw = int(0.3 * h), int(0.3 * w)
                top = random.randint(0, h - mh)
                left = random.randint(0, w - mw)
                # fill region with noise
                img_noised[:, top : top + mh, left : left + mw] = torch.rand(
                    img_noised[:, top : top + mh, left : left + mw].shape,
                    device=img.device,
                )
            else:
                img_noised = img
            out.append(img_noised)
        return torch.stack(out, dim=0), ops_list
