import torch
import torch.nn as nn
import lpips


class Losses:
    def __init__(self, device="cpu"):
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        # LPIPS perceptual loss requires inputs in [-1,1]
        self.perceptual = lpips.LPIPS(net="alex").to(device)

    def pixel_loss(self, cover, encoded):
        return self.mse(cover, encoded)

    def perceptual_loss(self, cover, encoded):
        # scale to [-1,1]
        return self.perceptual((cover * 2 - 1), (encoded * 2 - 1)).mean()

    def message_loss(self, msg, pred, robust=True):
        if robust:
            return self.bce(pred, msg)
        else:
            # maximize error: target is ones (message flipped)
            target = 1 - msg
            return self.bce(pred, target)

    def adversarial_loss(self, pred_real, pred_fake):
        # standard GAN loss for discriminator
        real_loss = self.bce(pred_real, torch.ones_like(pred_real))
        fake_loss = self.bce(pred_fake, torch.zeros_like(pred_fake))
        return real_loss + fake_loss

    def generator_adv_loss(self, pred_fake):
        # adversarial loss for encoder to fool discriminator
        return self.bce(pred_fake, torch.ones_like(pred_fake))
