import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS


from models.encoder import Encoder
from models.decoder import Decoder
from models.discriminator import Discriminator
from noise.mixed_noise_layer import MixedNoiseLayer
from utils.losses import Losses
import torch.nn.functional as F
from utils.evaluate import evaluate
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate every N epochs"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="BER threshold for Deepfake detection",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"E:\pages\InvMIHNet-master\InvMIHNet-master\datasets\DIV2K_train_HR",
        help="Path to training image folder",
    )
    parser.add_argument(
        "--msg_length", type=int, default=100, help="Length of the binary message"
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="Size of the input image"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for the optimizer"
    )
    parser.add_argument("--w_pixel", type=float, default=0.01, help="Pixel loss weight")
    parser.add_argument(
        "--w_feat", type=float, default=0.001, help="Feature (perceptual) loss weight"
    )
    parser.add_argument(
        "--w_extract", type=float, default=1000.0, help="Extraction loss weight"
    )
    parser.add_argument(
        "--w_adv", type=float, default=0.0, help="Adversarial loss weight for encoder"
    )
    parser.add_argument(
        "--adv_start_epoch",
        type=int,
        default=10,
        help="Epoch to start adversarial training",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="Number of epochs to ramp up extraction loss weight",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to YAML config file"
    )
    return parser.parse_args()


class SimpleImageFolder(torch.utils.data.Dataset):
    """Load images from a single directory without subfolders."""

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"Found 0 images in: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, 0


def main():
    args = parse_args()
    # Load configuration from YAML and override args
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    args.data_dir = cfg.get("train_data_dir", args.data_dir)
    args.val_data_dir = cfg.get("val_data_dir", getattr(args, "val_data_dir", None))
    args.batch_size = cfg.get("batch_size", args.batch_size)
    args.epochs = cfg.get("epochs", args.epochs)
    args.lr = cfg.get("lr", args.lr)
    args.msg_length = cfg.get("msg_length", args.msg_length)
    args.image_size = cfg.get("image_size", args.image_size)
    args.train_size = cfg.get("train_size", getattr(args, "train_size", 8000))
    args.val_size = cfg.get("val_size", getattr(args, "val_size", 2000))
    args.warmup_epochs = cfg.get("warmup_epochs", args.warmup_epochs)
    args.adv_start_epoch = cfg.get("adv_start_epoch", args.adv_start_epoch)
    args.w_pixel = cfg.get("w_pixel", args.w_pixel)
    args.w_feat = cfg.get("w_feat", args.w_feat)
    args.w_extract = cfg.get("w_extract", args.w_extract)
    args.w_adv = cfg.get("w_adv", args.w_adv)
    args.eval_every = cfg.get("eval_every", args.eval_every)
    args.threshold = cfg.get("threshold", args.threshold)
    # Debug: print effective configuration
    print(f"Config loaded: {cfg}")
    print(
        f"Training: {args.data_dir} ({args.train_size}), Validation: {args.val_data_dir} ({args.val_size})"
    )
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    # prepare dirs
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaders for training and validation
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
        ]
    )
    # full training dataset
    full_train = SimpleImageFolder(args.data_dir, transform=transform)
    train_size = min(args.train_size, len(full_train))
    train_dataset = Subset(full_train, list(range(train_size)))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    # validation dataset: use only first val_size samples
    full_val = SimpleImageFolder(args.val_data_dir, transform=transform)
    val_size = min(args.val_size, len(full_val))
    val_dataset = Subset(full_val, list(range(val_size)))
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Models
    encoder = Encoder(msg_length=args.msg_length).to(device)
    decoder = Decoder(msg_length=args.msg_length).to(device)
    noise_layer = MixedNoiseLayer(image_size=(args.image_size, args.image_size)).to(
        device
    )
    discriminator = Discriminator().to(device)

    # Loss and Optimizer
    losses = Losses(device=device)
    opt_enc = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    opt_dec = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Loss weights
    w_pixel = args.w_pixel
    w_feat = args.w_feat
    w_extract_max = args.w_extract
    w_adv = args.w_adv
    warmup_epochs = args.warmup_epochs
    adv_start = args.adv_start_epoch

    for epoch in range(1, args.epochs + 1):
        # dynamic extraction weight schedule
        w_extract = w_extract_max * min(1.0, epoch / warmup_epochs)
        is_warmup = epoch <= warmup_epochs
        # training progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, _ in pbar:
            images = images.to(device)
            B = images.size(0)
            msg = (torch.rand(B, args.msg_length, device=device) > 0.5).float()

            if is_warmup:
                # decoder warmup: freeze encoder, train decoder only on extraction loss
                with torch.no_grad():
                    encoded = encoder(images, msg)
                noised, ops = noise_layer(encoded)
                noised = noised.to(device)
                pred = decoder(noised)
                robust_ops = ["identity", "real_jpeg", "blur", "noise", "scale_crop"]
                mask = torch.tensor(
                    [op in robust_ops for op in ops], device=device
                ).float()
                bce = F.binary_cross_entropy(pred, msg, reduction="none")
                bce_flip = F.binary_cross_entropy(pred, 1 - msg, reduction="none")
                l_extract = (
                    mask.unsqueeze(1) * bce + (1 - mask).unsqueeze(1) * bce_flip
                ).mean()
                opt_dec.zero_grad()
                l_extract.backward()
                opt_dec.step()
                pbar.set_postfix({"L_extract": l_extract.item()})
                continue

            # encode
            encoded = encoder(images, msg)
            # compute visual losses
            l_pixel = losses.pixel_loss(images, encoded)
            l_feat = losses.perceptual_loss(images, encoded)

            # discriminator update (after adv_start)
            if epoch > adv_start and w_adv > 0:
                pred_real = discriminator(images)
                pred_fake_detach = discriminator(encoded.detach())
                loss_disc = losses.adversarial_loss(pred_real, pred_fake_detach)
                opt_disc.zero_grad()
                loss_disc.backward()
                opt_disc.step()
            else:
                loss_disc = 0

            # noise/distortion
            noised, ops = noise_layer(encoded)
            noised = noised.to(device)

            # decode for decoder update (detach encoder)
            pred_detach = decoder(noised.detach())
            # vectorized extraction loss for decoder
            robust_ops = ["identity", "real_jpeg", "blur", "noise", "scale_crop"]
            mask = torch.tensor([op in robust_ops for op in ops], device=device).float()
            bce = F.binary_cross_entropy(pred_detach, msg, reduction="none")
            bce_flip = F.binary_cross_entropy(pred_detach, 1 - msg, reduction="none")
            l_extract = (
                mask.unsqueeze(1) * bce + (1 - mask).unsqueeze(1) * bce_flip
            ).mean()
            # decode for encoder update (with gradients)
            pred = decoder(noised)
            # extraction loss for encoder
            bce_enc = F.binary_cross_entropy(pred, msg, reduction="none")
            bce_flip_enc = F.binary_cross_entropy(pred, 1 - msg, reduction="none")
            l_extract_enc = (
                mask.unsqueeze(1) * bce_enc + (1 - mask).unsqueeze(1) * bce_flip_enc
            ).mean()

            # generator adversarial loss (after adv_start)
            if epoch > adv_start and w_adv > 0:
                pred_fake = discriminator(encoded)
                l_adv = losses.generator_adv_loss(pred_fake)
            else:
                l_adv = 0

            # total loss
            loss_enc = (
                w_pixel * l_pixel
                + w_feat * l_feat
                + w_extract * l_extract_enc
                + w_adv * l_adv
            )
            loss_dec = w_extract * l_extract

            # gradient update: backward both losses first, then step to avoid in-place param updates before backward
            opt_dec.zero_grad()
            opt_enc.zero_grad()
            loss_dec.backward(retain_graph=True)
            loss_enc.backward()
            opt_dec.step()
            opt_enc.step()
            # update progress bar
            pbar.set_postfix(
                {
                    "L_pixel": l_pixel.item(),
                    "L_feat": l_feat.item(),
                    "L_extract": l_extract.item(),
                }
            )

        # end of epoch - skip summary and evaluation during warmup
        if not is_warmup:
            print(
                f"Epoch [{epoch}/{args.epochs}] L_pixel: {l_pixel.item():.4f} "
                f"L_feat: {l_feat.item():.4f} L_extract: {l_extract.item():.4f}"
            )
            if epoch % args.eval_every == 0:
                avg_psnr, avg_ssim, robust_ber, fragility_ber = evaluate(
                    encoder,
                    decoder,
                    noise_layer,
                    val_loader,
                    device,
                    threshold=args.threshold,
                )
                print(
                    f"Eval Epoch {epoch}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}, "
                    f"Robust BER={robust_ber:.4f}, Fragility BER={fragility_ber:.4f}"
                )

        # save checkpoint
        ckpt = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer_enc": opt_enc.state_dict(),
            "optimizer_dec": opt_dec.state_dict(),
        }
        torch.save(ckpt, os.path.join("checkpoints", f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()
