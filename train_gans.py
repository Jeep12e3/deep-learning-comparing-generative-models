import argparse

import pandas as pd
import torch

from config import CHECKPOINT_DIR, DEVICE, FIGURE_DIR, METRIC_DIR, SEED, ensure_output_dirs, set_seed
from data import fashion_mnist_loaders
from metrics import plot_losses, save_image_grid
from models import DCDiscriminator, DCGenerator, VanillaDiscriminator, VanillaGenerator
from training import train_gan


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih GAN fully connected dan DCGAN.")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    ensure_output_dirs()
    loaders = fashion_mnist_loaders()
    training_times = []

    configurations = [
        ("vanilla_gan", VanillaGenerator(64), VanillaDiscriminator(), 64),
        ("dcgan", DCGenerator(100), DCDiscriminator(), 100),
    ]
    for name, generator, discriminator, latent_dim in configurations:
        set_seed(SEED)
        generator, _, history, epoch_times = train_gan(
            generator,
            discriminator,
            loaders.train,
            latent_dim=latent_dim,
            epochs=args.epochs,
        )
        torch.save(generator.state_dict(), CHECKPOINT_DIR / f"{name}_generator.pt")
        plot_losses(
            history,
            f"Kurva loss {name.replace('_', ' ').title()}",
            FIGURE_DIR / f"{name}_loss.png",
        )
        generator.eval()
        with torch.no_grad():
            samples = generator(torch.randn(16, latent_dim, device=DEVICE)).cpu()
        save_image_grid(
            samples,
            f"Sampel hasil generasi {name}",
            FIGURE_DIR / f"{name}_samples.png",
            nrow=4,
            figsize=(8, 8),
        )
        training_times.extend(
            {"model": name, "epoch": epoch, "seconds": seconds}
            for epoch, seconds in enumerate(epoch_times, start=1)
        )
    pd.DataFrame(training_times).to_csv(METRIC_DIR / "gan_training_times.csv", index=False)
    print("Checkpoint generator GAN berhasil disimpan.")


if __name__ == "__main__":
    main()
