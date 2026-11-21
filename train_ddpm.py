import argparse

import pandas as pd
import torch

from config import CHECKPOINT_DIR, DEVICE, FIGURE_DIR, METRIC_DIR, SEED, ensure_output_dirs, set_seed
from data import fashion_mnist_loaders
from metrics import plot_losses, save_image_grid
from models import DDPM, TinyUNet
from training import train_ddpm


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih DDPM pada Fashion-MNIST.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=400)
    args = parser.parse_args()
    ensure_output_dirs()
    set_seed(SEED)
    loaders = fashion_mnist_loaders()
    model, diffusion, history, epoch_times = train_ddpm(
        loaders.train, epochs=args.epochs, timesteps=args.timesteps
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "timesteps": args.timesteps,
        },
        CHECKPOINT_DIR / "ddpm.pt",
    )
    plot_losses({"noise_prediction_mse": history}, "Loss Pelatihan DDPM", FIGURE_DIR / "ddpm_loss.png")
    samples, frames = diffusion.sample(
        16, capture_steps={args.timesteps - 1, 300, 200, 100, 50, 0}
    )
    save_image_grid(
        (samples * 0.5 + 0.5).clamp(0, 1),
        "Sampel hasil generasi DDPM",
        FIGURE_DIR / "ddpm_samples.png",
        nrow=4,
        figsize=(8, 8),
    )
    torch.save(frames, CHECKPOINT_DIR / "ddpm_trajectory.pt")
    pd.DataFrame(
        {"epoch": range(1, len(epoch_times) + 1), "seconds": epoch_times}
    ).to_csv(METRIC_DIR / "ddpm_training_times.csv", index=False)
    print(f"Checkpoint DDPM dan {len(frames)} frame lintasan berhasil disimpan.")


if __name__ == "__main__":
    main()
