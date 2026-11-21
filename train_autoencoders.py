import argparse

import pandas as pd
import torch

from config import CHECKPOINT_DIR, DEVICE, FIGURE_DIR, METRIC_DIR, SEED, ensure_output_dirs, set_seed
from data import fashion_mnist_loaders
from metrics import plot_losses
from models import SparseAE, VAE, VanillaAE
from training import train_autoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model AE, Sparse AE, dan VAE.")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    ensure_output_dirs()
    loaders = fashion_mnist_loaders()
    configurations = [
        ("vanilla_ae", lambda: VanillaAE(64), False),
        ("sparse_ae", lambda: SparseAE(sparse_dim=256, latent_dim=64), True),
        ("vae", lambda: VAE(64), False),
    ]
    times = []
    for name, create_model, sparse in configurations:
        set_seed(SEED)
        model = create_model()
        trained, history, epoch_times = train_autoencoder(
            model,
            loaders.train,
            loaders.validation,
            epochs=args.epochs,
            sparse=sparse,
            l1_weight=1e-3,
        )
        torch.save(trained.state_dict(), CHECKPOINT_DIR / f"{name}.pt")
        plot_losses(
            history,
            f"Kurva loss {name.replace('_', ' ').title()}",
            FIGURE_DIR / f"{name}_loss.png",
        )
        times.extend(
            {"model": name, "epoch": epoch, "seconds": seconds}
            for epoch, seconds in enumerate(epoch_times, start=1)
        )
        if sparse:
            trained.eval()
            with torch.no_grad():
                features = trained.encode_sparse_features(
                    next(iter(loaders.test))[0].to(DEVICE)
                )
            print(
                "Proporsi fitur sparse yang bernilai nol:",
                float((features.abs() < 1e-6).float().mean()),
            )
    pd.DataFrame(times).to_csv(METRIC_DIR / "autoencoder_training_times.csv", index=False)
    print("Checkpoint AE, Sparse AE, dan VAE berhasil disimpan.")


if __name__ == "__main__":
    main()
