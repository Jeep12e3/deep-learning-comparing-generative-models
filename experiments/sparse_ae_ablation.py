import argparse
import time

import pandas as pd
import torch

from config import CHECKPOINT_DIR, DEVICE, METRIC_DIR, SEED, ensure_output_dirs, set_seed
from data import fashion_mnist_loaders
from metrics import reconstruction_metrics
from models import SparseAE
from training import train_autoencoder


def measure_model(model, loader, name, training_seconds=0.0):
    images_all = []
    reconstructions = []
    zero_counts = 0
    activation_counts = 0
    l1_per_image = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            features = model.encode_sparse_features(images)
            reconstruction = model(images)
            zero_counts += (features.abs() < 1e-6).sum().item()
            activation_counts += features.numel()
            l1_per_image.extend(features.abs().sum(dim=1).cpu().tolist())
            images_all.append(images.cpu())
            reconstructions.append(reconstruction.cpu())

    mse, psnr, ssim = reconstruction_metrics(
        torch.cat(images_all), torch.cat(reconstructions)
    )
    return {
        "model": name,
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "hidden_zero_fraction": zero_counts / activation_counts,
        "mean_hidden_L1_per_image": sum(l1_per_image) / len(l1_per_image),
        "mean_abs_weight": float(model.sparse_encoder[1].weight.abs().mean().item()),
        "sparse_encoder_weight_L2": float(
            torch.linalg.vector_norm(model.sparse_encoder[1].weight).item()
        ),
        "training_seconds": training_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bandingkan Sparse AE dengan penalti L1 dan baseline tanpa penalti."
    )
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    ensure_output_dirs()
    torch.set_num_threads(1)
    loaders = fashion_mnist_loaders()

    sparse = SparseAE(sparse_dim=256, latent_dim=64).to(DEVICE)
    sparse.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / "sparse_ae.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )
    rows = [measure_model(sparse, loaders.test, "Sparse AE, lambda=0.001")]

    set_seed(SEED)
    baseline = SparseAE(sparse_dim=256, latent_dim=64).to(DEVICE)
    started = time.perf_counter()
    baseline, _, epoch_times = train_autoencoder(
        baseline,
        loaders.train,
        loaders.validation,
        epochs=args.epochs,
        sparse=False,
        l1_weight=0.0,
    )
    training_seconds = time.perf_counter() - started
    torch.save(
        baseline.state_dict(), CHECKPOINT_DIR / "sparse_ae_no_l1.pt"
    )
    rows.append(
        measure_model(
            baseline,
            loaders.test,
            "Sparse AE baseline, lambda=0",
            training_seconds=training_seconds,
        )
    )
    pd.DataFrame(rows).to_csv(
        METRIC_DIR / "sparse_ae_ablation.csv", index=False
    )
    pd.DataFrame(
        {"epoch": range(1, len(epoch_times) + 1), "seconds": epoch_times}
    ).to_csv(METRIC_DIR / "sparse_ae_baseline_training_times.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
