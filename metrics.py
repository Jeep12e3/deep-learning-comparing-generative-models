import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += float(value) * count
        self.count += count

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def collect_images(loader, max_count: Optional[int] = None) -> torch.Tensor:
    batches = []
    collected = 0
    for images, _ in loader:
        remaining = None if max_count is None else max_count - collected
        batch = images if remaining is None else images[:remaining]
        batches.append(batch)
        collected += len(batch)
        if max_count is not None and collected >= max_count:
            break
    if not batches:
        raise ValueError("Data loader tidak menghasilkan gambar apa pun.")
    return torch.cat(batches)


def reconstruction_metrics(real: torch.Tensor, reconstructed: torch.Tensor) -> Tuple[float, float, float]:
    real_np = torch.as_tensor(real).detach().cpu().numpy()
    reconstructed_np = torch.as_tensor(reconstructed).detach().cpu().numpy()
    if real_np.ndim == 4:
        real_np = real_np[:, 0]
    if reconstructed_np.ndim == 4:
        reconstructed_np = reconstructed_np[:, 0]
    if real_np.shape != reconstructed_np.shape:
        raise ValueError(
            f"Array gambar harus memiliki bentuk yang sama; diperoleh {real_np.shape} dan "
            f"{reconstructed_np.shape}."
        )
    mse = float(np.mean((real_np - reconstructed_np) ** 2))
    psnr_values = [
        peak_signal_noise_ratio(a, b, data_range=1.0)
        for a, b in zip(real_np, reconstructed_np)
    ]
    ssim_values = [
        structural_similarity(a, b, data_range=1.0)
        for a, b in zip(real_np, reconstructed_np)
    ]
    return mse, float(np.mean(psnr_values)), float(np.mean(ssim_values))


def save_image_grid(
    images: torch.Tensor,
    title: str,
    path: Path,
    nrow: int = 8,
    figsize: Tuple[int, int] = (12, 3),
) -> None:
    images = torch.as_tensor(images).detach().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(1)
    if images.min() < 0:
        images = (images.clamp(-1, 1) + 1) / 2
    else:
        images = images.clamp(0, 1)
    rows = (len(images) + nrow - 1) // nrow
    fig, axes = plt.subplots(rows, nrow, figsize=(figsize[0], figsize[1] * rows))
    axes = np.asarray(axes).reshape(-1)
    for i, axis in enumerate(axes):
        axis.axis("off")
        if i < len(images):
            axis.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_losses(history: Dict[str, Iterable[float]], title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 5))
    labels = {
        "train": "Pelatihan",
        "validation": "Validasi",
        "train_loss": "Loss pelatihan",
        "validation_loss": "Loss validasi",
        "val_loss": "Loss validasi",
        "generator_loss": "Loss generator",
        "discriminator_loss": "Loss discriminator",
        "noise_prediction_mse": "MSE prediksi noise",
    }
    for name, values in history.items():
        axis.plot(list(values), label=labels.get(name, name))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Nilai loss")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def timed_inference(run, device: torch.device, warmup: int = 20, repeats: int = 100) -> float:
    """Kembalikan median waktu tiap pemanggilan tanpa gradien, dengan sinkronisasi perangkat."""
    durations = []
    with torch.no_grad():
        for _ in range(warmup):
            run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            run()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append((time.perf_counter() - started) * 1000)
    return float(np.median(durations))
