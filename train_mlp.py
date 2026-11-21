import argparse

import pandas as pd
import torch

from config import CHECKPOINT_DIR, DEVICE, FIGURE_DIR, METRIC_DIR, SEED, ensure_output_dirs, set_seed
from data import fashion_mnist_loaders
from metrics import plot_losses
from training import train_mlp_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih pengklasifikasi MLP 784-256-64-10.")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    ensure_output_dirs()
    set_seed(SEED)
    loaders = fashion_mnist_loaders()
    model, history, epoch_times, _ = train_mlp_classifier(
        loaders.train, loaders.validation, epochs=args.epochs
    )

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loaders.test:
            logits, _ = model(images.to(DEVICE))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    print(f"Akurasi uji MLP: {test_accuracy:.2%}")
    if test_accuracy < 0.88:
        raise RuntimeError(
            "Akurasi uji MLP di bawah batas 88%; tambahkan epoch atau sesuaikan learning rate."
        )

    torch.save(model.state_dict(), CHECKPOINT_DIR / "mlp_classifier.pt")
    torch.save(
        {
            "W1": model.fc1.weight.detach().cpu(),
            "W2": model.fc2.weight.detach().cpu(),
            "W3": model.fc3.weight.detach().cpu(),
        },
        METRIC_DIR / "mlp_weights.pt",
    )
    plot_losses(
        {"train": history["train_loss"], "validation": history["validation_loss"]},
        "Loss Pengklasifikasi MLP",
        FIGURE_DIR / "mlp_loss.png",
    )
    pd.DataFrame(
        {
            "epoch": range(1, len(epoch_times) + 1),
            "training_time_seconds": epoch_times,
        }
    ).to_csv(METRIC_DIR / "mlp_training_times.csv", index=False)
    print("Checkpoint MLP dan W1/W2/W3 berhasil disimpan.")


if __name__ == "__main__":
    main()
