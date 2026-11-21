import argparse

import pandas as pd
import torch
import torch.nn.functional as F

from config import CHECKPOINT_DIR, DEVICE, FIGURE_DIR, METRIC_DIR, ensure_output_dirs
from data import fashion_mnist_loaders
from metrics import reconstruction_metrics, save_image_grid
from models import LatentDecoder, MLPClassifier, VanillaAE
from training import extract_latents, train_latent_decoder


def load_model(model, checkpoint):
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jalankan eksperimen proyeksi transpose MLP, decoder laten beku, dan sampling prior."
    )
    parser.add_argument("--decoder-epochs", type=int, default=20)
    args = parser.parse_args()
    ensure_output_dirs()
    loaders = fashion_mnist_loaders()

    classifier = load_model(MLPClassifier(), CHECKPOINT_DIR / "mlp_classifier.pt")
    train_z, _, train_images = extract_latents(classifier, loaders.train)
    test_z, _, test_images = extract_latents(classifier, loaders.test)

    # Tahap 4: latih decoder terpisah hanya dengan contoh latih dan fitur yang dibekukan.
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    decoder, history, _ = train_latent_decoder(
        train_z, train_images, epochs=args.decoder_epochs
    )
    torch.save(decoder.state_dict(), CHECKPOINT_DIR / "mlp_latent_decoder.pt")

    # Tahap 2: transpose bobot pengklasifikasi terlatih dari vektor kelas.
    w1 = classifier.fc1.weight.detach().cpu()
    w2 = classifier.fc2.weight.detach().cpu()
    w3 = classifier.fc3.weight.detach().cpu()
    class_templates = []
    for class_id in range(10):
        one_hot = F.one_hot(torch.tensor(class_id), num_classes=10).float()
        z_hat = F.relu(w3.T @ one_hot)
        h_hat = F.relu(w2.T @ z_hat)
        class_templates.append(torch.sigmoid(w1.T @ h_hat).view(1, 28, 28))
    save_image_grid(
        torch.cat(class_templates),
        "Tahap 2: proyeksi transpose vektor kelas",
        FIGURE_DIR / "mlp_class_transpose.png",
        nrow=5,
    )

    # Tahap 3: proyeksi transpose fitur laten dari gambar yang ditahan untuk pengujian.
    latent_projection = []
    with torch.no_grad():
        for latent in test_z[:16]:
            hidden = F.relu(w2.T @ latent)
            latent_projection.append(torch.sigmoid(w1.T @ hidden).view(1, 28, 28))
    latent_projection = torch.stack(latent_projection)

    # Tahap 4: bandingkan rekonstruksi data uji dengan Vanilla AE berdimensi sama.
    ae = load_model(VanillaAE(64), CHECKPOINT_DIR / "vanilla_ae.pt")
    decoder.eval()
    with torch.no_grad():
        decoder_recon = decoder(test_z.to(DEVICE)).cpu()
        ae_recon = ae(test_images.to(DEVICE)).cpu()
    decoder_metrics = reconstruction_metrics(test_images, decoder_recon)
    ae_metrics = reconstruction_metrics(test_images, ae_recon)
    pd.DataFrame(
        [
            {"model": "Frozen MLP latent decoder", "MSE": decoder_metrics[0], "PSNR": decoder_metrics[1], "SSIM": decoder_metrics[2]},
            {"model": "Vanilla AE", "MSE": ae_metrics[0], "PSNR": ae_metrics[1], "SSIM": ae_metrics[2]},
        ]
    ).to_csv(METRIC_DIR / "mlp_vs_vanilla_ae_metrics.csv", index=False)

    # Tahap 5: prior Gaussian empiris yang diestimasi hanya dari laten data uji, sesuai ketentuan.
    z_mean, z_std = test_z.mean(0), test_z.std(0)
    with torch.no_grad():
        random_z = torch.randn(16, 64, device=DEVICE) * z_std.to(DEVICE) + z_mean.to(DEVICE)
        prior_samples = decoder(random_z).cpu()
    save_image_grid(
        prior_samples,
        "Tahap 5: sampel prior Gaussian laten MLP",
        FIGURE_DIR / "mlp_prior_samples.png",
        nrow=4,
        figsize=(8, 8),
    )

    # Gambar bukti gabungan: gambar asli, proyeksi laten langsung, decoder vs AE, dan sampel prior.
    originals = test_images[:8]
    comparison = torch.cat((decoder_recon[:8], ae_recon[:8]))
    evidence = torch.cat((originals, latent_projection[:8], comparison, prior_samples))
    save_image_grid(
        evidence,
        "Bukti keterbatasan MLP: gambar asli, proyeksi W^T, decoder beku / AE, sampel prior",
        FIGURE_DIR / "figure3_mlp_limits.png",
        nrow=8,
        figsize=(14, 3),
    )
    pd.DataFrame(
        {"epoch": range(1, len(history) + 1), "training_mse": history}
    ).to_csv(METRIC_DIR / "mlp_decoder_training.csv", index=False)
    print("Eksperimen keterbatasan MLP selesai.")


if __name__ == "__main__":
    main()
