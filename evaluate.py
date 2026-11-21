import argparse
import shutil

import pandas as pd
import torch
from torchvision.utils import save_image

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    METRIC_DIR,
    OUTPUT_DIR,
    ensure_output_dirs,
)
from data import fashion_mnist_loaders
from metrics import collect_images, reconstruction_metrics, timed_inference
from models import (
    DCGenerator,
    DDPM,
    SparseAE,
    TinyUNet,
    VAE,
    VanillaAE,
    VanillaGenerator,
)


def load_state(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model.to(DEVICE).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi metrik rekonstruksi, latensi inferensi, dan FID.")
    parser.add_argument("--fid-samples", type=int, default=5000)
    args = parser.parse_args()
    ensure_output_dirs()
    loaders = fashion_mnist_loaders()
    real = collect_images(loaders.test)
    models = {
        "Vanilla AE": load_state(VanillaAE(64), CHECKPOINT_DIR / "vanilla_ae.pt"),
        "Sparse AE": load_state(SparseAE(256, 64), CHECKPOINT_DIR / "sparse_ae.pt"),
        "VAE": load_state(VAE(64), CHECKPOINT_DIR / "vae.pt"),
    }

    rows = []
    with torch.no_grad():
        for name, model in models.items():
            outputs = []
            for start in range(0, len(real), 256):
                batch = real[start : start + 256].to(DEVICE)
                if isinstance(model, VAE):
                    mu, _ = model.encode(batch)
                    outputs.append(model.decode(mu).cpu())
                else:
                    outputs.append(model(batch).cpu())
            mse, psnr, ssim = reconstruction_metrics(real, torch.cat(outputs))
            rows.append({"Model": name, "MSE": mse, "PSNR": psnr, "SSIM": ssim})

    vae = load_state(VAE(64), CHECKPOINT_DIR / "vae.pt")
    generators = {
        "VAE": vae,
        "Vanilla_GAN": load_state(VanillaGenerator(64), CHECKPOINT_DIR / "vanilla_gan_generator.pt"),
        "DCGAN": load_state(DCGenerator(100), CHECKPOINT_DIR / "dcgan_generator.pt"),
    }
    ddpm_checkpoint = torch.load(
        CHECKPOINT_DIR / "ddpm.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    if not isinstance(ddpm_checkpoint, dict) or not {
        "model_state_dict",
        "timesteps",
    }.issubset(ddpm_checkpoint):
        raise RuntimeError(
            "Checkpoint DDPM tidak menyimpan konfigurasi jumlah timestep. "
            "Latih ulang dengan `python train_ddpm.py` sebelum menjalankan evaluasi."
        )
    ddpm_model = TinyUNet().to(DEVICE)
    ddpm_model.load_state_dict(ddpm_checkpoint["model_state_dict"])
    ddpm_model.eval()
    diffusion = DDPM(
        ddpm_model,
        timesteps=int(ddpm_checkpoint["timesteps"]),
        device=DEVICE,
    )

    latency = []
    for name in ("Vanilla AE", "Sparse AE"):
        model = models[name]
        batch = torch.randn(64, 1, 28, 28, device=DEVICE)
        latency.append(
            {
                "Model": name,
                "Latency_ms_batch64": timed_inference(
                    lambda model=model, batch=batch: model(batch), DEVICE
                ),
            }
        )
    latent = torch.randn(64, 64, device=DEVICE)
    latency.append(
        {
            "Model": "VAE",
            "Latency_ms_batch64": timed_inference(
                lambda: generators["VAE"].decode(latent), DEVICE
            ),
        }
    )
    for name, generator, dim in (
        ("Vanilla_GAN", generators["Vanilla_GAN"], 64),
        ("DCGAN", generators["DCGAN"], 100),
    ):
        z = torch.randn(64, dim, device=DEVICE)
        latency.append(
            {"Model": name, "Latency_ms_batch64": timed_inference(lambda: generator(z), DEVICE)}
        )
    latency.append(
        {
            "Model": "DDPM",
            "Latency_ms_batch64": timed_inference(
                lambda: diffusion.sample(64), DEVICE, warmup=1, repeats=3
            ),
        }
    )

    # FID memerlukan sampel dalam ruang gambar; simpan bertahap agar penggunaan memori GPU terkendali.
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise RuntimeError(
            "Evaluasi FID memerlukan torch-fidelity. Pasang dengan `pip install torch-fidelity`."
        ) from exc

    sample_count = min(args.fid_samples, len(real))
    real_dir = OUTPUT_DIR / "fid_real"
    if real_dir.exists():
        shutil.rmtree(real_dir)
    real_dir.mkdir(parents=True)
    for index, image in enumerate(real[:sample_count]):
        save_image(image.repeat(3, 1, 1), real_dir / f"{index:05d}.png")

    fid_rows = []
    fid_samplers = {
        "VAE": lambda count: generators["VAE"].decode(
            torch.randn(count, 64, device=DEVICE)
        ),
        "Vanilla_GAN": lambda count: generators["Vanilla_GAN"](
            torch.randn(count, 64, device=DEVICE)
        ),
        "DCGAN": lambda count: generators["DCGAN"](
            torch.randn(count, 100, device=DEVICE)
        ),
        "DDPM": lambda count: diffusion.sample(count)[0] * 0.5 + 0.5,
    }
    for name, sampler in fid_samplers.items():
        fake_dir = OUTPUT_DIR / f"fid_fake_{name}"
        if fake_dir.exists():
            shutil.rmtree(fake_dir)
        fake_dir.mkdir(parents=True)
        saved = 0
        while saved < sample_count:
            count = min(64, sample_count - saved)
            generated = sampler(count)
            for image in generated.detach().cpu().clamp(0, 1):
                save_image(image.repeat(3, 1, 1), fake_dir / f"{saved:05d}.png")
                saved += 1
        fid = calculate_metrics(
            input1=str(real_dir),
            input2=str(fake_dir),
            cuda=DEVICE.type == "cuda",
            fid=True,
        )["frechet_inception_distance"]
        fid_rows.append({"Model": name, "FID": fid})

    pd.DataFrame(rows).to_csv(METRIC_DIR / "reconstruction_metrics.csv", index=False)
    pd.DataFrame(latency).to_csv(METRIC_DIR / "generation_latency.csv", index=False)
    pd.DataFrame(fid_rows).to_csv(METRIC_DIR / "fid_scores.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(latency).to_string(index=False))
    print(pd.DataFrame(fid_rows).to_string(index=False))


if __name__ == "__main__":
    main()
