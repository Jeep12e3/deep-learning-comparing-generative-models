import argparse
import gc
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    FIGURE_DIR,
    METRIC_DIR,
    SEED,
    ensure_output_dirs,
    set_seed,
)
from data import fashion_mnist_loaders
from metrics import reconstruction_metrics, timed_inference
from models import (
    DCGenerator,
    DDPM,
    MLPClassifier,
    SparseAE,
    TinyUNet,
    VAE,
    VanillaAE,
    VanillaGenerator,
)
from training import train_autoencoder


CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_state(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    model.load_state_dict(
        torch.load(path, map_location=DEVICE, weights_only=True)
    )
    return model.to(DEVICE).eval()


def load_generators():
    vae = load_state(VAE(64), CHECKPOINT_DIR / "vae.pt")
    vanilla_gan = load_state(
        VanillaGenerator(64), CHECKPOINT_DIR / "vanilla_gan_generator.pt"
    )
    dcgan = load_state(DCGenerator(100), CHECKPOINT_DIR / "dcgan_generator.pt")
    checkpoint = torch.load(
        CHECKPOINT_DIR / "ddpm.pt",
        map_location=DEVICE,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not {
        "model_state_dict",
        "timesteps",
    }.issubset(checkpoint):
        raise RuntimeError("Checkpoint DDPM tidak memuat state model dan timesteps.")
    unet = TinyUNet().to(DEVICE)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()
    diffusion = DDPM(unet, timesteps=int(checkpoint["timesteps"]), device=DEVICE)
    return {
        "VAE": (vae, 64),
        "Vanilla GAN": (vanilla_gan, 64),
        "DCGAN": (dcgan, 100),
        "DDPM": (diffusion, None),
    }


def measure_batch_one_latency(generators, repeats: int) -> None:
    rows = []
    for name in ("VAE", "Vanilla GAN", "DCGAN"):
        model, latent_dim = generators[name]
        latent = torch.randn(1, latent_dim, device=DEVICE)
        if name == "VAE":
            run = lambda: model.decode(latent)
            input_image = torch.rand(1, 1, 28, 28, device=DEVICE)
            encode_decode = lambda: model.decode(model.encode(input_image)[0])
        else:
            run = lambda: model(latent)
            encode_decode = None
        measurements = [
            timed_inference(run, DEVICE, warmup=20, repeats=repeats)
            for _ in range(5)
        ]
        row = {
                "Model": name,
                "Latency_ms_batch1_median": float(np.median(measurements)),
                "Latency_ms_batch1_p10": float(np.percentile(measurements, 10)),
                "Latency_ms_batch1_p90": float(np.percentile(measurements, 90)),
                "Warmup": 20,
                "Repeats_per_round": repeats,
                "Rounds": 5,
                "Device": str(DEVICE),
                "CPU_threads": torch.get_num_threads(),
            }
        if encode_decode is not None:
            end_to_end = [
                timed_inference(encode_decode, DEVICE, warmup=20, repeats=repeats)
                for _ in range(5)
            ]
            row.update(
                {
                    "Latency_ms_batch1_encode_decode_median": float(
                        np.median(end_to_end)
                    ),
                    "Latency_ms_batch1_encode_decode_p10": float(
                        np.percentile(end_to_end, 10)
                    ),
                    "Latency_ms_batch1_encode_decode_p90": float(
                        np.percentile(end_to_end, 90)
                    ),
                }
            )
        rows.append(row)
        print(f"Latensi batch-1 selesai: {name}")
    pd.DataFrame(rows).to_csv(METRIC_DIR / "batch1_generation_latency.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def measure_class_coverage(
    generators,
    samples_per_model: int,
    batch_size: int,
    confidence_threshold: float,
) -> None:
    classifier = load_state(
        MLPClassifier(), CHECKPOINT_DIR / "mlp_classifier.pt"
    )
    rows = []
    for model_index, (name, (model, latent_dim)) in enumerate(generators.items()):
        set_seed(SEED + model_index)
        class_counts = np.zeros(10, dtype=np.int64)
        confident_counts = np.zeros(10, dtype=np.int64)
        generated_count = 0
        confident_count = 0
        cache_name = name.replace(" ", "_")
        cached_files = sorted(
            (CHECKPOINT_DIR.parent / f"fid_fake_{cache_name}").glob("*.png")
        )
        use_cache = len(cached_files) >= samples_per_model
        while generated_count < samples_per_model:
            count = min(batch_size, samples_per_model - generated_count)
            if use_cache:
                cached_batch = []
                for path in cached_files[generated_count : generated_count + count]:
                    with Image.open(path) as image:
                        pixels = (
                            np.asarray(image.convert("L"), dtype=np.float32).copy()
                            / 255
                        )
                    cached_batch.append(torch.from_numpy(pixels).unsqueeze(0))
                images = torch.stack(cached_batch).to(DEVICE)
            else:
                with torch.no_grad():
                    if name == "VAE":
                        images = model.decode(
                            torch.randn(count, latent_dim, device=DEVICE)
                        )
                    elif name == "DDPM":
                        images = (model.sample(count)[0] + 1) / 2
                    else:
                        images = model(torch.randn(count, latent_dim, device=DEVICE))
            with torch.no_grad():
                logits, _ = classifier(images)
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = probabilities.max(dim=1)
                predictions = prediction.cpu().numpy()
                accepted = confidence.cpu().numpy() >= confidence_threshold
            class_counts += np.bincount(predictions, minlength=10)
            confident_counts += np.bincount(
                predictions[accepted], minlength=10
            )
            confident_count += int(accepted.sum())
            generated_count += count
            print(
                f"\rKlasifikasi sampel {name}: "
                f"{generated_count}/{samples_per_model}",
                end="",
                flush=True,
            )
        print()
        rows.extend(
            {
                "Model": name,
                "Class_id": class_id,
                "Class_name": CLASS_NAMES[class_id],
                "Count": int(class_counts[class_id]),
                "Fraction": float(class_counts[class_id] / samples_per_model),
                "Confident_count": int(confident_counts[class_id]),
                "Confident_fraction_all": float(
                    confident_counts[class_id] / samples_per_model
                ),
                "Confident_fraction_accepted": float(
                    confident_counts[class_id] / confident_count
                    if confident_count
                    else 0
                ),
                "Accepted_fraction": float(confident_count / samples_per_model),
                "Confidence_threshold": confidence_threshold,
                "Sample_source": "FID cache" if use_cache else "new generation",
                "Samples": samples_per_model,
            }
            for class_id in range(10)
        )
        del model
        gc.collect()

    frame = pd.DataFrame(rows)
    frame.to_csv(METRIC_DIR / "generated_class_coverage.csv", index=False)
    pivot = frame.pivot(
        index="Class_name", columns="Model", values="Confident_fraction_accepted"
    )
    pivot = pivot.reindex(CLASS_NAMES)
    axis = pivot.plot(kind="bar", figsize=(12, 5), width=0.82)
    axis.set_ylabel("Proporsi per kelas di antara prediksi yang lolos filter")
    axis.set_xlabel("Kelas Fashion-MNIST")
    axis.set_title(
        f"Cakupan kelas MLP (confidence ≥ {confidence_threshold:.2f}; "
        f"{samples_per_model} sampel per generator)"
    )
    axis.legend(title="Generator", ncols=4)
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "generated_class_coverage.png", dpi=220)
    plt.close()
    summary = frame.drop_duplicates("Model").set_index("Model")
    print(
        summary[
            ["Accepted_fraction", "Confident_count", "Sample_source"]
        ].to_string()
    )
    print(
        frame.pivot(index="Model", columns="Class_name", values="Count").to_string()
    )


def evaluate_sparse(model: SparseAE, loader, label: str, training_seconds: float):
    images_all = []
    reconstructions = []
    hidden_zero = 0
    hidden_count = 0
    hidden_l1 = []
    active_count_by_neuron = torch.zeros(model.sparse_encoder[1].out_features)
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(DEVICE)
            features = model.encode_sparse_features(images)
            active_count_by_neuron += (features > 1e-6).sum(dim=0).cpu()
            sample_count += len(images)
            reconstructions.append(model(images).cpu())
            images_all.append(images.cpu())
            hidden_zero += (features.abs() < 1e-6).sum().item()
            hidden_count += features.numel()
            hidden_l1.extend(features.abs().sum(dim=1).cpu().tolist())
    neuron_active_fraction = active_count_by_neuron.numpy() / sample_count
    active_non_dead = neuron_active_fraction[neuron_active_fraction > 0]
    dead_neuron_fraction = float((neuron_active_fraction == 0).mean())
    mse, psnr, ssim = reconstruction_metrics(
        torch.cat(images_all), torch.cat(reconstructions)
    )
    return (
        {
            "Model": label,
            "Lambda": label.split("=")[-1],
            "MSE": mse,
            "PSNR": psnr,
            "SSIM": ssim,
            "Hidden_zero_fraction": hidden_zero / hidden_count,
            "Dead_neuron_fraction": dead_neuron_fraction,
            "Mean_neuron_active_fraction": float(neuron_active_fraction.mean()),
            "Median_active_fraction_non_dead": float(np.median(active_non_dead)),
            "P10_active_fraction_non_dead": float(
                np.percentile(active_non_dead, 10)
            ),
            "P90_active_fraction_non_dead": float(
                np.percentile(active_non_dead, 90)
            ),
            "Mean_hidden_L1_per_image": float(np.mean(hidden_l1)),
            "Mean_abs_W1": float(model.sparse_encoder[1].weight.abs().mean().item()),
            "Training_seconds": training_seconds,
        },
        neuron_active_fraction,
    )


def run_sparse_lambda_sweep(loaders, epochs: int) -> None:
    validation_loader = DataLoader(
        loaders.validation.dataset,
        batch_size=loaders.validation.batch_size,
        shuffle=False,
        num_workers=0,
    )
    models = {}
    rows = []
    neuron_rows = []
    active_fractions_by_lambda = {}
    for weight in (0.0, 1e-5, 1e-4, 1e-3):
        label = f"Sparse AE lambda={weight:g}"
        if weight == 0.0:
            model = load_state(
                SparseAE(256, 64), CHECKPOINT_DIR / "sparse_ae_no_l1.pt"
            )
            training_seconds = 1198.5833948000218
        elif weight == 1e-3:
            model = load_state(
                SparseAE(256, 64), CHECKPOINT_DIR / "sparse_ae.pt"
            )
            training_seconds = 0.0
        else:
            checkpoint_path = CHECKPOINT_DIR / f"sparse_ae_lambda_{weight:g}.pt"
            if checkpoint_path.exists():
                model = load_state(SparseAE(256, 64), checkpoint_path)
                training_seconds = 0.0
            else:
                set_seed(SEED)
                training_loader = DataLoader(
                    loaders.train.dataset,
                    batch_size=loaders.train.batch_size,
                    shuffle=True,
                    generator=torch.Generator().manual_seed(SEED),
                    num_workers=0,
                )
                model = SparseAE(256, 64).to(DEVICE)
                started = time.perf_counter()
                model, _, _ = train_autoencoder(
                    model,
                    training_loader,
                    validation_loader,
                    epochs=epochs,
                    sparse=True,
                    l1_weight=weight,
                )
                training_seconds = time.perf_counter() - started
                torch.save(model.state_dict(), checkpoint_path)
                model.eval()
        models[label] = model
        summary, active_fractions = evaluate_sparse(
            model, loaders.test, label, training_seconds
        )
        rows.append(summary)
        active_fractions_by_lambda[weight] = active_fractions
        neuron_rows.extend(
            {
                "Lambda": weight,
                "Neuron_id": neuron_id,
                "Active_sample_fraction": float(active_fraction),
            }
            for neuron_id, active_fraction in enumerate(active_fractions)
        )
        print(f"Sparse AE lambda sweep selesai: lambda={weight:g}")

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(METRIC_DIR / "sparse_ae_lambda_sweep.csv", index=False)
    pd.DataFrame(neuron_rows).to_csv(
        METRIC_DIR / "sparse_ae_neuron_activity.csv", index=False
    )
    fig, axes = plt.subplots(4, 16, figsize=(18, 5))
    lambda_values = (0.0, 1e-5, 1e-4, 1e-3)
    selected_neuron_ids = {}
    for weight in lambda_values:
        active_fractions = active_fractions_by_lambda[weight]
        active_neuron_ids = np.flatnonzero(active_fractions > 0)
        sorted_active_ids = active_neuron_ids[
            np.argsort(active_fractions[active_neuron_ids])[::-1]
        ]
        selected_neuron_ids[weight] = sorted_active_ids[:16].copy()
    all_weights = [
        model.sparse_encoder[1].weight.detach()
        .cpu()[selected_neuron_ids[weight]]
        .view(-1, 28, 28)
        for weight, model in zip(lambda_values, models.values())
    ]
    shared_color_limit = max(float(weights.abs().max()) for weights in all_weights)
    for row_index, (label, weights) in enumerate(zip(models, all_weights)):
        for column_index, axis in enumerate(axes[row_index]):
            weight = lambda_values[row_index]
            if column_index >= len(selected_neuron_ids[weight]):
                axis.axis("off")
                continue
            axis.imshow(
                weights[column_index],
                cmap="coolwarm",
                vmin=-shared_color_limit,
                vmax=shared_color_limit,
            )
            axis.axis("off")
            neuron_id = selected_neuron_ids[weight][column_index]
            active_fraction = active_fractions_by_lambda[weight][neuron_id]
            axis.set_title(
                f"{neuron_id}: {active_fraction:.0%}",
                fontsize=5,
                pad=1,
            )
        fig.text(
            0.01,
            1 - (row_index + 0.5) / len(models),
            label.replace("Sparse AE lambda=", "λ="),
            ha="left",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        "Hingga 16 neuron aktif teratas per λ: ID dan frekuensi (skala bersama)"
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.88, bottom=0.02, wspace=0.04, hspace=0.08)
    fig.savefig(FIGURE_DIR / "sparse_ae_lambda_weights.png", dpi=220)
    plt.close(fig)
    print(summary_frame.to_string(index=False))


def collect_class_tensors(loader, target_class: int | None = None):
    images, labels = [], []
    for batch_images, batch_labels in loader:
        if target_class is None:
            selected = torch.ones_like(batch_labels, dtype=torch.bool)
        else:
            selected = batch_labels == target_class
        if selected.any():
            images.append(batch_images[selected])
            labels.append(batch_labels[selected])
    return torch.cat(images), torch.cat(labels)


def binary_auroc(scores: np.ndarray, anomaly_labels: np.ndarray) -> float:
    positive_count = int(anomaly_labels.sum())
    negative_count = len(anomaly_labels) - positive_count
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[anomaly_labels == 1].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def run_anomaly_auroc(loaders, normal_classes: tuple[int, ...], epochs: int) -> None:
    test_images, test_labels = collect_class_tensors(loaders.test)
    test_means = test_images.flatten(1).mean(dim=1).numpy()
    result_rows = []
    summary_rows = []
    model_specs = (
        (
            "Vanilla AE",
            lambda: VanillaAE(64),
            False,
            0.0,
            "anomaly_vanilla_ae_normal_{normal_class}.pt",
        ),
        (
            "Sparse AE (lambda=1e-3)",
            lambda: SparseAE(256, 64),
            True,
            1e-3,
            "anomaly_sparse_ae_normal_{normal_class}.pt",
        ),
        (
            "Sparse AE (lambda=1e-5)",
            lambda: SparseAE(256, 64),
            True,
            1e-5,
            "anomaly_sparse_ae_lambda_1e-05_normal_{normal_class}.pt",
        ),
        (
            "VAE",
            lambda: VAE(64),
            False,
            0.0,
            "anomaly_vae_normal_{normal_class}.pt",
        ),
    )

    for normal_class in normal_classes:
        train_images, _ = collect_class_tensors(loaders.train, normal_class)
        validation_images, _ = collect_class_tensors(
            loaders.validation, normal_class
        )
        train_dataset = TensorDataset(
            train_images, torch.zeros(len(train_images), dtype=torch.long)
        )
        validation_dataset = TensorDataset(
            validation_images, torch.zeros(len(validation_images), dtype=torch.long)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=128,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED + normal_class),
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=128, shuffle=False, num_workers=0
        )
        normal_reference_intensity = float(train_images.flatten(1).mean(dim=1).mean())
        normal_test_mask = test_labels == normal_class
        baseline_scores = np.abs(test_means - normal_reference_intensity)

        model_scores = {
            "Mean intensity baseline": baseline_scores,
        }
        for model_index, (
            model_name,
            create_model,
            sparse,
            l1_weight,
            checkpoint_template,
        ) in enumerate(model_specs):
            checkpoint_path = CHECKPOINT_DIR / checkpoint_template.format(
                normal_class=normal_class
            )
            if checkpoint_path.exists():
                model = load_state(create_model(), checkpoint_path)
                training_seconds = 0.0
            else:
                set_seed(SEED + normal_class * len(model_specs) + model_index)
                model = create_model().to(DEVICE)
                started = time.perf_counter()
                model, _, _ = train_autoencoder(
                    model,
                    train_loader,
                    validation_loader,
                    epochs=epochs,
                    sparse=sparse,
                    l1_weight=l1_weight,
                )
                training_seconds = time.perf_counter() - started
                torch.save(model.state_dict(), checkpoint_path)
                model.eval()

            model_errors = []
            with torch.no_grad():
                for start in range(0, len(test_images), 256):
                    batch = test_images[start : start + 256].to(DEVICE)
                    if isinstance(model, VAE):
                        mu, _ = model.encode(batch)
                        reconstruction = model.decode(mu)
                    else:
                        reconstruction = model(batch)
                    error = (reconstruction - batch).square().flatten(1).mean(dim=1)
                    model_errors.extend(error.cpu().tolist())
            model_scores[model_name] = np.asarray(model_errors)
            del model
            gc.collect()
            print(
                f"Detektor {model_name}, kelas normal {CLASS_NAMES[normal_class]} "
                f"selesai dalam {training_seconds:.1f} s."
            )

        normal_count = int(normal_test_mask.sum())
        for model_name, scores in model_scores.items():
            anomaly_mask = ~normal_test_mask.numpy()
            normal_scores = scores[normal_test_mask.numpy()]
            all_anomaly_scores = scores[anomaly_mask]
            overall_labels = anomaly_mask.astype(np.uint8)
            overall_auc = binary_auroc(scores, overall_labels)
            class_aucs = []
            result_rows.append(
                {
                    "Model": model_name,
                    "Normal_class_id": normal_class,
                    "Normal_class": CLASS_NAMES[normal_class],
                    "Anomaly_class": "All other classes",
                    "AUROC": overall_auc,
                    "Normal_test_images": normal_count,
                    "Anomaly_test_images": int(anomaly_mask.sum()),
                    "Training_epochs": epochs if model_name != "Mean intensity baseline" else 0,
                    "Normal_score_median": float(np.median(normal_scores)),
                    "Normal_score_p10": float(np.percentile(normal_scores, 10)),
                    "Normal_score_p90": float(np.percentile(normal_scores, 90)),
                    "Anomaly_score_median": float(np.median(all_anomaly_scores)),
                    "Anomaly_score_p10": float(np.percentile(all_anomaly_scores, 10)),
                    "Anomaly_score_p90": float(np.percentile(all_anomaly_scores, 90)),
                }
            )
            for anomaly_class, anomaly_name in enumerate(CLASS_NAMES):
                if anomaly_class == normal_class:
                    continue
                pair_mask = np.isin(
                    test_labels.numpy(), (normal_class, anomaly_class)
                )
                labels = (test_labels.numpy()[pair_mask] == anomaly_class).astype(
                    np.uint8
                )
                auc = binary_auroc(scores[pair_mask], labels)
                class_aucs.append((anomaly_name, auc))
                normal_pair_scores = scores[test_labels.numpy() == normal_class]
                anomaly_pair_scores = scores[test_labels.numpy() == anomaly_class]
                result_rows.append(
                    {
                        "Model": model_name,
                        "Normal_class_id": normal_class,
                        "Normal_class": CLASS_NAMES[normal_class],
                        "Anomaly_class": anomaly_name,
                        "AUROC": auc,
                        "Normal_test_images": normal_count,
                        "Anomaly_test_images": int((test_labels == anomaly_class).sum()),
                        "Training_epochs": epochs if model_name != "Mean intensity baseline" else 0,
                        "Normal_score_median": float(np.median(normal_pair_scores)),
                        "Normal_score_p10": float(
                            np.percentile(normal_pair_scores, 10)
                        ),
                        "Normal_score_p90": float(
                            np.percentile(normal_pair_scores, 90)
                        ),
                        "Anomaly_score_median": float(
                            np.median(anomaly_pair_scores)
                        ),
                        "Anomaly_score_p10": float(
                            np.percentile(anomaly_pair_scores, 10)
                        ),
                        "Anomaly_score_p90": float(
                            np.percentile(anomaly_pair_scores, 90)
                        ),
                    }
                )
            hardest_class, hardest_auc = min(class_aucs, key=lambda row: row[1])
            summary_rows.append(
                {
                    "Model": model_name,
                    "Normal_class_id": normal_class,
                    "Normal_class": CLASS_NAMES[normal_class],
                    "Overall_AUROC": overall_auc,
                    "Macro_per_class_AUROC": float(
                        np.mean([auc for _, auc in class_aucs])
                    ),
                    "Hardest_anomaly_class": hardest_class,
                    "Hardest_anomaly_AUROC": hardest_auc,
                }
            )

    results = pd.DataFrame(result_rows)
    results.to_csv(METRIC_DIR / "anomaly_auroc.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(METRIC_DIR / "anomaly_auroc_summary.csv", index=False)
    pivot = summary.pivot(
        index="Normal_class", columns="Model", values="Macro_per_class_AUROC"
    )
    axis = pivot.plot(kind="bar", figsize=(10, 5))
    axis.set_ylabel("Makro-AUROC per kelas anomali")
    axis.set_xlabel("Kelas normal")
    axis.set_title("Deteksi anomali: skor rekonstruksi vs baseline intensitas")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Skor")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "anomaly_auroc_summary.png", dpi=220)
    plt.close()
    print(summary.to_string(index=False))


def measure_latent_smoothness(loaders, pairs: int) -> None:
    models = {
        "Vanilla AE": load_state(VanillaAE(64), CHECKPOINT_DIR / "vanilla_ae.pt"),
        "Sparse AE": load_state(
            SparseAE(256, 64), CHECKPOINT_DIR / "sparse_ae.pt"
        ),
        "VAE": load_state(VAE(64), CHECKPOINT_DIR / "vae.pt"),
    }
    images, _ = collect_class_tensors(loaders.test)
    generator = torch.Generator().manual_seed(SEED)
    pair_indices = torch.randint(
        len(images), (pairs, 2), generator=generator
    )
    endpoints = images[pair_indices].to(DEVICE)
    alpha = torch.linspace(0, 1, 6, device=DEVICE).view(1, 6, 1)
    rows = []

    with torch.no_grad():
        for name, model in models.items():
            first, second = endpoints[:, 0], endpoints[:, 1]
            if isinstance(model, VAE):
                first_latent = model.encode(first)[0]
                second_latent = model.encode(second)[0]
            else:
                first_latent = model.encode(first)
                second_latent = model.encode(second)
            interpolated = (
                first_latent[:, None, :] * (1 - alpha)
                + second_latent[:, None, :] * alpha
            )
            flat_latents = interpolated.flatten(0, 1)
            if isinstance(model, VAE):
                decoded = model.decode(flat_latents)
            else:
                decoded = model.decoder(flat_latents).view(-1, 1, 28, 28)
            frames = decoded.view(pairs, 6, 1, 28, 28)
            adjacent_mse = (frames[:, 1:] - frames[:, :-1]).square().flatten(2).mean(2)
            per_pair = adjacent_mse.mean(dim=1).cpu().numpy()
            segment_lengths = torch.linalg.vector_norm(
                (frames[:, 1:] - frames[:, :-1]).flatten(2), dim=2
            )
            endpoint_lengths = torch.linalg.vector_norm(
                (frames[:, -1] - frames[:, 0]).flatten(1), dim=1
            ).clamp_min(1e-8)
            normalized_path_length = (
                segment_lengths.sum(dim=1) / endpoint_lengths
            ).cpu().numpy()
            rows.append(
                {
                    "Model": name,
                    "Random_pairs": pairs,
                    "Frames_per_pair": 6,
                    "Mean_adjacent_frame_MSE": float(per_pair.mean()),
                    "P10": float(np.percentile(per_pair, 10)),
                    "P90": float(np.percentile(per_pair, 90)),
                    "Mean_path_to_endpoint_ratio": float(
                        normalized_path_length.mean()
                    ),
                    "Path_ratio_P10": float(
                        np.percentile(normalized_path_length, 10)
                    ),
                    "Path_ratio_P90": float(
                        np.percentile(normalized_path_length, 90)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(METRIC_DIR / "latent_smoothness.csv", index=False)
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eksperimen tambahan untuk melengkapi tabel dan analisis laporan."
    )
    parser.add_argument("--sparse-epochs", type=int, default=20)
    parser.add_argument("--anomaly-epochs", type=int, default=20)
    parser.add_argument("--coverage-samples", type=int, default=1000)
    parser.add_argument("--coverage-batch-size", type=int, default=32)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--normal-classes", default="0,1,8")
    parser.add_argument("--latent-pairs", type=int, default=100)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--skip-anomaly", action="store_true")
    parser.add_argument("--skip-latent-smoothness", action="store_true")
    args = parser.parse_args()
    try:
        normal_classes = tuple(
            int(value.strip()) for value in args.normal_classes.split(",")
        )
    except ValueError:
        parser.error("--normal-classes harus berupa daftar ID kelas, misalnya 0,1,8.")
    if (
        not normal_classes
        or len(set(normal_classes)) != len(normal_classes)
        or any(not 0 <= class_id < len(CLASS_NAMES) for class_id in normal_classes)
    ):
        parser.error("--normal-classes harus berisi ID kelas unik dari 0 sampai 9.")
    if not 0 < args.confidence_threshold <= 1:
        parser.error("--confidence-threshold harus lebih besar dari 0 dan paling tinggi 1.")
    if args.latent_pairs < 1:
        parser.error("--latent-pairs harus lebih besar dari 0.")

    ensure_output_dirs()
    torch.set_num_threads(1)
    set_seed(SEED)
    loaders = fashion_mnist_loaders()
    if not args.skip_latency or not args.skip_coverage:
        generators = load_generators()
        if not args.skip_latency:
            measure_batch_one_latency(generators, args.latency_repeats)
        if not args.skip_coverage:
            measure_class_coverage(
                generators,
                args.coverage_samples,
                args.coverage_batch_size,
                args.confidence_threshold,
            )
    if not args.skip_sparse:
        run_sparse_lambda_sweep(loaders, args.sparse_epochs)
    if not args.skip_anomaly:
        run_anomaly_auroc(loaders, normal_classes, args.anomaly_epochs)
    if not args.skip_latent_smoothness:
        measure_latent_smoothness(loaders, args.latent_pairs)


if __name__ == "__main__":
    main()
