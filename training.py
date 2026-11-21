import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from config import BATCH_SIZE, DEVICE, SEED
from metrics import AverageMeter
from models import (
    DCDiscriminator,
    DCGenerator,
    DDPM,
    LatentDecoder,
    MLPClassifier,
    SparseAE,
    TinyUNet,
    VAE,
    VanillaAE,
    VanillaDiscriminator,
    VanillaGenerator,
)
from models.gan import weights_init_normal


def train_mlp_classifier(
    train_loader,
    validation_loader,
    epochs: int = 15,
    learning_rate: float = 1e-3,
) -> Tuple[MLPClassifier, Dict, list, float]:
    model = MLPClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    epoch_times = []
    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        correct = total = 0
        started = time.perf_counter()
        for images, labels in tqdm(train_loader, desc=f"Pelatihan MLP {epoch}/{epochs}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        epoch_times.append(time.perf_counter() - started)
        history["train_loss"].append(meter.avg)
        history["train_accuracy"].append(correct / total)

        model.eval()
        meter = AverageMeter()
        correct = total = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, _ = model(images)
                meter.update(criterion(logits, labels).item(), images.size(0))
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        history["validation_loss"].append(meter.avg)
        history["validation_accuracy"].append(correct / total)
        print(
            f"Epoch {epoch}/{epochs}: akurasi_latih={history['train_accuracy'][-1]:.4f}, "
            f"akurasi_validasi={history['validation_accuracy'][-1]:.4f}"
        )

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in validation_loader:
            logits, _ = model(images.to(DEVICE))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.size(0)
    return model, history, epoch_times, correct / total


def extract_latents(model: MLPClassifier, loader):
    model.eval()
    latents, labels, images = [], [], []
    with torch.no_grad():
        for batch, target in loader:
            _, z = model(batch.to(DEVICE))
            latents.append(z.cpu())
            labels.append(target.cpu())
            images.append(batch.cpu())
    return torch.cat(latents), torch.cat(labels), torch.cat(images)


def train_latent_decoder(
    train_latent: torch.Tensor,
    train_images: torch.Tensor,
    epochs: int = 20,
    batch_size: int = BATCH_SIZE,
):
    model = LatentDecoder().to(DEVICE)
    loader = DataLoader(
        TensorDataset(train_latent, train_images),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    epoch_times = []
    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        started = time.perf_counter()
        for latent, images in loader:
            latent, images = latent.to(DEVICE), images.to(DEVICE)
            optimizer.zero_grad()
            loss = F.mse_loss(model(latent), images)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
        history.append(meter.avg)
        epoch_times.append(time.perf_counter() - started)
        if epoch % 5 == 0:
            print(f"Epoch decoder laten {epoch}: MSE={meter.avg:.6f}")
    return model, history, epoch_times


def train_autoencoder(
    model: nn.Module,
    train_loader,
    validation_loader,
    epochs: int = 20,
    sparse: bool = False,
    l1_weight: float = 1e-3,
):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {"train_loss": [], "validation_loss": []}
    epoch_times = []
    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        started = time.perf_counter()
        for images, _ in tqdm(train_loader, desc=f"Pelatihan {type(model).__name__} {epoch}/{epochs}", leave=False):
            images = images.to(DEVICE)
            optimizer.zero_grad()
            if isinstance(model, VAE):
                reconstruction, mu, logvar = model(images)
                reconstruction_loss = F.binary_cross_entropy(
                    reconstruction, images, reduction="sum"
                ) / images.size(0)
                kl = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp()
                ) / images.size(0)
                loss = reconstruction_loss + kl
            else:
                reconstruction = model(images)
                loss = F.mse_loss(reconstruction, images)
                if sparse:
                    loss = loss + l1_weight * model.encode_sparse_features(images).abs().sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), images.size(0))
        epoch_times.append(time.perf_counter() - started)
        history["train_loss"].append(meter.avg)

        model.eval()
        validation_meter = AverageMeter()
        with torch.no_grad():
            for images, _ in validation_loader:
                images = images.to(DEVICE)
                if isinstance(model, VAE):
                    reconstruction, mu, logvar = model(images)
                    loss = F.binary_cross_entropy(
                        reconstruction, images, reduction="sum"
                    ) / images.size(0)
                    loss -= 0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    ) / images.size(0)
                else:
                    loss = F.mse_loss(model(images), images)
                    if sparse:
                        loss += l1_weight * model.encode_sparse_features(images).abs().sum(dim=1).mean()
                validation_meter.update(loss.item(), images.size(0))
        history["validation_loss"].append(validation_meter.avg)
    return model, history, epoch_times


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader,
    latent_dim: int,
    epochs: int = 30,
):
    generator, discriminator = generator.to(DEVICE), discriminator.to(DEVICE)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = {"generator_loss": [], "discriminator_loss": []}
    epoch_times = []
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        generator_meter, discriminator_meter = AverageMeter(), AverageMeter()
        started = time.perf_counter()
        for real_images, _ in tqdm(train_loader, desc=f"Pelatihan GAN {epoch}/{epochs}", leave=False):
            real_images = real_images.to(DEVICE)
            batch_size = real_images.size(0)
            discriminator_optimizer.zero_grad()
            real_loss = criterion(
                discriminator(real_images), torch.ones(batch_size, 1, device=DEVICE)
            )
            latent = torch.randn(batch_size, latent_dim, device=DEVICE)
            fake_images = generator(latent).detach()
            fake_loss = criterion(
                discriminator(fake_images), torch.zeros(batch_size, 1, device=DEVICE)
            )
            discriminator_loss = real_loss + fake_loss
            discriminator_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            latent = torch.randn(batch_size, latent_dim, device=DEVICE)
            generated = generator(latent)
            generator_loss = criterion(
                discriminator(generated), torch.ones(batch_size, 1, device=DEVICE)
            )
            generator_loss.backward()
            generator_optimizer.step()
            discriminator_meter.update(discriminator_loss.item(), batch_size)
            generator_meter.update(generator_loss.item(), batch_size)
        epoch_times.append(time.perf_counter() - started)
        history["generator_loss"].append(generator_meter.avg)
        history["discriminator_loss"].append(discriminator_meter.avg)
    return generator, discriminator, history, epoch_times


def train_ddpm(train_loader, epochs: int = 30, timesteps: int = 400):
    model = TinyUNet().to(DEVICE)
    diffusion = DDPM(model, timesteps=timesteps, device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    history = []
    epoch_times = []
    for epoch in range(1, epochs + 1):
        model.train()
        meter = AverageMeter()
        started = time.perf_counter()
        for images, _ in tqdm(train_loader, desc=f"Pelatihan DDPM {epoch}/{epochs}", leave=False):
            images = images.to(DEVICE) * 2 - 1
            batch_size = images.size(0)
            steps = torch.randint(0, timesteps, (batch_size,), device=DEVICE)
            noisy_images, noise = diffusion.add_noise(images, steps)
            optimizer.zero_grad()
            loss = F.mse_loss(model(noisy_images, steps), noise)
            loss.backward()
            optimizer.step()
            meter.update(loss.item(), batch_size)
        history.append(meter.avg)
        epoch_times.append(time.perf_counter() - started)
        print(f"Epoch DDPM {epoch}/{epochs}: MSE prediksi noise={meter.avg:.6f}")
    return model, diffusion, history, epoch_times
