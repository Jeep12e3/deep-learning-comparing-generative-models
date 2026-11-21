import math
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=timesteps.device, dtype=torch.float32)
            * (-math.log(10000.0) / (half - 1))
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        return self.net(embedding)


class TinyUNet(nn.Module):
    """U-Net ringkas yang sesuai untuk baseline Fashion-MNIST berukuran 28x28."""

    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, 64), nn.SiLU())
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
        )
        self.down = nn.Conv2d(32, 64, 4, 2, 1)
        self.mid = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
        )
        self.up = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        skip = self.down1(images)
        hidden = self.down(skip)
        hidden = hidden + self.time_mlp(self.time(timesteps)).view(-1, 64, 1, 1)
        hidden = self.mid(hidden)
        hidden = self.up(hidden)
        return self.out(torch.cat((hidden, skip), dim=1))


class DDPM:
    def __init__(self, model: nn.Module, timesteps: int = 400, device=None):
        self.model = model
        self.timesteps = timesteps
        self.device = device or next(model.parameters()).device
        schedule_steps = torch.arange(timesteps + 1, device=self.device, dtype=torch.float32)
        offset = 0.008
        alpha_bar = torch.cos(
            (schedule_steps / timesteps + offset) / (1 + offset) * math.pi / 2
        ).pow(2)
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-4, 0.999)
        self.alpha_bar = alpha_bar[1:]
        self.beta = beta
        self.alpha = 1 - beta

    def add_noise(self, images: torch.Tensor, timesteps: torch.Tensor, noise=None):
        if noise is None:
            noise = torch.randn_like(images)
        cumulative_alpha = self.alpha_bar[timesteps].view(-1, 1, 1, 1)
        noisy = cumulative_alpha.sqrt() * images + (1 - cumulative_alpha).sqrt() * noise
        return noisy, noise

    @torch.no_grad()
    def sample(self, count: int, capture_steps: Iterable[int] = ()) -> Tuple[torch.Tensor, List]:
        self.model.eval()
        images = torch.randn(count, 1, 28, 28, device=self.device)
        frames = []
        capture_steps = set(capture_steps)
        for index in reversed(range(self.timesteps)):
            step = torch.full((count,), index, device=self.device, dtype=torch.long)
            predicted_noise = self.model(images, step)
            alpha_bar = self.alpha_bar[index]
            previous_alpha_bar = (
                self.alpha_bar[index - 1]
                if index > 0
                else torch.ones((), device=self.device)
            )
            beta = self.beta[index]
            predicted_x0 = (
                (images - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
            ).clamp(-1, 1)
            mean = (
                beta * previous_alpha_bar.sqrt() / (1 - alpha_bar) * predicted_x0
                + (1 - previous_alpha_bar)
                * self.alpha[index].sqrt()
                / (1 - alpha_bar)
                * images
            )
            variance = beta * (1 - previous_alpha_bar) / (1 - alpha_bar)
            images = (
                mean
                if index == 0
                else mean + variance.clamp_min(1e-20).sqrt() * torch.randn_like(images)
            )
            if index in capture_steps:
                frames.append((((images * 0.5 + 0.5).clamp(0, 1).cpu()), index))
        return images.clamp(-1, 1), frames
