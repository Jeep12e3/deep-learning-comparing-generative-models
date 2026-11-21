import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor):
        hidden = self.encoder(images)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images: torch.Tensor):
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar
