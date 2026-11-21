import torch
from torch import nn


class VanillaAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(images)).view(-1, 1, 28, 28)


class SparseAE(nn.Module):
    """Lapisan fitur sparse 256 unit yang diikuti bottleneck 64 dimensi."""

    def __init__(self, sparse_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.sparse_encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(784, sparse_dim), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(sparse_dim, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode_sparse_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.sparse_encoder(images)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encode_sparse_features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(images)).view(-1, 1, 28, 28)
