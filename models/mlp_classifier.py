import torch
import torch.nn.functional as F
from torch import nn


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, images: torch.Tensor):
        hidden = F.relu(self.fc1(images.flatten(1)))
        latent = F.relu(self.fc2(hidden))
        return self.fc3(latent), latent


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent).view(-1, 1, 28, 28)
