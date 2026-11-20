from typing import NamedTuple

import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

from config import BATCH_SIZE, DATA_DIR, SEED


class FashionMNISTLoaders(NamedTuple):
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    train_size: int
    validation_size: int
    test_size: int


def fashion_mnist_loaders(batch_size: int = BATCH_SIZE, data_dir=DATA_DIR) -> FashionMNISTLoaders:
    """Gabungkan split resmi Fashion-MNIST, lalu bagi ulang secara konsisten dengan rasio 80/10/10."""
    transform = transforms.ToTensor()
    official_train = datasets.FashionMNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    official_test = datasets.FashionMNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    dataset = ConcatDataset((official_train, official_test))
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_validation = int(0.1 * n_total)
    n_test = n_total - n_train - n_validation
    train_set, validation_set, test_set = random_split(
        dataset,
        (n_train, n_validation, n_test),
        generator=torch.Generator().manual_seed(SEED),
    )
    train_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return FashionMNISTLoaders(
        train_loader,
        validation_loader,
        test_loader,
        len(train_set),
        len(validation_set),
        len(test_set),
    )
