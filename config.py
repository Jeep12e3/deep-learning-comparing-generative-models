from pathlib import Path

import torch

SEED = 42
BATCH_SIZE = 128
LATENT_DIM = 64
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs() -> None:
    for path in (OUTPUT_DIR, CHECKPOINT_DIR, FIGURE_DIR, METRIC_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
