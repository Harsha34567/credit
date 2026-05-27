"""Utility for consistent experiment reproducibility."""

import os
import random

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """Set deterministic random seeds for NumPy, Python, and environment-level randomness."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
