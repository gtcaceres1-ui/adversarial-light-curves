"""Deterministic seeding for numpy / random / TensorFlow."""
from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Set seeds across all relevant libraries for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
