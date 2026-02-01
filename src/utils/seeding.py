"""Utility functions for seeding and reproducibility."""

import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, env: Optional[object] = None) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        env: Optional environment to seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    if env is not None and hasattr(env, 'seed'):
        env.seed(seed)
        if hasattr(env, 'action_space'):
            env.action_space.seed(seed)
        if hasattr(env, 'observation_space'):
            env.observation_space.seed(seed)


def get_device() -> str:
    """Get the best available device.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
