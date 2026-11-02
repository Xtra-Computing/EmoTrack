"""Optional reproducible training setup."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
    """Configure all RNG sources for reproducible training."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if torch.cuda.is_initialized():
        logger.warning(
            "CUDA already initialized when set_determinism(%d) was called; "
            "CUBLAS_WORKSPACE_CONFIG may not take effect. Enable deterministic "
            "training before any module touches CUDA, or set the env var in "
            "the shell.",
            seed,
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader worker seeder from the PyTorch reproducibility guide."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    """Generator for ``DataLoader(shuffle=True)`` to make shuffle reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
