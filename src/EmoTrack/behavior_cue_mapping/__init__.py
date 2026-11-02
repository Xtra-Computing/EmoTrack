# Shared path for the tracked PHQ-8 runtime cue bank.
from __future__ import annotations

from EmoTrack.utils.paths import REPO_ROOT


PHQ8_RUNTIME_CUE_BANK_PATH = (
    REPO_ROOT / "configs" / "latent_variable" / "phq8_behavior_cue_bank.json"
)

__all__ = [
    "PHQ8_RUNTIME_CUE_BANK_PATH",
]
