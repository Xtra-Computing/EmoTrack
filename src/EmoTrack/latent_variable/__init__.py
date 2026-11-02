from __future__ import annotations

from EmoTrack.behavior_cue_mapping import PHQ8_RUNTIME_CUE_BANK_PATH
from EmoTrack.latent_variable.behavior_map import PHQ8_ITEMS, SymptomBehaviorTarget, build_symptom_behavior_targets
from EmoTrack.latent_variable.trend import LatentTrend, LatentTrendPlan, ResolvedLatentState
from EmoTrack.utils.paths import REPO_ROOT


LATENT_VARIABLE_ROOT = REPO_ROOT / "configs" / "latent_variable"
LATENT_MAPPING_PATH = PHQ8_RUNTIME_CUE_BANK_PATH
SCALES_ROOT = LATENT_VARIABLE_ROOT / "scales"
PHQ8_SCALE_PATH = SCALES_ROOT / "phq8.json"

__all__ = [
    "LATENT_MAPPING_PATH",
    "LatentTrend",
    "LatentTrendPlan",
    "LATENT_VARIABLE_ROOT",
    "PHQ8_SCALE_PATH",
    "PHQ8_ITEMS",
    "ResolvedLatentState",
    "SCALES_ROOT",
    "SymptomBehaviorTarget",
    "build_symptom_behavior_targets",
]
