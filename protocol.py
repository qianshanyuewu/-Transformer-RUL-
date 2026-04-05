"""Frozen protocol constants shared by Chapter 2, 3 and 4."""
from __future__ import annotations

PROTOCOL_NAME = "buaa_paper"
PROTOCOL_LABEL_CN = "北航论文式固定12特征协议"

UNIFIED_VIBRATION_CHANNEL_MODE = "horizontal"
CHAPTER2_ANALYSIS_SCOPE = "all_bearings_for_chapter2"
FEATURE_SELECTION_MODE = "fixed_time_domain_12"

UNIFIED_CANDIDATE_BASE_FEATURES = [
    "atan_std",
    "asinh_std",
    "std",
    "peak_to_peak",
    "rms",
    "upper_bound",
    "impulse_factor",
    "crest_factor",
    "margin_factor",
    "energy",
    "kurtosis",
    "mean_abs",
    "skewness",
]

UNIFIED_SELECTED_BASE_FEATURES = [
    "atan_std",
    "asinh_std",
    "std",
    "peak_to_peak",
    "rms",
    "upper_bound",
    "impulse_factor",
    "crest_factor",
    "margin_factor",
    "energy",
    "kurtosis",
    "skewness",
]

REMOVED_BASE_FEATURES = ["mean_abs"]
UNIFIED_FEATURE_PREFIX = "h"

UNIFIED_WINDOW_SIZE = 10
UNIFIED_STEP_SIZE = 1
UNIFIED_RUL_TARGET_MODE = "life_ratio"
UNIFIED_FEATURE_SCALER_MODE = "minmax"
UNIFIED_DECODER_START_VALUE = 1.0
UNIFIED_USE_FPT_START = False

MANUAL_SEARCH_GRID = {
    "d_model": [32, 48],
    "num_heads": [2],
    "num_layers": [1, 2, 3],
    "ffn_dim": [64, 96],
    "dropout": [0.0, 0.1, 0.2],
    "learning_rate": [5e-4, 1e-3],
    "weight_decay": [1e-4],
}
MANUAL_SEARCH_PILOT_CONDITION = "35Hz12kN"
MANUAL_SEARCH_PILOT_SEED = 13
MANUAL_SEARCH_TOP_CONFIGS = 2
MANUAL_SEARCH_MAX_EPOCHS = 20
MANUAL_SEARCH_PATIENCE = 5

AUTOTUNE_TRIALS = 30
AUTOTUNE_SEED = 42
AUTOTUNE_SEARCH_SPACE = {
    "window_size": [8, 10, 12, 16],
    "d_model": [32, 48, 64],
    "num_heads": [2, 4],
    "num_layers": [1, 2, 3],
    "ffn_dim": [64, 96, 128, 192, 256],
    "dropout_range": [0.0, 0.3],
    "learning_rate_range": [3e-4, 3e-3],
    "weight_decay_range": [1e-6, 1e-3],
}


def prefixed_feature_names(base_features, prefix: str = UNIFIED_FEATURE_PREFIX):
    return [f"{prefix}_{feature_name}" for feature_name in base_features]


def fixed_rejected_feature_rows(prefix: str = UNIFIED_FEATURE_PREFIX):
    return [
        {
            "feature_name": f"{prefix}_{feature_name}",
            "redundant_with": None,
            "correlation": None,
            "reason": "removed_by_paper_fixed_feature_set",
        }
        for feature_name in REMOVED_BASE_FEATURES
    ]


UNIFIED_CANDIDATE_FEATURE_NAMES = prefixed_feature_names(UNIFIED_CANDIDATE_BASE_FEATURES)
UNIFIED_SELECTED_FEATURE_NAMES = prefixed_feature_names(UNIFIED_SELECTED_BASE_FEATURES)
