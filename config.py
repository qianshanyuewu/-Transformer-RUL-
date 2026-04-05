"""
重建版工程配置。
仅服务于 thesis_rebuild 目录，不依赖旧版默认参数。
"""
import os

from thesis_rebuild.protocol import (
    AUTOTUNE_SEED,
    AUTOTUNE_TRIALS,
    CHAPTER2_ANALYSIS_SCOPE,
    FEATURE_SELECTION_MODE,
    MANUAL_SEARCH_MAX_EPOCHS,
    MANUAL_SEARCH_PATIENCE,
    PROTOCOL_NAME,
    UNIFIED_DECODER_START_VALUE,
    UNIFIED_FEATURE_SCALER_MODE,
    UNIFIED_RUL_TARGET_MODE,
    UNIFIED_SELECTED_BASE_FEATURES,
    UNIFIED_STEP_SIZE,
    UNIFIED_USE_FPT_START,
    UNIFIED_VIBRATION_CHANNEL_MODE,
    UNIFIED_WINDOW_SIZE,
)


REBUILD_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REBUILD_ROOT)

DATASET_ROOT_CANDIDATES = [
    os.path.join(REBUILD_ROOT, "extracted", "XJTU-SY_Bearing_Datasets"),
    os.path.join(PROJECT_ROOT, "XJT U数据集", "extracted", "XJTU-SY_Bearing_Datasets"),
    os.path.join(PROJECT_ROOT, "extracted", "XJTU-SY_Bearing_Datasets"),
]
DATASET_ROOT = next(
    (path for path in DATASET_ROOT_CANDIDATES if os.path.exists(path)),
    DATASET_ROOT_CANDIDATES[0],
)

RESULTS_DIR = os.path.join(REBUILD_ROOT, "results")
FIGURES_DIR = os.path.join(REBUILD_ROOT, "figures")
DOCS_DIR = os.path.join(REBUILD_ROOT, "docs")

SAMPLING_RATE = 25600
SAMPLE_POINTS = 32768

CONDITIONS = {
    "35Hz12kN": {"rpm": 2100, "load": 12, "bearings": 5},
    "37.5Hz11kN": {"rpm": 2250, "load": 11, "bearings": 5},
    "40Hz10kN": {"rpm": 2400, "load": 10, "bearings": 5},
}

# 当前毕设口径：仅振动，不使用温度
USE_TEMPERATURE = False
VIBRATION_CHANNEL_MODE = UNIFIED_VIBRATION_CHANNEL_MODE

# 第2章重建起点
WAVELET_NAME = "db4"
WAVELET_LEVEL = 1
WAVELET_MODE = "soft"
SG_WINDOW = 11
SG_POLYORDER = 3
FEATURE_SCORE_WEIGHTS = {"monotonicity": 0.5, "correlation": 0.5}
MAX_FEATURE_CORRELATION = 0.95
TOP_K_FEATURES = len(UNIFIED_SELECTED_BASE_FEATURES)
CHAPTER2_PROTOCOL = PROTOCOL_NAME
CHAPTER2_ANALYSIS_MODE = CHAPTER2_ANALYSIS_SCOPE
FEATURE_SELECTION_PROTOCOL = FEATURE_SELECTION_MODE

# FPT
FPT_WINDOW = 10
FPT_SIGMA = 3.0
FPT_CONSECUTIVE = 3

# 第3章起点
WINDOW_SIZE = UNIFIED_WINDOW_SIZE
STEP_SIZE = UNIFIED_STEP_SIZE
CHAPTER3_PROTOCOL = PROTOCOL_NAME
CHAPTER3_USE_FPT_START = UNIFIED_USE_FPT_START
FEATURE_SCALER_MODE = UNIFIED_FEATURE_SCALER_MODE
RUL_TARGET_MODE = UNIFIED_RUL_TARGET_MODE
DECODER_START_VALUE = UNIFIED_DECODER_START_VALUE
MAPE_EPSILON = 1.0
TRAIN_BATCH_SIZE = 64
TRAIN_MAX_EPOCHS = 80
TRAIN_PATIENCE = 10
TRAIN_LEARNING_RATE = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
TRAIN_SEEDS = [13, 42, 3407]
MANUAL_TRAIN_MAX_EPOCHS = MANUAL_SEARCH_MAX_EPOCHS
MANUAL_TRAIN_PATIENCE = MANUAL_SEARCH_PATIENCE
AUTOTUNE_TRIAL_COUNT = AUTOTUNE_TRIALS
AUTOTUNE_RANDOM_SEED = AUTOTUNE_SEED

DEFAULT_INTRA_CONDITION_SPLIT = {
    "35Hz12kN": {
        "train": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
        "val": ["Bearing1_4"],
        "test": ["Bearing1_5"],
    },
    "37.5Hz11kN": {
        "train": ["Bearing2_1", "Bearing2_3", "Bearing2_5"],
        "val": ["Bearing2_2"],
        "test": ["Bearing2_4"],
    },
    "40Hz10kN": {
        "train": ["Bearing3_1", "Bearing3_2", "Bearing3_4"],
        "val": ["Bearing3_3"],
        "test": ["Bearing3_5"],
    },
}

for path in (RESULTS_DIR, FIGURES_DIR, DOCS_DIR):
    os.makedirs(path, exist_ok=True)
