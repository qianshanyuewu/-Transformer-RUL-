"""重建版数据读取与划分辅助。"""
import glob
import os

import numpy as np


def numerical_sort_key(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    digits = "".join(ch for ch in filename if ch.isdigit())
    return int(digits) if digits else filename


def read_csv_file(filepath):
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def parse_bearing_folder(bearing_path):
    csv_files = sorted(glob.glob(os.path.join(bearing_path, "*.csv")), key=numerical_sort_key)
    return {
        "bearing_name": os.path.basename(bearing_path),
        "folder_path": bearing_path,
        "csv_files": csv_files,
        "total_files": len(csv_files),
    }


def parse_dataset(dataset_root, conditions):
    dataset_info = {}
    for condition in conditions:
        condition_path = os.path.join(dataset_root, condition)
        if not os.path.exists(condition_path):
            continue
        bearing_folders = sorted(glob.glob(os.path.join(condition_path, "Bearing*")))
        dataset_info[condition] = [
            parse_bearing_folder(folder)
            for folder in bearing_folders
            if parse_bearing_folder(folder)["total_files"] > 0
        ]
    return dataset_info


def load_bearing_signals(bearing_info):
    signals = []
    for csv_path in bearing_info["csv_files"]:
        try:
            h_signal, v_signal = read_csv_file(csv_path)
        except Exception:
            continue
        signals.append((h_signal, v_signal))
    return signals


def get_channel_signals(signals, vibration_channel_mode):
    if vibration_channel_mode == "horizontal":
        return [("h", [sig[0] for sig in signals])]
    if vibration_channel_mode == "vertical":
        return [("v", [sig[1] for sig in signals])]
    if vibration_channel_mode == "both":
        return [
            ("h", [sig[0] for sig in signals]),
            ("v", [sig[1] for sig in signals]),
        ]
    raise ValueError(f"unsupported vibration_channel_mode: {vibration_channel_mode}")


def iter_full_bearing_names(split_config, split_name):
    full_names = []
    for condition, split in split_config.items():
        for bearing_name in split.get(split_name, []):
            full_names.append(f"{condition}/{bearing_name}")
    return full_names


def normalize_intra_condition_split(split_config, all_conditions):
    normalized = {}
    for condition in all_conditions:
        split = split_config.get(condition, {})
        normalized[condition] = {
            "train": list(split.get("train", [])),
            "val": list(split.get("val", [])),
            "test": list(split.get("test", [])),
        }
    return normalized
