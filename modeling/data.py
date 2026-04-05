"""Data loading utilities for Chapter 3 modeling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitArrays:
    X: np.ndarray
    y: np.ndarray
    decoder_input: np.ndarray
    rul_labels: np.ndarray
    total_samples: np.ndarray
    bearing_names: np.ndarray
    start_indices: np.ndarray
    end_indices: np.ndarray
    feature_names: np.ndarray
    scaler_offset: np.ndarray
    scaler_scale: np.ndarray
    feature_scaler_mode: str
    target_mode: str


class BearingWindowDataset(Dataset):
    def __init__(self, split_arrays: SplitArrays):
        self.X = torch.as_tensor(split_arrays.X, dtype=torch.float32)
        self.y = torch.as_tensor(split_arrays.y, dtype=torch.float32)
        self.decoder_input = torch.as_tensor(split_arrays.decoder_input, dtype=torch.float32)
        self.rul_labels = torch.as_tensor(split_arrays.rul_labels, dtype=torch.float32)
        self.total_samples = torch.as_tensor(split_arrays.total_samples, dtype=torch.float32)
        self.bearing_names = split_arrays.bearing_names
        self.start_indices = split_arrays.start_indices
        self.end_indices = split_arrays.end_indices
        self.target_mode = split_arrays.target_mode

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, index):
        return {
            "X": self.X[index],
            "y": self.y[index],
            "decoder_input": self.decoder_input[index],
            "rul_label": self.rul_labels[index],
            "total_samples": self.total_samples[index],
            "bearing_name": self.bearing_names[index],
            "start_index": self.start_indices[index],
            "end_index": self.end_indices[index],
        }


def load_split_arrays(npz_path: str | Path) -> SplitArrays:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return SplitArrays(
        X=data["X"].astype(np.float32),
        y=data["y"].astype(np.float32),
        decoder_input=data["decoder_input"].astype(np.float32) if "decoder_input" in data else np.empty((0, 0, 0), dtype=np.float32),
        rul_labels=data["rul_labels"].astype(np.float32) if "rul_labels" in data else np.empty((0,), dtype=np.float32),
        total_samples=data["total_samples"].astype(np.float32) if "total_samples" in data else np.empty((0,), dtype=np.float32),
        bearing_names=data["bearing_names"],
        start_indices=data["start_indices"].astype(np.int32),
        end_indices=data["end_indices"].astype(np.int32),
        feature_names=data["feature_names"],
        scaler_offset=data["scaler_offset"].astype(np.float32) if "scaler_offset" in data else data["scaler_mean"].astype(np.float32),
        scaler_scale=data["scaler_scale"].astype(np.float32) if "scaler_scale" in data else data["scaler_std"].astype(np.float32),
        feature_scaler_mode=(data["feature_scaler_mode"].item() if "feature_scaler_mode" in data else "standard"),
        target_mode=(data["target_mode"].item() if "target_mode" in data else "rul"),
    )


def merge_split_arrays(npz_paths: list[str | Path]) -> SplitArrays:
    if not npz_paths:
        raise ValueError("npz_paths must not be empty")

    split_arrays_list = [load_split_arrays(path) for path in npz_paths]
    reference = split_arrays_list[0]
    reference_feature_names = reference.feature_names.tolist()
    feature_dim = int(reference.X.shape[2]) if reference.X.ndim == 3 else 0

    for split_arrays in split_arrays_list[1:]:
        if split_arrays.X.ndim != reference.X.ndim or int(split_arrays.X.shape[2]) != feature_dim:
            raise ValueError("all merged splits must share the same feature dimension")
        if split_arrays.target_mode != reference.target_mode:
            raise ValueError("all merged splits must share the same target_mode")
        if split_arrays.feature_names.tolist() != reference_feature_names:
            raise ValueError("all merged splits must share identical feature names")

    return SplitArrays(
        X=np.concatenate([item.X for item in split_arrays_list], axis=0).astype(np.float32),
        y=np.concatenate([item.y for item in split_arrays_list], axis=0).astype(np.float32),
        decoder_input=np.concatenate([item.decoder_input for item in split_arrays_list], axis=0).astype(np.float32),
        rul_labels=np.concatenate([item.rul_labels for item in split_arrays_list], axis=0).astype(np.float32),
        total_samples=np.concatenate([item.total_samples for item in split_arrays_list], axis=0).astype(np.float32),
        bearing_names=np.concatenate([item.bearing_names for item in split_arrays_list], axis=0),
        start_indices=np.concatenate([item.start_indices for item in split_arrays_list], axis=0).astype(np.int32),
        end_indices=np.concatenate([item.end_indices for item in split_arrays_list], axis=0).astype(np.int32),
        feature_names=reference.feature_names,
        scaler_offset=np.zeros((feature_dim,), dtype=np.float32),
        scaler_scale=np.ones((feature_dim,), dtype=np.float32),
        feature_scaler_mode="mixed_per_condition",
        target_mode=reference.target_mode,
    )


def build_dataloader(
    npz_path: str | Path,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> tuple[BearingWindowDataset, DataLoader]:
    split_arrays = load_split_arrays(npz_path)
    dataset = BearingWindowDataset(split_arrays)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataset, loader


def build_merged_dataloader(
    npz_paths: list[str | Path],
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
) -> tuple[BearingWindowDataset, DataLoader]:
    split_arrays = merge_split_arrays(npz_paths)
    dataset = BearingWindowDataset(split_arrays)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataset, loader
