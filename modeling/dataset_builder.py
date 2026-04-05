"""Chapter 3 dataset construction based on the rebuilt Chapter 2 pipeline."""
import json
import os

import numpy as np

from thesis_rebuild.config import (
    CHAPTER3_PROTOCOL,
    CHAPTER3_USE_FPT_START,
    CONDITIONS,
    DATASET_ROOT,
    DECODER_START_VALUE,
    DEFAULT_INTRA_CONDITION_SPLIT,
    FEATURE_SCALER_MODE,
    FEATURE_SCORE_WEIGHTS,
    FPT_CONSECUTIVE,
    FPT_SIGMA,
    FPT_WINDOW,
    MAX_FEATURE_CORRELATION,
    RESULTS_DIR,
    SAMPLING_RATE,
    SG_POLYORDER,
    SG_WINDOW,
    STEP_SIZE,
    TOP_K_FEATURES,
    RUL_TARGET_MODE,
    VIBRATION_CHANNEL_MODE,
    WAVELET_LEVEL,
    WAVELET_MODE,
    WAVELET_NAME,
    WINDOW_SIZE,
)
from thesis_rebuild.data_processing.chapter2_pipeline import process_single_bearing
from thesis_rebuild.data_processing.features import (
    PAPER_SELECTED_TIME_DOMAIN_FEATURES,
    PAPER_TIME_DOMAIN_FEATURES,
)
from thesis_rebuild.data_processing.health import cumulative_transform_features, summarize_health_stage
from thesis_rebuild.data_processing.io import normalize_intra_condition_split, parse_dataset
from thesis_rebuild.data_processing.selection import (
    aggregate_feature_scores,
    select_features_by_score_and_pearson,
)
from thesis_rebuild.protocol import (
    FEATURE_SELECTION_MODE,
    UNIFIED_CANDIDATE_FEATURE_NAMES,
    UNIFIED_SELECTED_FEATURE_NAMES,
    fixed_rejected_feature_rows,
)


DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "chapter3_datasets")


def _full_bearing_name(condition_name, bearing_name):
    return f"{condition_name}/{bearing_name}"


def _feature_sequences_for_bearings(all_bearing_data, bearing_names, feature_key):
    if not bearing_names:
        return {}
    feature_names = list(all_bearing_data[bearing_names[0]][feature_key].keys())
    feature_sequences = {feature_name: [] for feature_name in feature_names}
    for bearing_name in bearing_names:
        for feature_name in feature_names:
            feature_sequences[feature_name].append(all_bearing_data[bearing_name][feature_key][feature_name])
    return feature_sequences


def _paper_feature_names(prefix="h"):
    if prefix == "h":
        return list(UNIFIED_CANDIDATE_FEATURE_NAMES)
    return [f"{prefix}_{name}" for name in PAPER_TIME_DOMAIN_FEATURES]


def _paper_selected_names(prefix="h"):
    if prefix == "h":
        return list(UNIFIED_SELECTED_FEATURE_NAMES)
    return [f"{prefix}_{name}" for name in PAPER_SELECTED_TIME_DOMAIN_FEATURES]


def _feature_dict_to_matrix(feature_dict, feature_names):
    if not feature_names:
        return np.empty((0, 0), dtype=np.float32)
    matrix = np.asarray([feature_dict[name] for name in feature_names], dtype=np.float64).T
    if matrix.ndim != 2:
        return np.empty((0, len(feature_names)), dtype=np.float32)
    return matrix.astype(np.float32)


def _fit_feature_scaler(train_matrices, scaler_mode):
    valid_matrices = [matrix for matrix in train_matrices if matrix.size > 0]
    if not valid_matrices:
        raise ValueError("no valid training matrices to fit feature scaler")
    stacked = np.concatenate(valid_matrices, axis=0)
    if scaler_mode == "minmax":
        feature_min = np.min(stacked, axis=0)
        feature_max = np.max(stacked, axis=0)
        feature_range = feature_max - feature_min
        feature_range[feature_range < 1e-6] = 1.0
        return {
            "mode": "minmax",
            "offset": feature_min.astype(np.float32),
            "scale": feature_range.astype(np.float32),
        }

    mean = np.mean(stacked, axis=0, dtype=np.float64)
    std = np.std(stacked, axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    return {
        "mode": "standard",
        "offset": mean.astype(np.float32),
        "scale": std.astype(np.float32),
    }


def _scale_matrix(matrix, scaler_stats):
    if matrix.size == 0:
        return matrix.astype(np.float32)
    offset = scaler_stats["offset"]
    scale = scaler_stats["scale"]
    if scaler_stats["mode"] == "minmax":
        return ((matrix - offset) / scale).astype(np.float32)
    return ((matrix - offset) / scale).astype(np.float32)


def _build_target_series(total_samples, target_mode):
    if total_samples <= 0:
        return np.empty((0,), dtype=np.float32)

    if target_mode == "life_ratio":
        remaining = np.arange(total_samples, 0, -1, dtype=np.float32)
        denominator = max(float(total_samples - 1), 1.0)
        return np.clip((remaining - 1.0) / denominator, 0.0, 1.0).astype(np.float32)

    remaining = np.arange(total_samples - 1, -1, -1, dtype=np.float32)
    return remaining.astype(np.float32)


def _build_windows_for_bearing(
    bearing_name,
    split_name,
    scaled_matrix,
    total_samples,
    fpt_index,
    window_size,
    step_size,
    target_mode,
    decoder_start_value,
    use_fpt_start,
):
    feature_dim = scaled_matrix.shape[1] if scaled_matrix.ndim == 2 else 0
    if scaled_matrix.ndim != 2 or scaled_matrix.shape[0] == 0:
        return {
            "X": np.empty((0, window_size, feature_dim), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "decoder_input": np.empty((0, window_size, 1), dtype=np.float32),
            "bearing_names": [],
            "start_indices": [],
            "end_indices": [],
            "rul_labels": [],
            "total_samples": [],
            "target_labels": [],
            "summary": {
                "split": split_name,
                "bearing_name": bearing_name,
                "window_count": 0,
                "effective_length": 0,
                "status": "empty_matrix",
            },
        }

    start_index = 0
    if use_fpt_start and fpt_index is not None:
        start_index = int(fpt_index)
    start_index = max(0, min(start_index, scaled_matrix.shape[0] - 1))
    effective_length = int(scaled_matrix.shape[0] - start_index)

    if effective_length < window_size:
        return {
            "X": np.empty((0, window_size, feature_dim), dtype=np.float32),
            "y": np.empty((0,), dtype=np.float32),
            "decoder_input": np.empty((0, window_size, 1), dtype=np.float32),
            "bearing_names": [],
            "start_indices": [],
            "end_indices": [],
            "rul_labels": [],
            "total_samples": [],
            "target_labels": [],
            "summary": {
                "split": split_name,
                "bearing_name": bearing_name,
                "window_count": 0,
                "effective_length": effective_length,
                "status": "effective_length_too_short",
            },
        }

    target_series = _build_target_series(total_samples, target_mode=target_mode)
    windows = []
    labels = []
    decoder_inputs = []
    start_indices = []
    end_indices = []
    bearing_names = []
    rul_labels = []
    total_sample_labels = []

    for window_start in range(start_index, scaled_matrix.shape[0] - window_size + 1, step_size):
        window_end = window_start + window_size
        endpoint_index = window_end - 1
        target_window = target_series[window_start:window_end]
        shifted = np.concatenate((
            np.array([decoder_start_value], dtype=np.float32),
            np.asarray(target_window[:-1], dtype=np.float32),
        ))
        rul_label = float(total_samples - (endpoint_index + 1))
        windows.append(scaled_matrix[window_start:window_end])
        labels.append(float(target_window[-1]))
        decoder_inputs.append(shifted[:, None])
        start_indices.append(int(window_start))
        end_indices.append(int(endpoint_index))
        bearing_names.append(bearing_name)
        rul_labels.append(rul_label)
        total_sample_labels.append(float(total_samples))

    return {
        "X": np.stack(windows).astype(np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "decoder_input": np.stack(decoder_inputs).astype(np.float32),
        "bearing_names": bearing_names,
        "start_indices": start_indices,
        "end_indices": end_indices,
        "rul_labels": rul_labels,
        "total_samples": total_sample_labels,
        "target_labels": labels,
        "summary": {
            "split": split_name,
            "bearing_name": bearing_name,
            "window_count": len(windows),
            "effective_length": effective_length,
            "status": "ok",
        },
    }


def _concat_split_payloads(split_payloads, window_size, feature_dim):
    all_windows = [payload["X"] for payload in split_payloads if payload["X"].size > 0]
    all_labels = [payload["y"] for payload in split_payloads if payload["y"].size > 0]
    all_decoder_inputs = [payload["decoder_input"] for payload in split_payloads if payload["decoder_input"].size > 0]

    if all_windows:
        X = np.concatenate(all_windows, axis=0).astype(np.float32)
        y = np.concatenate(all_labels, axis=0).astype(np.float32)
        decoder_input = np.concatenate(all_decoder_inputs, axis=0).astype(np.float32)
    else:
        X = np.empty((0, window_size, feature_dim), dtype=np.float32)
        y = np.empty((0,), dtype=np.float32)
        decoder_input = np.empty((0, window_size, 1), dtype=np.float32)

    bearing_names = np.asarray(
        [name for payload in split_payloads for name in payload["bearing_names"]],
        dtype=object,
    )
    start_indices = np.asarray(
        [idx for payload in split_payloads for idx in payload["start_indices"]],
        dtype=np.int32,
    )
    end_indices = np.asarray(
        [idx for payload in split_payloads for idx in payload["end_indices"]],
        dtype=np.int32,
    )
    rul_labels = np.asarray(
        [value for payload in split_payloads for value in payload["rul_labels"]],
        dtype=np.float32,
    )
    total_samples = np.asarray(
        [value for payload in split_payloads for value in payload["total_samples"]],
        dtype=np.float32,
    )

    return {
        "X": X,
        "y": y,
        "decoder_input": decoder_input,
        "bearing_names": bearing_names,
        "start_indices": start_indices,
        "end_indices": end_indices,
        "rul_labels": rul_labels,
        "total_samples": total_samples,
    }


def _save_split_npz(split_path, payload, feature_names, scaler_stats, target_mode):
    np.savez_compressed(
        split_path,
        X=payload["X"],
        y=payload["y"],
        decoder_input=payload["decoder_input"],
        rul_labels=payload["rul_labels"],
        total_samples=payload["total_samples"],
        bearing_names=payload["bearing_names"],
        start_indices=payload["start_indices"],
        end_indices=payload["end_indices"],
        feature_names=np.asarray(feature_names, dtype=object),
        scaler_offset=np.asarray(scaler_stats["offset"], dtype=np.float32),
        scaler_scale=np.asarray(scaler_stats["scale"], dtype=np.float32),
        feature_scaler_mode=np.asarray(scaler_stats["mode"]),
        target_mode=np.asarray(target_mode),
    )


def build_condition_dataset(condition_name, split_config=None, config=None, output_dir=None):
    config = config or {}
    output_dir = output_dir or os.path.join(DEFAULT_OUTPUT_DIR, condition_name)
    os.makedirs(output_dir, exist_ok=True)

    all_conditions = [condition_name]
    split_config = normalize_intra_condition_split(
        split_config or DEFAULT_INTRA_CONDITION_SPLIT,
        all_conditions=all_conditions,
    )
    split = split_config[condition_name]

    chapter3_config = {
        "protocol": config.get("protocol", CHAPTER3_PROTOCOL),
        "wavelet_name": config.get("wavelet_name", WAVELET_NAME),
        "wavelet_level": int(config.get("wavelet_level", WAVELET_LEVEL)),
        "wavelet_mode": config.get("wavelet_mode", WAVELET_MODE),
        "sg_window": int(config.get("sg_window", SG_WINDOW)),
        "sg_polyorder": int(config.get("sg_polyorder", SG_POLYORDER)),
        "feature_score_weights": config.get("feature_score_weights", FEATURE_SCORE_WEIGHTS),
        "top_k_features": int(config.get("top_k_features", TOP_K_FEATURES)),
        "max_feature_correlation": float(config.get("max_feature_correlation", MAX_FEATURE_CORRELATION)),
        "fpt_window": int(config.get("fpt_window", FPT_WINDOW)),
        "fpt_sigma": float(config.get("fpt_sigma", FPT_SIGMA)),
        "fpt_consecutive": int(config.get("fpt_consecutive", FPT_CONSECUTIVE)),
        "vibration_channel_mode": config.get("vibration_channel_mode", VIBRATION_CHANNEL_MODE),
        "window_size": int(config.get("window_size", WINDOW_SIZE)),
        "step_size": int(config.get("step_size", STEP_SIZE)),
        "sampling_rate": int(config.get("sampling_rate", SAMPLING_RATE)),
        "rul_target_mode": config.get("rul_target_mode", RUL_TARGET_MODE),
        "feature_scaler_mode": config.get("feature_scaler_mode", FEATURE_SCALER_MODE),
        "decoder_start_value": float(config.get("decoder_start_value", DECODER_START_VALUE)),
        "use_fpt_start": bool(config.get("use_fpt_start", CHAPTER3_USE_FPT_START)),
        "feature_selection_mode": config.get("feature_selection_mode", FEATURE_SELECTION_MODE),
    }
    if chapter3_config["protocol"] == "buaa_paper":
        chapter3_config["vibration_channel_mode"] = "horizontal"

    dataset_info = parse_dataset(DATASET_ROOT, [condition_name])
    condition_bearings = dataset_info.get(condition_name, [])
    all_bearing_data = {}
    for bearing_info in condition_bearings:
        result = process_single_bearing(
            bearing_info=bearing_info,
            condition_name=condition_name,
            vibration_channel_mode=chapter3_config["vibration_channel_mode"],
            wavelet_name=chapter3_config["wavelet_name"],
            wavelet_level=chapter3_config["wavelet_level"],
            wavelet_mode=chapter3_config["wavelet_mode"],
            sg_window=chapter3_config["sg_window"],
            sg_polyorder=chapter3_config["sg_polyorder"],
        )
        if result is not None:
            all_bearing_data[result["bearing_name"]] = result

    train_full_names = [_full_bearing_name(condition_name, name) for name in split["train"]]
    val_full_names = [_full_bearing_name(condition_name, name) for name in split["val"]]
    test_full_names = [_full_bearing_name(condition_name, name) for name in split["test"]]
    expected_names = train_full_names + val_full_names + test_full_names
    missing_names = [name for name in expected_names if name not in all_bearing_data]
    if missing_names:
        raise ValueError(f"missing bearings for {condition_name}: {missing_names}")

    if chapter3_config["protocol"] == "buaa_paper":
        candidate_feature_names = _paper_feature_names(prefix="h")
        selected_names = _paper_selected_names(prefix="h")
        feature_sequences = {
            feature_name: [
                all_bearing_data[bearing_name]["smoothed_features"][feature_name]
                for bearing_name in train_full_names
            ]
            for feature_name in candidate_feature_names
        }
        aggregated_scores = aggregate_feature_scores(
            feature_sequences,
            weights=chapter3_config["feature_score_weights"],
        )
        score_lookup = {
            feature_name: (total_score, avg_mono, avg_corr)
            for feature_name, total_score, avg_mono, avg_corr in aggregated_scores
        }
        rejected_features = []
        for row in fixed_rejected_feature_rows():
            feature_name = row["feature_name"]
            total_score, avg_mono, avg_corr = score_lookup.get(feature_name, (0.0, 0.0, 0.0))
            rejected_features.append(
                {
                    "feature_name": feature_name,
                    "total_score": round(float(total_score), 6),
                    "avg_monotonicity": round(float(avg_mono), 6),
                    "avg_correlation": round(float(avg_corr), 6),
                    "redundant_with": row["redundant_with"],
                    "correlation": row["correlation"],
                    "reason": row["reason"],
                }
            )
    else:
        feature_sequences = _feature_sequences_for_bearings(
            all_bearing_data=all_bearing_data,
            bearing_names=train_full_names,
            feature_key="smoothed_features",
        )
        aggregated_scores = aggregate_feature_scores(
            feature_sequences,
            weights=chapter3_config["feature_score_weights"],
        )
        selected_names, rejected_features = select_features_by_score_and_pearson(
            aggregated_scores=aggregated_scores,
            feature_sequences=feature_sequences,
            top_k=chapter3_config["top_k_features"],
            max_feature_correlation=chapter3_config["max_feature_correlation"],
        )

    if not selected_names:
        raise ValueError(f"no selected features for {condition_name}")

    for bearing_name in expected_names:
        data = all_bearing_data[bearing_name]
        selected_features = {
            feature_name: data["smoothed_features"][feature_name]
            for feature_name in selected_names
        }
        engineered_features = cumulative_transform_features(selected_features)
        health_stage = summarize_health_stage(
            engineered_features,
            baseline_window=chapter3_config["fpt_window"],
            sigma=chapter3_config["fpt_sigma"],
            consecutive=chapter3_config["fpt_consecutive"],
        )
        feature_matrix = _feature_dict_to_matrix(engineered_features, selected_names)
        data["selected_features"] = selected_features
        data["engineered_features"] = engineered_features
        data["health_stage"] = health_stage
        data["engineered_feature_matrix"] = feature_matrix

    train_effective_matrices = []
    for bearing_name in train_full_names:
        data = all_bearing_data[bearing_name]
        matrix = data["engineered_feature_matrix"]
        effective_start = 0
        if chapter3_config["use_fpt_start"] and data["health_stage"]["fpt_index"] is not None:
            effective_start = int(data["health_stage"]["fpt_index"])
        train_effective_matrices.append(matrix[effective_start:])

    scaler_stats = _fit_feature_scaler(
        train_effective_matrices,
        scaler_mode=chapter3_config["feature_scaler_mode"],
    )

    split_payloads = {}
    bearing_reports = {}
    split_name_to_bearings = {
        "train": train_full_names,
        "val": val_full_names,
        "test": test_full_names,
    }

    for split_name, bearing_names in split_name_to_bearings.items():
        per_bearing_payloads = []
        for bearing_name in bearing_names:
            data = all_bearing_data[bearing_name]
            scaled_matrix = _scale_matrix(
                data["engineered_feature_matrix"],
                scaler_stats,
            )
            payload = _build_windows_for_bearing(
                bearing_name=bearing_name,
                split_name=split_name,
                scaled_matrix=scaled_matrix,
                total_samples=data["total_samples"],
                fpt_index=data["health_stage"]["fpt_index"],
                window_size=chapter3_config["window_size"],
                step_size=chapter3_config["step_size"],
                target_mode=chapter3_config["rul_target_mode"],
                decoder_start_value=chapter3_config["decoder_start_value"],
                use_fpt_start=chapter3_config["use_fpt_start"],
            )
            per_bearing_payloads.append(payload)
            bearing_reports[bearing_name] = {
                "split": split_name,
                "total_samples": int(data["total_samples"]),
                "selected_feature_count": len(selected_names),
                "fpt_index": data["health_stage"]["fpt_index"],
                "fpt_cycle": data["health_stage"]["fpt_cycle"],
                "effective_start": 0 if not chapter3_config["use_fpt_start"] else data["health_stage"]["fpt_index"],
                "effective_length": payload["summary"]["effective_length"],
                "window_count": payload["summary"]["window_count"],
                "status": payload["summary"]["status"],
            }

        split_payloads[split_name] = _concat_split_payloads(
            per_bearing_payloads,
            window_size=chapter3_config["window_size"],
            feature_dim=len(selected_names),
        )

    split_paths = {}
    split_summaries = {}
    for split_name, payload in split_payloads.items():
        split_path = os.path.join(output_dir, f"{split_name}.npz")
        _save_split_npz(
            split_path,
            payload,
            feature_names=selected_names,
            scaler_stats=scaler_stats,
            target_mode=chapter3_config["rul_target_mode"],
        )
        split_paths[split_name] = split_path
        split_summaries[split_name] = {
            "sample_count": int(payload["X"].shape[0]),
            "window_size": int(chapter3_config["window_size"]),
            "feature_dim": int(payload["X"].shape[2]) if payload["X"].ndim == 3 else len(selected_names),
            "bearing_count": len(split_name_to_bearings[split_name]),
        }

    report = {
        "protocol": chapter3_config["protocol"],
        "vibration_channel_mode": chapter3_config["vibration_channel_mode"],
        "feature_selection_mode": chapter3_config["feature_selection_mode"],
        "condition": condition_name,
        "config": chapter3_config,
        "split_config": split,
        "candidate_feature_names": list(candidate_feature_names),
        "selected_features": selected_names,
        "selected_count": len(selected_names),
        "rejected_features": rejected_features,
        "feature_scores": [
            {
                "feature_name": feature_name,
                "total_score": round(float(total_score), 6),
                "avg_monotonicity": round(float(avg_mono), 6),
                "avg_correlation": round(float(avg_corr), 6),
                "selected": feature_name in selected_names,
            }
            for feature_name, total_score, avg_mono, avg_corr in aggregated_scores
        ],
        "scaler": {
            "mode": scaler_stats["mode"],
            "offset": [round(float(x), 6) for x in scaler_stats["offset"].tolist()],
            "scale": [round(float(x), 6) for x in scaler_stats["scale"].tolist()],
        },
        "split_summary": split_summaries,
        "bearings": bearing_reports,
        "paths": split_paths,
    }

    report_path = os.path.join(output_dir, "dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report, report_path


def build_chapter3_datasets(config=None, output_dir=None):
    config = config or {}
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    split_config = normalize_intra_condition_split(
        config.get("split_config") or DEFAULT_INTRA_CONDITION_SPLIT,
        all_conditions=list(CONDITIONS.keys()),
    )

    condition_reports = {}
    for condition_name in CONDITIONS:
        report, report_path = build_condition_dataset(
            condition_name=condition_name,
            split_config=split_config,
            config=config,
            output_dir=os.path.join(output_dir, condition_name),
        )
        condition_reports[condition_name] = {
            "selected_count": report["selected_count"],
            "selected_features": report["selected_features"],
            "split_summary": report["split_summary"],
            "report_path": report_path,
        }

    manifest = {
        "protocol": config.get("protocol", CHAPTER3_PROTOCOL),
        "conditions": condition_reports,
        "output_dir": output_dir,
    }
    manifest_path = os.path.join(output_dir, "chapter3_dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest, manifest_path
