"""重建版第2章：固定12特征协议下的健康状态识别。"""
import json
import os

import numpy as np

from thesis_rebuild.config import (
    CHAPTER2_ANALYSIS_MODE,
    CHAPTER2_PROTOCOL,
    CONDITIONS,
    DATASET_ROOT,
    FPT_CONSECUTIVE,
    FPT_SIGMA,
    FPT_WINDOW,
    FEATURE_SELECTION_PROTOCOL,
    FEATURE_SCORE_WEIGHTS,
    MAX_FEATURE_CORRELATION,
    RESULTS_DIR,
    SAMPLING_RATE,
    SG_POLYORDER,
    SG_WINDOW,
    TOP_K_FEATURES,
    VIBRATION_CHANNEL_MODE,
    WAVELET_LEVEL,
    WAVELET_MODE,
    WAVELET_NAME,
)
from thesis_rebuild.data_processing.features import denoise_signal_wavelet, extract_features
from thesis_rebuild.data_processing.health import cumulative_transform_features, summarize_health_stage
from thesis_rebuild.data_processing.io import (
    get_channel_signals,
    load_bearing_signals,
    parse_dataset,
)
from thesis_rebuild.data_processing.selection import (
    aggregate_feature_scores,
    calculate_pairwise_pearson_across_sequences,
)
from thesis_rebuild.protocol import (
    UNIFIED_CANDIDATE_FEATURE_NAMES,
    UNIFIED_SELECTED_FEATURE_NAMES,
    fixed_rejected_feature_rows,
)


DEFAULT_OUTPUT_DIR = os.path.join(RESULTS_DIR, "chapter2")


def process_single_bearing(
    bearing_info,
    condition_name,
    vibration_channel_mode,
    wavelet_name,
    wavelet_level,
    wavelet_mode,
    sg_window,
    sg_polyorder,
):
    from scipy.signal import savgol_filter

    signals = load_bearing_signals(bearing_info)
    if not signals:
        return None

    candidate_feature_names = list(UNIFIED_CANDIDATE_FEATURE_NAMES)
    raw_features = {feature_name: [] for feature_name in candidate_feature_names}
    channel_signals = get_channel_signals(signals, vibration_channel_mode)

    for prefix, selected_signals in channel_signals:
        for signal in selected_signals:
            denoised = denoise_signal_wavelet(
                signal,
                wavelet=wavelet_name,
                level=wavelet_level,
                mode=wavelet_mode,
            )
            feature_row = extract_features(denoised, sampling_rate=SAMPLING_RATE)
            for feature_name in candidate_feature_names:
                base_feature_name = feature_name.split("_", 1)[1]
                raw_features[feature_name].append(feature_row[base_feature_name])

    smoothed_features = {}
    for feature_name, values in raw_features.items():
        arr = np.asarray(values, dtype=np.float64)
        if arr.size < 5:
            smoothed_features[feature_name] = arr.tolist()
            continue
        window = min(sg_window, arr.size if arr.size % 2 == 1 else arr.size - 1)
        if window < 3:
            smoothed_features[feature_name] = arr.tolist()
            continue
        polyorder = min(sg_polyorder, window - 1)
        smoothed = savgol_filter(arr, window_length=window, polyorder=polyorder, mode="interp")
        smoothed_features[feature_name] = smoothed.tolist()

    return {
        "bearing_name": f"{condition_name}/{bearing_info['bearing_name']}",
        "condition": condition_name,
        "total_samples": bearing_info["total_files"],
        "raw_features": raw_features,
        "smoothed_features": smoothed_features,
    }


def build_feature_sequence_accumulator(all_bearing_data, bearing_names, feature_key):
    all_feature_names = list(next(iter(all_bearing_data.values()))[feature_key].keys())
    feature_sequences = {feature_name: [] for feature_name in all_feature_names}
    for bearing_name in bearing_names:
        for feature_name in all_feature_names:
            feature_sequences[feature_name].append(all_bearing_data[bearing_name][feature_key][feature_name])
    return feature_sequences


def build_mean_correlation_matrix(feature_sequences, feature_names):
    n_features = len(feature_names)
    corr_matrix = np.eye(n_features, dtype=float)
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = calculate_pairwise_pearson_across_sequences(
                feature_sequences[feature_names[i]],
                feature_sequences[feature_names[j]],
            )
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
    return corr_matrix


def summarize_dataset(all_bearing_data):
    per_condition = {}
    total_samples = 0
    total_bearings = 0
    for condition in CONDITIONS:
        bearing_names = sorted(
            name for name, data in all_bearing_data.items() if data["condition"] == condition
        )
        lifetimes = [all_bearing_data[name]["total_samples"] for name in bearing_names]
        if not lifetimes:
            continue
        per_condition[condition] = {
            "bearing_names": bearing_names,
            "bearing_count": len(lifetimes),
            "lifetime_min": int(np.min(lifetimes)),
            "lifetime_max": int(np.max(lifetimes)),
            "lifetime_mean": round(float(np.mean(lifetimes)), 2),
            "total_samples": int(np.sum(lifetimes)),
        }
        total_samples += int(np.sum(lifetimes))
        total_bearings += len(lifetimes)
    return {
        "condition_count": len(per_condition),
        "bearing_count": total_bearings,
        "total_samples": total_samples,
        "per_condition": per_condition,
    }


def build_report(
    all_bearing_data,
    analysis_bearings,
    aggregated_scores,
    selected_names,
    rejected_features,
    candidate_corr_matrix,
    selected_corr_matrix,
    config,
):
    feature_scores = [
        {
            "feature_name": feature_name,
            "total_score": round(float(total_score), 6),
            "avg_monotonicity": round(float(avg_mono), 6),
            "avg_correlation": round(float(avg_corr), 6),
            "selected": feature_name in selected_names,
        }
        for feature_name, total_score, avg_mono, avg_corr in aggregated_scores
    ]

    bearing_summary = {}
    detected_fpts = []
    for bearing_name, data in all_bearing_data.items():
        health_stage = data["health_stage"]
        if health_stage["fpt_index"] is not None:
            detected_fpts.append(health_stage["fpt_index"])
        bearing_summary[bearing_name] = {
            "condition": data["condition"],
            "total_samples": int(data["total_samples"]),
            "selected_features": list(data["selected_features"].keys()),
            "health_stage": health_stage,
        }

    return {
        "protocol": config["protocol"],
        "vibration_channel_mode": config["vibration_channel_mode"],
        "feature_selection_mode": config["feature_selection_mode"],
        "candidate_feature_names": [row["feature_name"] for row in feature_scores],
        "selected_features": selected_names,
        "rejected_features": rejected_features,
        "config": config,
        "dataset_summary": summarize_dataset(all_bearing_data),
        "analysis_scope": {
            "mode": config["analysis_scope"],
            "bearing_names": analysis_bearings,
        },
        "feature_scores": feature_scores,
        "selection": {
            "candidate_count": len(feature_scores),
            "selected_count": len(selected_names),
            "selected_features": selected_names,
            "rejected_features": rejected_features,
            "top_k_requested": int(config["top_k_features"]),
            "max_feature_correlation": float(config["max_feature_correlation"]),
            "mode": config["feature_selection_mode"],
        },
        "correlations": {
            "candidate_feature_names": [row["feature_name"] for row in feature_scores],
            "candidate_mean_pearson_matrix": candidate_corr_matrix.round(6).tolist(),
            "selected_feature_names": selected_names,
            "selected_mean_pearson_matrix": selected_corr_matrix.round(6).tolist(),
        },
        "fpt_summary": {
            "detected_count": len(detected_fpts),
            "fpt_min": None if not detected_fpts else int(np.min(detected_fpts)),
            "fpt_max": None if not detected_fpts else int(np.max(detected_fpts)),
            "fpt_mean": None if not detected_fpts else round(float(np.mean(detected_fpts)), 2),
        },
        "bearings": bearing_summary,
    }


def save_report(report, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "chapter2_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report_path


def run_chapter2_pipeline(config=None, output_dir=None):
    config = config or {}
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    active_conditions = list(config.get("conditions") or CONDITIONS.keys())
    chapter2_config = {
        "protocol": config.get("protocol", CHAPTER2_PROTOCOL),
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
        "feature_selection_mode": config.get("feature_selection_mode", FEATURE_SELECTION_PROTOCOL),
    }
    chapter2_config["analysis_scope"] = config.get("analysis_scope", CHAPTER2_ANALYSIS_MODE)

    dataset_info = parse_dataset(DATASET_ROOT, active_conditions)
    all_bearing_data = {}
    for condition, bearings in dataset_info.items():
        for bearing_info in bearings:
            result = process_single_bearing(
                bearing_info=bearing_info,
                condition_name=condition,
                vibration_channel_mode=chapter2_config["vibration_channel_mode"],
                wavelet_name=chapter2_config["wavelet_name"],
                wavelet_level=chapter2_config["wavelet_level"],
                wavelet_mode=chapter2_config["wavelet_mode"],
                sg_window=chapter2_config["sg_window"],
                sg_polyorder=chapter2_config["sg_polyorder"],
            )
            if result is not None:
                all_bearing_data[result["bearing_name"]] = result

    analysis_bearings = sorted(all_bearing_data.keys())
    feature_sequences = build_feature_sequence_accumulator(
        all_bearing_data, analysis_bearings, feature_key="smoothed_features"
    )
    aggregated_scores = aggregate_feature_scores(
        feature_sequences,
        weights=chapter2_config["feature_score_weights"],
    )
    selected_names = list(UNIFIED_SELECTED_FEATURE_NAMES)
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

    candidate_feature_names = [row[0] for row in aggregated_scores]
    candidate_corr_matrix = build_mean_correlation_matrix(feature_sequences, candidate_feature_names)
    selected_sequences = {name: feature_sequences[name] for name in selected_names}
    selected_corr_matrix = build_mean_correlation_matrix(selected_sequences, selected_names)

    for bearing_name, data in all_bearing_data.items():
        selected_features = {
            feature_name: data["smoothed_features"][feature_name]
            for feature_name in selected_names
        }
        engineered_features = cumulative_transform_features(selected_features)
        health_stage = summarize_health_stage(
            engineered_features,
            baseline_window=chapter2_config["fpt_window"],
            sigma=chapter2_config["fpt_sigma"],
            consecutive=chapter2_config["fpt_consecutive"],
        )
        data["selected_features"] = selected_features
        data["engineered_features"] = engineered_features
        data["health_stage"] = health_stage

    report = build_report(
        all_bearing_data=all_bearing_data,
        analysis_bearings=analysis_bearings,
        aggregated_scores=aggregated_scores,
        selected_names=selected_names,
        rejected_features=rejected_features,
        candidate_corr_matrix=candidate_corr_matrix,
        selected_corr_matrix=selected_corr_matrix,
        config=chapter2_config,
    )
    report_path = save_report(report, output_dir)
    return report, all_bearing_data, report_path
