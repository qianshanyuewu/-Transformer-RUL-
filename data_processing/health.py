"""重建版健康状态识别。"""
import numpy as np


def cumulative_transform_feature(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    cumulative_sum = np.cumsum(arr)
    cumulative_energy = np.cumsum(arr ** 2)
    scale = np.sqrt(cumulative_energy)
    scale[scale < 1e-12] = 1.0
    return (cumulative_sum / scale).tolist()


def cumulative_transform_features(feature_dict):
    return {
        feature_name: cumulative_transform_feature(values)
        for feature_name, values in feature_dict.items()
    }


def build_health_indicator(feature_dict, baseline_window):
    feature_matrix = np.array(list(feature_dict.values()), dtype=np.float64).T
    if feature_matrix.ndim != 2 or feature_matrix.size == 0:
        return np.array([], dtype=np.float32)

    baseline_end = min(max(int(baseline_window), 3), feature_matrix.shape[0])
    baseline = feature_matrix[:baseline_end]
    baseline_mean = np.mean(baseline, axis=0)
    baseline_std = np.std(baseline, axis=0)
    baseline_std[baseline_std < 1e-6] = 1.0

    z_scores = np.abs((feature_matrix - baseline_mean) / baseline_std)
    return np.mean(z_scores, axis=1).astype(np.float32)


def detect_fpt_by_sigma(health_indicator, baseline_window, sigma, consecutive):
    arr = np.asarray(health_indicator, dtype=np.float64)
    if arr.size == 0:
        return {"threshold": 0.0, "fpt_index": None, "healthy_samples": 0, "degrading_samples": 0}

    baseline_end = min(max(int(baseline_window), 3), arr.size)
    threshold = float(np.mean(arr[:baseline_end]) + sigma * np.std(arr[:baseline_end]))
    consecutive = max(int(consecutive), 1)

    fpt_index = None
    for idx in range(baseline_end, max(arr.size - consecutive + 1, baseline_end)):
        if np.all(arr[idx: idx + consecutive] > threshold):
            fpt_index = idx
            break

    healthy_samples = arr.size if fpt_index is None else int(fpt_index)
    degrading_samples = 0 if fpt_index is None else int(arr.size - fpt_index)
    return {
        "threshold": threshold,
        "fpt_index": None if fpt_index is None else int(fpt_index),
        "healthy_samples": healthy_samples,
        "degrading_samples": degrading_samples,
    }


def summarize_health_stage(feature_dict, baseline_window, sigma, consecutive):
    health_indicator = build_health_indicator(feature_dict, baseline_window=baseline_window)
    fpt_result = detect_fpt_by_sigma(
        health_indicator,
        baseline_window=baseline_window,
        sigma=sigma,
        consecutive=consecutive,
    )
    fpt_index = fpt_result["fpt_index"]
    total_samples = int(health_indicator.size)
    remaining_rul_at_fpt = None if fpt_index is None else int(total_samples - fpt_index)
    return {
        "health_indicator": health_indicator.tolist(),
        "threshold": round(float(fpt_result["threshold"]), 6),
        "fpt_index": fpt_index,
        "fpt_cycle": None if fpt_index is None else int(fpt_index + 1),
        "healthy_samples": int(fpt_result["healthy_samples"]),
        "degrading_samples": int(fpt_result["degrading_samples"]),
        "remaining_rul_at_fpt": remaining_rul_at_fpt,
    }
