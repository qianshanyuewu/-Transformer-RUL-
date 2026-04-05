"""重建版特征评分与去冗余。"""
import numpy as np


def _safe_abs_pearson(arr_a, arr_b):
    arr_a = np.asarray(arr_a, dtype=np.float64)
    arr_b = np.asarray(arr_b, dtype=np.float64)
    if arr_a.size <= 1 or arr_b.size <= 1 or arr_a.size != arr_b.size:
        return 0.0

    centered_a = arr_a - np.mean(arr_a)
    centered_b = arr_b - np.mean(arr_b)
    std_a = np.std(centered_a)
    std_b = np.std(centered_b)
    if std_a < 1e-12 or std_b < 1e-12:
        return 0.0

    normalized_a = centered_a / std_a
    normalized_b = centered_b / std_b
    corr = np.mean(normalized_a * normalized_b)
    if np.isnan(corr):
        return 0.0
    return float(abs(np.clip(corr, -1.0, 1.0)))


def calculate_monotonicity(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    diffs = np.diff(arr)
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)
    return float(abs(pos_count - neg_count) / max(arr.size - 1, 1))


def calculate_time_correlation(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1 or np.std(arr) == 0:
        return 0.0
    time_idx = np.arange(1, arr.size + 1, dtype=np.float64)
    return _safe_abs_pearson(time_idx, arr)


def calculate_pairwise_pearson(values_a, values_b):
    return _safe_abs_pearson(values_a, values_b)


def calculate_pairwise_pearson_across_sequences(sequence_list_a, sequence_list_b):
    if len(sequence_list_a) != len(sequence_list_b):
        raise ValueError("sequence lists must have the same number of bearings")
    corrs = []
    for values_a, values_b in zip(sequence_list_a, sequence_list_b):
        corrs.append(calculate_pairwise_pearson(values_a, values_b))
    return float(np.mean(corrs)) if corrs else 0.0


def aggregate_feature_scores(feature_sequences_by_feature, weights=None):
    if weights is None:
        weights = {"monotonicity": 0.5, "correlation": 0.5}

    aggregated_scores = []
    for feature_name, sequence_list in feature_sequences_by_feature.items():
        mono_scores = [calculate_monotonicity(values) for values in sequence_list]
        corr_scores = [calculate_time_correlation(values) for values in sequence_list]
        avg_mono = float(np.mean(mono_scores)) if mono_scores else 0.0
        avg_corr = float(np.mean(corr_scores)) if corr_scores else 0.0
        total_score = (
            weights["monotonicity"] * avg_mono
            + weights["correlation"] * avg_corr
        )
        aggregated_scores.append((feature_name, total_score, avg_mono, avg_corr))

    aggregated_scores.sort(key=lambda row: row[1], reverse=True)
    return aggregated_scores


def select_features_by_score_and_pearson(
    aggregated_scores,
    feature_sequences,
    top_k,
    max_feature_correlation,
):
    selected_names = []
    rejected_rows = []

    for feature_name, total_score, avg_mono, avg_corr in aggregated_scores:
        redundant_with = None
        redundant_corr = 0.0
        for selected_name in selected_names:
            corr = calculate_pairwise_pearson_across_sequences(
                feature_sequences[feature_name],
                feature_sequences[selected_name],
            )
            if corr >= max_feature_correlation:
                redundant_with = selected_name
                redundant_corr = corr
                break

        if redundant_with is not None:
            rejected_rows.append(
                {
                    "feature_name": feature_name,
                    "total_score": round(float(total_score), 6),
                    "avg_monotonicity": round(float(avg_mono), 6),
                    "avg_correlation": round(float(avg_corr), 6),
                    "redundant_with": redundant_with,
                    "mean_pearson": round(float(redundant_corr), 6),
                }
            )
            continue

        selected_names.append(feature_name)
        if len(selected_names) >= top_k:
            break

    return selected_names, rejected_rows
