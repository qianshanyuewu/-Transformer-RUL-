"""重建版特征工程。"""
import numpy as np
import pywt
from scipy.stats import kurtosis, skew

from thesis_rebuild.protocol import (
    UNIFIED_CANDIDATE_BASE_FEATURES,
    UNIFIED_SELECTED_BASE_FEATURES,
)


PAPER_TIME_DOMAIN_FEATURES = list(UNIFIED_CANDIDATE_BASE_FEATURES)
PAPER_SELECTED_TIME_DOMAIN_FEATURES = list(UNIFIED_SELECTED_BASE_FEATURES)

FREQ_DOMAIN_FEATURES = [
    "freq_centroid",
    "freq_mean_square",
    "freq_peak",
    "freq_band_energy_ratio",
    "freq_std",
    "freq_entropy",
]

TIME_DOMAIN_FEATURES = PAPER_TIME_DOMAIN_FEATURES
FEATURE_NAMES = TIME_DOMAIN_FEATURES + FREQ_DOMAIN_FEATURES

BASE_FEATURE_CN_MAP = {
    "atan_std": "反正切标准差",
    "asinh_std": "反双曲正弦标准差",
    "std": "标准差",
    "peak_to_peak": "峰峰值",
    "rms": "均方根",
    "upper_bound": "上边界",
    "impulse_factor": "冲击因子",
    "crest_factor": "峰值因子",
    "margin_factor": "裕度因子",
    "energy": "能量",
    "kurtosis": "峭度",
    "mean_abs": "平均绝对值",
    "skewness": "偏度",
    "freq_centroid": "频谱质心",
    "freq_mean_square": "频谱均方值",
    "freq_peak": "频谱峰值频率",
    "freq_band_energy_ratio": "频带能量比",
    "freq_std": "频谱标准差",
    "freq_entropy": "频谱熵",
}

CHANNEL_CN_MAP = {
    "h": "水平",
    "v": "垂直",
}


def feature_display_name(feature_name):
    if "_" not in feature_name:
        return BASE_FEATURE_CN_MAP.get(feature_name, feature_name)
    prefix, base_name = feature_name.split("_", 1)
    channel = CHANNEL_CN_MAP.get(prefix, prefix)
    base_label = BASE_FEATURE_CN_MAP.get(base_name, base_name)
    return f"{channel}{base_label}"


def denoise_signal_wavelet(signal, wavelet="db4", level=1, mode="soft"):
    arr = np.asarray(signal, dtype=np.float64)
    coeffs = pywt.wavedec(arr, wavelet=wavelet, level=level)
    if len(coeffs) <= 1:
        return arr.astype(np.float32)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
    threshold = sigma * np.sqrt(2.0 * np.log(arr.size)) if sigma > 0 else 0.0

    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode=mode))

    reconstructed = pywt.waverec(denoised_coeffs, wavelet)
    return reconstructed[: arr.size].astype(np.float32)


def _extract_freq_features(arr, sampling_rate):
    n = arr.size
    fft_vals = np.fft.rfft(arr)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    mag_sum = np.sum(fft_mag)
    if mag_sum < 1e-12:
        return {name: 0.0 for name in FREQ_DOMAIN_FEATURES}

    mag_norm = fft_mag / mag_sum
    centroid = float(np.sum(freqs * mag_norm))
    mean_sq_freq = float(np.sqrt(np.sum(freqs ** 2 * mag_norm)))
    peak_freq = float(freqs[np.argmax(fft_mag)])
    band_ratio = float(np.sum(fft_mag[freqs > sampling_rate / 4.0]) / mag_sum)
    freq_std = float(np.sqrt(np.sum((freqs - centroid) ** 2 * mag_norm)))

    power = fft_mag ** 2
    power_sum = np.sum(power)
    if power_sum > 1e-12:
        p = power / power_sum
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = 0.0

    return {
        "freq_centroid": round(centroid, 6),
        "freq_mean_square": round(mean_sq_freq, 6),
        "freq_peak": round(peak_freq, 6),
        "freq_band_energy_ratio": round(band_ratio, 6),
        "freq_std": round(freq_std, 6),
        "freq_entropy": round(entropy, 6),
    }


def extract_features(signal, sampling_rate):
    arr = np.asarray(signal, dtype=np.float64)
    n = arr.size
    if n == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    std_val = np.std(arr)
    peak_abs = np.max(np.abs(arr))
    pp_val = np.max(arr) - np.min(arr)
    rms_val = np.sqrt(np.mean(arr ** 2))
    mean_abs = np.mean(np.abs(arr))
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(arr)))
    energy = np.sum(arr ** 2)
    kurt_val = kurtosis(arr, fisher=False)
    skew_val = skew(arr)
    if np.isnan(kurt_val):
        kurt_val = 0.0
    if np.isnan(skew_val):
        skew_val = 0.0

    atan_std = np.std(np.arctan(arr))
    asinh_std = np.std(np.arcsinh(arr))
    upper_bound = np.max(arr) + 0.5 * (np.max(arr) - np.min(arr)) / max(n - 1, 1)
    impulse_factor = peak_abs / mean_abs if mean_abs > 0 else 0.0
    crest_factor = peak_abs / rms_val if rms_val > 0 else 0.0
    margin_factor = peak_abs / (sqrt_abs_mean ** 2) if sqrt_abs_mean > 0 else 0.0

    features = {
        "atan_std": round(float(atan_std), 6),
        "asinh_std": round(float(asinh_std), 6),
        "std": round(float(std_val), 6),
        "rms": round(float(rms_val), 6),
        "peak_to_peak": round(float(pp_val), 6),
        "upper_bound": round(float(upper_bound), 6),
        "impulse_factor": round(float(impulse_factor), 6),
        "crest_factor": round(float(crest_factor), 6),
        "margin_factor": round(float(margin_factor), 6),
        "energy": round(float(energy), 6),
        "kurtosis": round(float(kurt_val), 6),
        "mean_abs": round(float(mean_abs), 6),
        "skewness": round(float(skew_val), 6),
    }
    features.update(_extract_freq_features(arr, sampling_rate))
    return features
