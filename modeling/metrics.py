"""Regression metrics for Chapter 3 experiments."""
from __future__ import annotations

import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon=1.0):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.maximum(np.abs(y_true), float(epsilon))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def phm_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    valid_mask = np.abs(y_true) > 1.0
    if not np.any(valid_mask):
        return 1.0
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]
    errors = (yt - yp) / yt * 100.0
    ln05 = np.log(0.5)
    negative_branch = np.clip(-ln05 * (errors / 5.0), -60.0, 60.0)
    positive_branch = np.clip(ln05 * (errors / 20.0), -60.0, 60.0)
    score = np.where(
        errors <= 0.0,
        np.exp(negative_branch),
        np.exp(positive_branch),
    )
    return float(np.mean(score))


def regression_metrics(y_true, y_pred, mape_epsilon=1.0):
    return {
        "mae": round(mae(y_true, y_pred), 6),
        "rmse": round(rmse(y_true, y_pred), 6),
        "mape": round(mape(y_true, y_pred, epsilon=mape_epsilon), 6),
        "phm_score": round(phm_score(y_true, y_pred), 6),
    }
