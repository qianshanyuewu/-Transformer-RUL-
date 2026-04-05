"""Baseline experiment utilities for Chapter 3."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from thesis_rebuild.config import (
    TRAIN_BATCH_SIZE,
    TRAIN_LEARNING_RATE,
    TRAIN_MAX_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_SEEDS,
    TRAIN_WEIGHT_DECAY,
)
from thesis_rebuild.modeling.data import build_dataloader, build_merged_dataloader, load_split_arrays, merge_split_arrays
from thesis_rebuild.modeling.models import build_model
from thesis_rebuild.modeling.trainer import evaluate_model, set_seed, train_model


DEFAULT_BASELINE_MODEL_CONFIGS = {
    "paper_transformer": {"d_model": 32, "num_heads": 2, "num_layers": 2, "ffn_dim": 64, "dropout": 0.1},
    "encoder_only_transformer": {"d_model": 32, "num_heads": 2, "num_layers": 2, "ffn_dim": 64, "dropout": 0.1},
    "lstm": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
    "gru": {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
}


@dataclass
class BaselineTrainConfig:
    batch_size: int = TRAIN_BATCH_SIZE
    learning_rate: float = TRAIN_LEARNING_RATE
    weight_decay: float = TRAIN_WEIGHT_DECAY
    max_epochs: int = TRAIN_MAX_EPOCHS
    patience: int = TRAIN_PATIENCE
    seeds: tuple[int, ...] = tuple(TRAIN_SEEDS)
    grad_clip_norm: float | None = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    scheduler_min_lr: float = 1e-5
    transformer_eval_mode: str = "teacher_forcing"


def count_trainable_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def run_single_experiment(
    dataset_dir: str,
    condition_name: str,
    model_name: str,
    seed: int,
    train_config: BaselineTrainConfig,
    model_config: dict,
    evaluate_test: bool = True,
):
    train_path = os.path.join(dataset_dir, condition_name, "train.npz")
    val_path = os.path.join(dataset_dir, condition_name, "val.npz")
    test_path = os.path.join(dataset_dir, condition_name, "test.npz")

    train_arrays = load_split_arrays(train_path)
    input_dim = int(train_arrays.X.shape[2])
    feature_names = [str(x) for x in train_arrays.feature_names.tolist()]

    _, train_loader = build_dataloader(train_path, batch_size=train_config.batch_size, shuffle=True)
    _, val_loader = build_dataloader(val_path, batch_size=train_config.batch_size, shuffle=False)
    test_loader = None
    if evaluate_test:
        _, test_loader = build_dataloader(test_path, batch_size=train_config.batch_size, shuffle=False)

    set_seed(seed)
    model = build_model(model_name, input_dim=input_dim, model_config=model_config)
    parameter_count = count_trainable_parameters(model)
    train_result = train_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        max_epochs=train_config.max_epochs,
        patience=train_config.patience,
        grad_clip_norm=train_config.grad_clip_norm,
        scheduler_factor=train_config.scheduler_factor,
        scheduler_patience=train_config.scheduler_patience,
        scheduler_min_lr=train_config.scheduler_min_lr,
        transformer_eval_mode=train_config.transformer_eval_mode,
    )
    val_eval = evaluate_model(
        model_name,
        train_result["model"],
        val_loader,
        device=train_result["device"],
        transformer_eval_mode=train_config.transformer_eval_mode,
    )
    test_eval = None
    if evaluate_test and test_loader is not None:
        test_eval = evaluate_model(
            model_name,
            train_result["model"],
            test_loader,
            device=train_result["device"],
            transformer_eval_mode=train_config.transformer_eval_mode,
        )

    return {
        "condition": condition_name,
        "model_name": model_name,
        "seed": int(seed),
        "input_dim": input_dim,
        "feature_names": feature_names,
        "parameter_count": parameter_count,
        "device": train_result["device"],
        "model_config": model_config,
        "train_config": {
            "batch_size": int(train_config.batch_size),
            "learning_rate": float(train_config.learning_rate),
            "weight_decay": float(train_config.weight_decay),
            "max_epochs": int(train_config.max_epochs),
            "patience": int(train_config.patience),
            "grad_clip_norm": None if train_config.grad_clip_norm is None else float(train_config.grad_clip_norm),
            "scheduler_factor": float(train_config.scheduler_factor),
            "scheduler_patience": int(train_config.scheduler_patience),
            "scheduler_min_lr": float(train_config.scheduler_min_lr),
            "transformer_eval_mode": str(train_config.transformer_eval_mode),
        },
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_rmse": float(train_result["best_val_rmse"]),
        "val_metrics": val_eval["metrics"],
        "test_metrics": None if test_eval is None else test_eval["metrics"],
        "test_metrics_available": bool(test_eval is not None),
        "test_predictions": None if test_eval is None else {
            "y_true": test_eval["y_true"],
            "y_pred": test_eval["y_pred"],
        },
        "history": train_result["history"],
    }


def run_pooled_experiment(
    dataset_dir: str,
    condition_names: list[str],
    model_name: str,
    seed: int,
    train_config: BaselineTrainConfig,
    model_config: dict,
    evaluate_test: bool = True,
):
    train_paths = [os.path.join(dataset_dir, condition_name, "train.npz") for condition_name in condition_names]
    val_paths = [os.path.join(dataset_dir, condition_name, "val.npz") for condition_name in condition_names]
    test_paths = [os.path.join(dataset_dir, condition_name, "test.npz") for condition_name in condition_names]

    merged_train_arrays = merge_split_arrays(train_paths)
    input_dim = int(merged_train_arrays.X.shape[2])
    feature_names = [str(x) for x in merged_train_arrays.feature_names.tolist()]

    train_dataset, train_loader = build_merged_dataloader(train_paths, batch_size=train_config.batch_size, shuffle=True)
    val_dataset, val_loader = build_merged_dataloader(val_paths, batch_size=train_config.batch_size, shuffle=False)
    test_dataset = None
    test_loader = None
    if evaluate_test:
        test_dataset, test_loader = build_merged_dataloader(test_paths, batch_size=train_config.batch_size, shuffle=False)

    set_seed(seed)
    model = build_model(model_name, input_dim=input_dim, model_config=model_config)
    parameter_count = count_trainable_parameters(model)
    train_result = train_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        max_epochs=train_config.max_epochs,
        patience=train_config.patience,
        grad_clip_norm=train_config.grad_clip_norm,
        scheduler_factor=train_config.scheduler_factor,
        scheduler_patience=train_config.scheduler_patience,
        scheduler_min_lr=train_config.scheduler_min_lr,
        transformer_eval_mode=train_config.transformer_eval_mode,
    )
    pooled_val_eval = evaluate_model(
        model_name,
        train_result["model"],
        val_loader,
        device=train_result["device"],
        transformer_eval_mode=train_config.transformer_eval_mode,
    )
    pooled_test_eval = None
    if evaluate_test and test_loader is not None:
        pooled_test_eval = evaluate_model(
            model_name,
            train_result["model"],
            test_loader,
            device=train_result["device"],
            transformer_eval_mode=train_config.transformer_eval_mode,
        )

    per_condition_metrics = {}
    for condition_name in condition_names:
        val_path = os.path.join(dataset_dir, condition_name, "val.npz")
        _, per_val_loader = build_dataloader(val_path, batch_size=train_config.batch_size, shuffle=False)
        val_eval = evaluate_model(
            model_name,
            train_result["model"],
            per_val_loader,
            device=train_result["device"],
            transformer_eval_mode=train_config.transformer_eval_mode,
        )

        test_eval = None
        if evaluate_test:
            test_path = os.path.join(dataset_dir, condition_name, "test.npz")
            _, per_test_loader = build_dataloader(test_path, batch_size=train_config.batch_size, shuffle=False)
            test_eval = evaluate_model(
                model_name,
                train_result["model"],
                per_test_loader,
                device=train_result["device"],
                transformer_eval_mode=train_config.transformer_eval_mode,
            )

        per_condition_metrics[condition_name] = {
            "val_metrics": val_eval["metrics"],
            "test_metrics": None if test_eval is None else test_eval["metrics"],
        }

    return {
        "conditions": list(condition_names),
        "model_name": model_name,
        "seed": int(seed),
        "input_dim": input_dim,
        "feature_names": feature_names,
        "parameter_count": parameter_count,
        "device": train_result["device"],
        "model_config": model_config,
        "train_config": {
            "batch_size": int(train_config.batch_size),
            "learning_rate": float(train_config.learning_rate),
            "weight_decay": float(train_config.weight_decay),
            "max_epochs": int(train_config.max_epochs),
            "patience": int(train_config.patience),
            "grad_clip_norm": None if train_config.grad_clip_norm is None else float(train_config.grad_clip_norm),
            "scheduler_factor": float(train_config.scheduler_factor),
            "scheduler_patience": int(train_config.scheduler_patience),
            "scheduler_min_lr": float(train_config.scheduler_min_lr),
            "transformer_eval_mode": str(train_config.transformer_eval_mode),
        },
        "pooled_train_sample_count": int(len(train_dataset)),
        "pooled_val_sample_count": int(len(val_dataset)),
        "pooled_test_sample_count": 0 if test_dataset is None else int(len(test_dataset)),
        "best_epoch": int(train_result["best_epoch"]),
        "best_val_rmse": float(train_result["best_val_rmse"]),
        "pooled_val_metrics": pooled_val_eval["metrics"],
        "pooled_test_metrics": None if pooled_test_eval is None else pooled_test_eval["metrics"],
        "per_condition_metrics": per_condition_metrics,
        "history": train_result["history"],
    }


def aggregate_results(run_results):
    grouped = {}
    for result in run_results:
        key = (result["condition"], result["model_name"])
        grouped.setdefault(key, []).append(result)

    summary = {}
    for (condition_name, model_name), results in grouped.items():
        val_mae = [row["val_metrics"]["mae"] for row in results]
        val_rmse = [row["val_metrics"]["rmse"] for row in results]
        val_mape = [row["val_metrics"]["mape"] for row in results]
        val_phm = [row["val_metrics"]["phm_score"] for row in results]
        has_test_metrics = all(row.get("test_metrics_available", True) and row.get("test_metrics") for row in results)
        test_mae = [row["test_metrics"]["mae"] for row in results] if has_test_metrics else []
        test_rmse = [row["test_metrics"]["rmse"] for row in results] if has_test_metrics else []
        test_mape = [row["test_metrics"]["mape"] for row in results] if has_test_metrics else []
        test_phm = [row["test_metrics"]["phm_score"] for row in results] if has_test_metrics else []

        summary.setdefault(condition_name, {})
        summary[condition_name][model_name] = {
            "run_count": len(results),
            "parameter_count": int(results[0]["parameter_count"]),
            "best_val_rmse_mean": round(float(np.mean([row["best_val_rmse"] for row in results])), 6),
            "val_mae_mean": round(float(np.mean(val_mae)), 6),
            "val_mae_std": round(float(np.std(val_mae)), 6),
            "val_rmse_mean": round(float(np.mean(val_rmse)), 6),
            "val_rmse_std": round(float(np.std(val_rmse)), 6),
            "val_mape_mean": round(float(np.mean(val_mape)), 6),
            "val_mape_std": round(float(np.std(val_mape)), 6),
            "val_phm_score_mean": round(float(np.mean(val_phm)), 6),
            "val_phm_score_std": round(float(np.std(val_phm)), 6),
            "test_mae_mean": None if not has_test_metrics else round(float(np.mean(test_mae)), 6),
            "test_mae_std": None if not has_test_metrics else round(float(np.std(test_mae)), 6),
            "test_rmse_mean": None if not has_test_metrics else round(float(np.mean(test_rmse)), 6),
            "test_rmse_std": None if not has_test_metrics else round(float(np.std(test_rmse)), 6),
            "test_mape_mean": None if not has_test_metrics else round(float(np.mean(test_mape)), 6),
            "test_mape_std": None if not has_test_metrics else round(float(np.std(test_mape)), 6),
            "test_phm_score_mean": None if not has_test_metrics else round(float(np.mean(test_phm)), 6),
            "test_phm_score_std": None if not has_test_metrics else round(float(np.std(test_phm)), 6),
        }

    return summary


def save_experiment_outputs(output_dir: str, run_results, summary):
    os.makedirs(output_dir, exist_ok=True)

    runs_path = os.path.join(output_dir, "baseline_runs.json")
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(output_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "runs_path": runs_path,
        "summary_path": summary_path,
    }
