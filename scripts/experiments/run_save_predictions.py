"""Re-run experiments with a single seed to save per-sample predictions for RUL figures."""
from __future__ import annotations

import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.config import CONDITIONS
from thesis_rebuild.modeling.experiment import (
    DEFAULT_BASELINE_MODEL_CONFIGS,
    BaselineTrainConfig,
    run_single_experiment,
)

DATASET_DIR = os.path.join(CURRENT_DIR, "results", "chapter3_datasets")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "results", "predictions")
BEST_PARAMS_PATH = os.path.join(
    CURRENT_DIR, "results", "chapter4_optuna", "optuna_newsplit_v1", "best_params.json"
)

SEED = 42
MODELS = ["paper_transformer", "lstm", "gru"]


def run_chapter3_predictions():
    """Run chapter 3 baselines (3 models x 3 conditions x 1 seed) and save predictions."""
    print("=" * 60)
    print("Chapter 3 Baseline Predictions")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, "chapter3")
    os.makedirs(output_dir, exist_ok=True)

    train_config = BaselineTrainConfig(seeds=(SEED,))
    all_predictions = {}

    for condition_name in CONDITIONS.keys():
        all_predictions[condition_name] = {}
        for model_name in MODELS:
            model_config = dict(DEFAULT_BASELINE_MODEL_CONFIGS[model_name])
            result = run_single_experiment(
                dataset_dir=DATASET_DIR,
                condition_name=condition_name,
                model_name=model_name,
                seed=SEED,
                train_config=train_config,
                model_config=model_config,
            )
            preds = result.get("test_predictions")
            if preds:
                all_predictions[condition_name][model_name] = {
                    "y_true": preds["y_true"],
                    "y_pred": preds["y_pred"],
                    "test_rmse": result["test_metrics"]["rmse"],
                }
                print(f"  {condition_name} / {model_name}: test_rmse={result['test_metrics']['rmse']:.2f}")

    output_path = os.path.join(output_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    print(f"Saved: {output_path}")
    return all_predictions


def run_chapter4_predictions():
    """Run Optuna-tuned Transformer (3 conditions x 1 seed) and save predictions."""
    print("=" * 60)
    print("Chapter 4 Optuna Transformer Predictions")
    print("=" * 60)

    output_dir = os.path.join(OUTPUT_DIR, "chapter4")
    os.makedirs(output_dir, exist_ok=True)

    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        best_params = json.load(f)

    model_config = {
        "d_model": best_params["d_model"],
        "num_heads": best_params["num_heads"],
        "num_layers": best_params["num_layers"],
        "ffn_dim": best_params["ffn_dim"],
        "dropout": best_params["dropout"],
    }
    train_config = BaselineTrainConfig(
        seeds=(SEED,),
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
    )

    # If Optuna used a different window_size, use the cached dataset
    optuna_window = best_params.get("window_size", 10)
    if optuna_window != 10:
        optuna_dataset_dir = os.path.join(
            CURRENT_DIR, "results", "chapter4_optuna", "optuna_newsplit_v1",
            "dataset_cache", f"window_{optuna_window}"
        )
    else:
        optuna_dataset_dir = DATASET_DIR

    all_predictions = {}
    for condition_name in CONDITIONS.keys():
        result = run_single_experiment(
            dataset_dir=optuna_dataset_dir,
            condition_name=condition_name,
            model_name="paper_transformer",
            seed=SEED,
            train_config=train_config,
            model_config=model_config,
        )
        preds = result.get("test_predictions")
        if preds:
            all_predictions[condition_name] = {
                "y_true": preds["y_true"],
                "y_pred": preds["y_pred"],
                "test_rmse": result["test_metrics"]["rmse"],
            }
            print(f"  {condition_name}: test_rmse={result['test_metrics']['rmse']:.2f}")

    output_path = os.path.join(output_dir, "predictions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False)
    print(f"Saved: {output_path}")
    return all_predictions


if __name__ == "__main__":
    run_chapter3_predictions()
    run_chapter4_predictions()
    print("\nAll predictions saved.")
