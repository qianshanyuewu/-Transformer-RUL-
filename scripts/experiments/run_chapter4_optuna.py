"""Run Chapter 4 Optuna tuning for the paper transformer only."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.config import (
    AUTOTUNE_RANDOM_SEED,
    AUTOTUNE_TRIAL_COUNT,
    CONDITIONS,
    MANUAL_TRAIN_MAX_EPOCHS,
    MANUAL_TRAIN_PATIENCE,
    TRAIN_BATCH_SIZE,
    TRAIN_SEEDS,
)
from thesis_rebuild.modeling.dataset_builder import build_chapter3_datasets
from thesis_rebuild.modeling.experiment import (
    BaselineTrainConfig,
    aggregate_results,
    run_single_experiment,
    save_experiment_outputs,
)
from thesis_rebuild.protocol import AUTOTUNE_SEARCH_SPACE, PROTOCOL_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="Run Chapter 4 Optuna tuning.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter3_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter4_optuna"),
    )
    parser.add_argument("--trials", type=int, default=AUTOTUNE_TRIAL_COUNT)
    parser.add_argument("--seed", type=int, default=AUTOTUNE_RANDOM_SEED)
    parser.add_argument("--tag", default="fixed12_optuna_v1_t30_s42")
    parser.add_argument("--epochs", type=int, default=MANUAL_TRAIN_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=MANUAL_TRAIN_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument(
        "--baseline-summary",
        default=os.path.join(
            CURRENT_DIR,
            "results",
            "chapter3_baseline",
            "fixed12_final_v1_e20_p5_s3",
            "baseline_summary.json",
        ),
    )
    return parser.parse_args()


def ensure_dataset_cache(cache_root: str, window_size: int):
    dataset_dir = os.path.join(cache_root, f"window_{window_size}")
    manifest_path = os.path.join(dataset_dir, "chapter3_dataset_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return dataset_dir, json.load(f)

    manifest, _ = build_chapter3_datasets(
        config={
            "protocol": PROTOCOL_NAME,
            "window_size": int(window_size),
        },
        output_dir=dataset_dir,
    )
    return dataset_dir, manifest


def trial_to_record(trial: optuna.trial.FrozenTrial):
    return {
        "number": int(trial.number),
        "state": str(trial.state),
        "value": None if trial.value is None else round(float(trial.value), 6),
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def plot_trial_history(trials, output_path):
    numbers = [trial["number"] for trial in trials if trial["value"] is not None]
    values = [trial["value"] for trial in trials if trial["value"] is not None]
    if not numbers:
        return None

    best_so_far = []
    current_best = math.inf
    for value in values:
        current_best = min(current_best, value)
        best_so_far.append(current_best)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(numbers, values, color="#9a9a9a", linewidth=1.0, marker="o", markersize=3, label="trial value")
    ax.plot(numbers, best_so_far, color="#000000", linewidth=1.2, label="best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Average validation RMSE")
    ax.set_title("Optuna Objective History", fontsize=11)
    ax.grid(alpha=0.2, linewidth=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_baseline_vs_optuna(baseline_summary, tuned_summary, output_path):
    conditions = list(CONDITIONS.keys())
    baseline_values = []
    tuned_values = []
    for condition_name in conditions:
        baseline_values.append(baseline_summary[condition_name]["paper_transformer"]["test_rmse_mean"])
        tuned_values.append(tuned_summary[condition_name]["paper_transformer"]["test_rmse_mean"])

    x = range(len(conditions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar([idx - width / 2 for idx in x], baseline_values, width=width, color="#bdbdbd", edgecolor="black", label="Chapter 3 manual")
    ax.bar([idx + width / 2 for idx in x], tuned_values, width=width, color="#4d4d4d", edgecolor="black", label="Chapter 4 Optuna")
    ax.set_xticks(list(x))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Manual vs Optuna Transformer Results", fontsize=11)
    ax.grid(axis="y", alpha=0.2, linewidth=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)
    dataset_cache_root = os.path.join(output_dir, "dataset_cache")
    os.makedirs(dataset_cache_root, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=int(args.seed))
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        window_size = trial.suggest_categorical("window_size", AUTOTUNE_SEARCH_SPACE["window_size"])
        d_model = trial.suggest_categorical("d_model", AUTOTUNE_SEARCH_SPACE["d_model"])
        num_heads = trial.suggest_categorical("num_heads", AUTOTUNE_SEARCH_SPACE["num_heads"])
        if d_model % num_heads != 0:
            raise optuna.TrialPruned("d_model must be divisible by num_heads")
        num_layers = trial.suggest_categorical("num_layers", AUTOTUNE_SEARCH_SPACE["num_layers"])
        ffn_dim = trial.suggest_categorical("ffn_dim", AUTOTUNE_SEARCH_SPACE["ffn_dim"])
        dropout = trial.suggest_float("dropout", *AUTOTUNE_SEARCH_SPACE["dropout_range"])
        learning_rate = trial.suggest_float("learning_rate", *AUTOTUNE_SEARCH_SPACE["learning_rate_range"], log=True)
        weight_decay = trial.suggest_float("weight_decay", *AUTOTUNE_SEARCH_SPACE["weight_decay_range"], log=True)

        dataset_dir, manifest = ensure_dataset_cache(dataset_cache_root, window_size=window_size)
        train_config = BaselineTrainConfig(
            batch_size=int(args.batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            max_epochs=int(args.epochs),
            patience=int(args.patience),
            seeds=(int(args.seed),),
        )
        model_config = {
            "d_model": int(d_model),
            "num_heads": int(num_heads),
            "num_layers": int(num_layers),
            "ffn_dim": int(ffn_dim),
            "dropout": float(dropout),
        }

        val_rmses = []
        per_condition = {}
        for condition_name in CONDITIONS:
            result = run_single_experiment(
                dataset_dir=dataset_dir,
                condition_name=condition_name,
                model_name="paper_transformer",
                seed=int(args.seed),
                train_config=train_config,
                model_config=model_config,
                evaluate_test=False,
            )
            val_rmses.append(result["val_metrics"]["rmse"])
            per_condition[condition_name] = {
                "val_rmse": result["val_metrics"]["rmse"],
                "val_mae": result["val_metrics"]["mae"],
                "best_epoch": result["best_epoch"],
            }

        avg_val_rmse = float(mean(val_rmses))
        print(
            f"trial={trial.number:02d} "
            f"avg_val_rmse={avg_val_rmse:.6f} "
            f"window_size={window_size} "
            f"d_model={d_model} "
            f"num_layers={num_layers} "
            f"ffn_dim={ffn_dim} "
            f"dropout={dropout:.4f}"
        )
        trial.set_user_attr("model_config", model_config)
        trial.set_user_attr("train_config", train_config.__dict__)
        trial.set_user_attr("dataset_dir", dataset_dir)
        trial.set_user_attr("dataset_manifest", manifest)
        trial.set_user_attr("per_condition", per_condition)
        return avg_val_rmse

    study.optimize(objective, n_trials=int(args.trials))

    best_params = dict(study.best_params)
    best_window_size = int(best_params["window_size"])
    best_dataset_dir, _ = ensure_dataset_cache(dataset_cache_root, window_size=best_window_size)
    final_train_config = BaselineTrainConfig(
        batch_size=int(args.batch_size),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params["weight_decay"]),
        max_epochs=int(args.epochs),
        patience=int(args.patience),
        seeds=tuple(int(seed) for seed in TRAIN_SEEDS),
    )
    best_model_config = {
        "d_model": int(best_params["d_model"]),
        "num_heads": int(best_params["num_heads"]),
        "num_layers": int(best_params["num_layers"]),
        "ffn_dim": int(best_params["ffn_dim"]),
        "dropout": float(best_params["dropout"]),
    }

    final_runs = []
    for condition_name in CONDITIONS:
        for seed in TRAIN_SEEDS:
            result = run_single_experiment(
                dataset_dir=best_dataset_dir,
                condition_name=condition_name,
                model_name="paper_transformer",
                seed=int(seed),
                train_config=final_train_config,
                model_config=best_model_config,
                evaluate_test=True,
            )
            final_runs.append(result)

    final_summary = aggregate_results(final_runs)
    final_eval_dir = os.path.join(output_dir, "final_eval")
    final_paths = save_experiment_outputs(final_eval_dir, final_runs, final_summary)

    baseline_summary = None
    if os.path.exists(args.baseline_summary):
        with open(args.baseline_summary, "r", encoding="utf-8") as f:
            baseline_summary = json.load(f)

    trials_payload = [trial_to_record(trial) for trial in study.trials]
    history_path = plot_trial_history(trials_payload, os.path.join(output_dir, "fig4_1_trial_history.png"))
    comparison_path = None
    if baseline_summary is not None:
        comparison_path = plot_baseline_vs_optuna(
            baseline_summary,
            final_summary,
            os.path.join(output_dir, "fig4_2_baseline_vs_optuna.png"),
        )

    manifest = {
        "protocol": PROTOCOL_NAME,
        "output_dir": output_dir,
        "search_settings": {
            "trials": int(args.trials),
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "search_space": AUTOTUNE_SEARCH_SPACE,
        },
        "best_params": best_params,
        "best_value": round(float(study.best_value), 6),
        "best_trial_number": int(study.best_trial.number),
        "dataset_cache_root": dataset_cache_root,
        "best_dataset_dir": best_dataset_dir,
        "trial_history_path": os.path.join(output_dir, "trial_history.json"),
        "final_eval_dir": final_eval_dir,
        "final_paths": final_paths,
        "figure_paths": {
            "trial_history": history_path,
            "baseline_vs_optuna": comparison_path,
        },
    }

    with open(os.path.join(output_dir, "trial_history.json"), "w", encoding="utf-8") as f:
        json.dump(trials_payload, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "study_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 4 Optuna completed")
    print(f"output_dir: {output_dir}")
    print(f"best_trial: {study.best_trial.number}")
    print(f"best_value: {study.best_value:.6f}")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
