"""Run the bounded Chapter 3 transformer manual search workflow."""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from statistics import mean, pstdev


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.config import CONDITIONS, MANUAL_TRAIN_MAX_EPOCHS, MANUAL_TRAIN_PATIENCE, TRAIN_BATCH_SIZE
from thesis_rebuild.modeling.experiment import BaselineTrainConfig, run_single_experiment
from thesis_rebuild.protocol import (
    MANUAL_SEARCH_GRID,
    MANUAL_SEARCH_PILOT_CONDITION,
    MANUAL_SEARCH_PILOT_SEED,
    MANUAL_SEARCH_TOP_CONFIGS,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run manual transformer search for Chapter 3.")
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter3_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter3_search"),
    )
    parser.add_argument("--tag", default="manual_search_v1")
    parser.add_argument("--epochs", type=int, default=MANUAL_TRAIN_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=MANUAL_TRAIN_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    return parser.parse_args()


def iter_search_configs():
    keys = list(MANUAL_SEARCH_GRID.keys())
    values = [MANUAL_SEARCH_GRID[key] for key in keys]
    for index, combo in enumerate(itertools.product(*values), start=1):
        config = dict(zip(keys, combo))
        model_config = {
            "d_model": int(config["d_model"]),
            "num_heads": int(config["num_heads"]),
            "num_layers": int(config["num_layers"]),
            "ffn_dim": int(config["ffn_dim"]),
            "dropout": float(config["dropout"]),
        }
        train_overrides = {
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
        }
        yield {
            "tag": f"cfg_{index:03d}",
            "model_config": model_config,
            "train_overrides": train_overrides,
        }


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def pilot_sort_key(row):
    result = row["pilot_result"]
    return (
        float(result["val_metrics"]["rmse"]),
        int(result["parameter_count"]),
        row["tag"],
    )


def summarize_validation_runs(rows):
    if not rows:
        return {}

    grouped = {}
    for row in rows:
        grouped.setdefault(row["search_tag"], []).append(row)

    summary = []
    for search_tag, items in grouped.items():
        val_rmses = [item["val_metrics"]["rmse"] for item in items]
        val_maes = [item["val_metrics"]["mae"] for item in items]
        val_scores = [item["val_metrics"]["phm_score"] for item in items]
        model_config = items[0]["model_config"]
        train_config = items[0]["train_config"]
        per_condition = {}
        for item in items:
            per_condition.setdefault(item["condition"], []).append(item["val_metrics"]["rmse"])

        summary.append(
            {
                "search_tag": search_tag,
                "run_count": len(items),
                "parameter_count": int(items[0]["parameter_count"]),
                "model_config": model_config,
                "train_config": train_config,
                "avg_val_rmse": round(float(mean(val_rmses)), 6),
                "std_val_rmse": round(float(pstdev(val_rmses)), 6),
                "avg_val_mae": round(float(mean(val_maes)), 6),
                "avg_val_phm_score": round(float(mean(val_scores)), 6),
                "per_condition_avg_val_rmse": {
                    condition: round(float(mean(values)), 6)
                    for condition, values in per_condition.items()
                },
            }
        )

    summary.sort(
        key=lambda row: (
            float(row["avg_val_rmse"]),
            float(row["std_val_rmse"]),
            int(row["parameter_count"]),
            row["search_tag"],
        )
    )
    return summary


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    pilot_rows = []
    search_candidates = []
    for candidate in iter_search_configs():
        train_config = BaselineTrainConfig(
            batch_size=int(args.batch_size),
            learning_rate=float(candidate["train_overrides"]["learning_rate"]),
            weight_decay=float(candidate["train_overrides"]["weight_decay"]),
            max_epochs=int(args.epochs),
            patience=int(args.patience),
            seeds=(MANUAL_SEARCH_PILOT_SEED,),
        )
        result = run_single_experiment(
            dataset_dir=args.dataset_dir,
            condition_name=MANUAL_SEARCH_PILOT_CONDITION,
            model_name="paper_transformer",
            seed=MANUAL_SEARCH_PILOT_SEED,
            train_config=train_config,
            model_config=candidate["model_config"],
            evaluate_test=False,
        )
        pilot_rows.append(
            {
                "search_tag": candidate["tag"],
                "condition": MANUAL_SEARCH_PILOT_CONDITION,
                "seed": MANUAL_SEARCH_PILOT_SEED,
                "parameter_count": result["parameter_count"],
                "model_config": candidate["model_config"],
                "train_config": result["train_config"],
                "val_metrics": result["val_metrics"],
                "best_epoch": result["best_epoch"],
            }
        )
        search_candidates.append({"tag": candidate["tag"], "pilot_result": result, **candidate})

    search_candidates.sort(key=pilot_sort_key)
    top_candidates = search_candidates[:MANUAL_SEARCH_TOP_CONFIGS]

    formal_rows = []
    for candidate in top_candidates:
        for condition_name in CONDITIONS:
            for seed in (13, 42, 3407):
                train_config = BaselineTrainConfig(
                    batch_size=int(args.batch_size),
                    learning_rate=float(candidate["train_overrides"]["learning_rate"]),
                    weight_decay=float(candidate["train_overrides"]["weight_decay"]),
                    max_epochs=int(args.epochs),
                    patience=int(args.patience),
                    seeds=(seed,),
                )
                result = run_single_experiment(
                    dataset_dir=args.dataset_dir,
                    condition_name=condition_name,
                    model_name="paper_transformer",
                    seed=seed,
                    train_config=train_config,
                    model_config=candidate["model_config"],
                    evaluate_test=False,
                )
                formal_rows.append(
                    {
                        "search_tag": candidate["tag"],
                        "condition": condition_name,
                        "seed": seed,
                        "parameter_count": result["parameter_count"],
                        "model_config": candidate["model_config"],
                        "train_config": result["train_config"],
                        "val_metrics": result["val_metrics"],
                        "best_epoch": result["best_epoch"],
                    }
                )

    formal_summary = summarize_validation_runs(formal_rows)
    selected_config = formal_summary[0] if formal_summary else None

    payload = {
        "dataset_dir": args.dataset_dir,
        "pilot_condition": MANUAL_SEARCH_PILOT_CONDITION,
        "pilot_seed": MANUAL_SEARCH_PILOT_SEED,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "grid": MANUAL_SEARCH_GRID,
        "pilot_runs": pilot_rows,
        "pilot_top_candidates": [
            {
                "search_tag": candidate["tag"],
                "parameter_count": int(candidate["pilot_result"]["parameter_count"]),
                "model_config": candidate["model_config"],
                "train_overrides": candidate["train_overrides"],
                "val_metrics": candidate["pilot_result"]["val_metrics"],
                "best_epoch": int(candidate["pilot_result"]["best_epoch"]),
            }
            for candidate in top_candidates
        ],
        "formal_validation_runs": formal_rows,
        "formal_validation_summary": formal_summary,
        "selected_config": selected_config,
    }
    save_json(os.path.join(output_dir, "manual_search_report.json"), payload)
    if selected_config is not None:
        save_json(
            os.path.join(output_dir, "selected_transformer_config.json"),
            {
                "search_tag": selected_config["search_tag"],
                "parameter_count": selected_config["parameter_count"],
                "model_config": selected_config["model_config"],
                "train_config": selected_config["train_config"],
                "selection_rule": "min(avg_val_rmse) -> min(std_val_rmse) -> min(parameter_count)",
            },
        )

    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 3 transformer search completed")
    print(f"output_dir: {output_dir}")
    print("top pilot candidates:")
    for row in payload["pilot_top_candidates"]:
        print(
            f"  {row['search_tag']}: "
            f"val_rmse={row['val_metrics']['rmse']} "
            f"params={row['parameter_count']} "
            f"model_config={row['model_config']} "
            f"train_overrides={row['train_overrides']}"
        )
    if selected_config is not None:
        print("-" * 60)
        print("selected_config")
        print(json.dumps(selected_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
