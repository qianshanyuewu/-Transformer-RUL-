"""Run pooled cross-condition Chapter 3 baseline experiments."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.config import CONDITIONS
from thesis_rebuild.modeling.experiment import (
    DEFAULT_BASELINE_MODEL_CONFIGS,
    BaselineTrainConfig,
    run_pooled_experiment,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pooled cross-condition baseline experiments.")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_BASELINE_MODEL_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--dataset-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter3_datasets"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(CURRENT_DIR, "results", "chapter3_pooled_baseline"),
    )
    parser.add_argument("--paper-transformer-search-report", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def _load_transformer_search_selection(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    selected_config = payload.get("selected_config")
    if not selected_config:
        raise ValueError(f"selected_config not found in search report: {report_path}")

    return {
        "report_path": report_path,
        "search_tag": selected_config.get("search_tag"),
        "model_config": dict(selected_config["model_config"]),
        "train_config": dict(selected_config["train_config"]),
    }


def aggregate_pooled_results(run_results):
    summary = {}
    for model_name in {row["model_name"] for row in run_results}:
        model_rows = [row for row in run_results if row["model_name"] == model_name]
        pooled_val_rmse = [row["pooled_val_metrics"]["rmse"] for row in model_rows]
        pooled_test_rmse = [row["pooled_test_metrics"]["rmse"] for row in model_rows if row["pooled_test_metrics"]]
        pooled_val_score = [row["pooled_val_metrics"]["phm_score"] for row in model_rows]
        pooled_test_score = [row["pooled_test_metrics"]["phm_score"] for row in model_rows if row["pooled_test_metrics"]]

        per_condition = {}
        for condition_name in model_rows[0]["conditions"]:
            val_rmse = [row["per_condition_metrics"][condition_name]["val_metrics"]["rmse"] for row in model_rows]
            test_rmse = [
                row["per_condition_metrics"][condition_name]["test_metrics"]["rmse"]
                for row in model_rows
                if row["per_condition_metrics"][condition_name]["test_metrics"] is not None
            ]
            val_score = [row["per_condition_metrics"][condition_name]["val_metrics"]["phm_score"] for row in model_rows]
            test_score = [
                row["per_condition_metrics"][condition_name]["test_metrics"]["phm_score"]
                for row in model_rows
                if row["per_condition_metrics"][condition_name]["test_metrics"] is not None
            ]
            per_condition[condition_name] = {
                "val_rmse_mean": round(float(np.mean(val_rmse)), 6),
                "val_rmse_std": round(float(np.std(val_rmse)), 6),
                "val_phm_score_mean": round(float(np.mean(val_score)), 6),
                "test_rmse_mean": None if not test_rmse else round(float(np.mean(test_rmse)), 6),
                "test_rmse_std": None if not test_rmse else round(float(np.std(test_rmse)), 6),
                "test_phm_score_mean": None if not test_score else round(float(np.mean(test_score)), 6),
            }

        summary[model_name] = {
            "run_count": len(model_rows),
            "parameter_count": int(model_rows[0]["parameter_count"]),
            "pooled_train_sample_count": int(model_rows[0]["pooled_train_sample_count"]),
            "pooled_val_sample_count": int(model_rows[0]["pooled_val_sample_count"]),
            "pooled_test_sample_count": int(model_rows[0]["pooled_test_sample_count"]),
            "pooled_val_rmse_mean": round(float(np.mean(pooled_val_rmse)), 6),
            "pooled_val_rmse_std": round(float(np.std(pooled_val_rmse)), 6),
            "pooled_val_phm_score_mean": round(float(np.mean(pooled_val_score)), 6),
            "pooled_test_rmse_mean": None if not pooled_test_rmse else round(float(np.mean(pooled_test_rmse)), 6),
            "pooled_test_rmse_std": None if not pooled_test_rmse else round(float(np.std(pooled_test_rmse)), 6),
            "pooled_test_phm_score_mean": None if not pooled_test_score else round(float(np.mean(pooled_test_score)), 6),
            "per_condition": per_condition,
        }

    return summary


def main():
    args = parse_args()
    train_config = BaselineTrainConfig()
    if args.seeds is not None:
        train_config.seeds = tuple(args.seeds)
    if args.epochs is not None:
        train_config.max_epochs = int(args.epochs)
    if args.patience is not None:
        train_config.patience = int(args.patience)
    if args.batch_size is not None:
        train_config.batch_size = int(args.batch_size)
    if args.learning_rate is not None:
        train_config.learning_rate = float(args.learning_rate)
    if args.weight_decay is not None:
        train_config.weight_decay = float(args.weight_decay)

    transformer_selection = None
    if args.paper_transformer_search_report:
        transformer_selection = _load_transformer_search_selection(args.paper_transformer_search_report)

    run_tag = args.tag or f"pooled_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(output_dir, exist_ok=True)

    run_results = []
    applied_model_configs = {}
    for model_name in args.models:
        if model_name not in DEFAULT_BASELINE_MODEL_CONFIGS:
            raise ValueError(f"unsupported model_name: {model_name}")
        model_config = dict(DEFAULT_BASELINE_MODEL_CONFIGS[model_name])
        per_model_train_config = BaselineTrainConfig(
            batch_size=int(train_config.batch_size),
            learning_rate=float(train_config.learning_rate),
            weight_decay=float(train_config.weight_decay),
            max_epochs=int(train_config.max_epochs),
            patience=int(train_config.patience),
            seeds=tuple(train_config.seeds),
            grad_clip_norm=train_config.grad_clip_norm,
            scheduler_factor=float(train_config.scheduler_factor),
            scheduler_patience=int(train_config.scheduler_patience),
            scheduler_min_lr=float(train_config.scheduler_min_lr),
            transformer_eval_mode=str(train_config.transformer_eval_mode),
        )
        if model_name == "paper_transformer" and transformer_selection is not None:
            model_config = dict(transformer_selection["model_config"])
            selected_train_config = transformer_selection["train_config"]
            per_model_train_config.learning_rate = float(selected_train_config["learning_rate"])
            per_model_train_config.weight_decay = float(selected_train_config["weight_decay"])

        for seed in train_config.seeds:
            result = run_pooled_experiment(
                dataset_dir=args.dataset_dir,
                condition_names=args.conditions,
                model_name=model_name,
                seed=seed,
                train_config=per_model_train_config,
                model_config=model_config,
            )
            run_results.append(result)
            print("-" * 60)
            print(f"pooled[{','.join(args.conditions)}] | {model_name} | seed={seed}")
            print(f"params: {result['parameter_count']}")
            print(f"best_epoch: {result['best_epoch']}")
            print(f"pooled_val_metrics: {result['pooled_val_metrics']}")
            print(f"pooled_test_metrics: {result['pooled_test_metrics']}")
            for condition_name, condition_metrics in result["per_condition_metrics"].items():
                print(
                    f"  {condition_name}: "
                    f"val_rmse={condition_metrics['val_metrics']['rmse']} "
                    f"test_rmse={None if condition_metrics['test_metrics'] is None else condition_metrics['test_metrics']['rmse']}"
                )
        applied_model_configs[model_name] = model_config

    summary = aggregate_pooled_results(run_results)
    runs_path = os.path.join(output_dir, "pooled_runs.json")
    summary_path = os.path.join(output_dir, "pooled_summary.json")
    with open(runs_path, "w", encoding="utf-8") as f:
        json.dump(run_results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "tag": run_tag,
        "dataset_dir": args.dataset_dir,
        "output_dir": output_dir,
        "conditions": args.conditions,
        "models": args.models,
        "seeds": list(train_config.seeds),
        "paper_transformer_search_report": None if transformer_selection is None else transformer_selection["report_path"],
        "paper_transformer_search_tag": None if transformer_selection is None else transformer_selection["search_tag"],
        "train_config": {
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "weight_decay": train_config.weight_decay,
            "max_epochs": train_config.max_epochs,
            "patience": train_config.patience,
            "grad_clip_norm": train_config.grad_clip_norm,
            "scheduler_factor": train_config.scheduler_factor,
            "scheduler_patience": train_config.scheduler_patience,
            "scheduler_min_lr": train_config.scheduler_min_lr,
            "transformer_eval_mode": train_config.transformer_eval_mode,
        },
        "applied_model_configs": applied_model_configs,
        "paths": {
            "runs_path": runs_path,
            "summary_path": summary_path,
        },
        "summary": summary,
    }
    manifest_path = os.path.join(output_dir, "pooled_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 3 pooled baseline experiments completed")
    print(f"manifest_path: {manifest_path}")
    for model_name, metrics in summary.items():
        print("-" * 60)
        print(model_name)
        print(
            f"pooled_val_rmse_mean={metrics['pooled_val_rmse_mean']}, "
            f"pooled_test_rmse_mean={metrics['pooled_test_rmse_mean']}, "
            f"pooled_test_phm_score_mean={metrics['pooled_test_phm_score_mean']}, "
            f"params={metrics['parameter_count']}"
        )


if __name__ == "__main__":
    main()
