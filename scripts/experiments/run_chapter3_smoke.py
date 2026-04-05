"""Minimal smoke test for the Chapter 3 training framework."""
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from thesis_rebuild.config import TRAIN_BATCH_SIZE, TRAIN_LEARNING_RATE, TRAIN_PATIENCE, TRAIN_WEIGHT_DECAY
from thesis_rebuild.modeling.data import build_dataloader, load_split_arrays
from thesis_rebuild.modeling.models import build_model
from thesis_rebuild.modeling.trainer import evaluate_model, set_seed, train_model


def run_smoke(condition_name="35Hz12kN", epochs=2, seed=13):
    base_dir = os.path.join(
        CURRENT_DIR,
        "results",
        "chapter3_datasets",
        condition_name,
    )
    train_path = os.path.join(base_dir, "train.npz")
    val_path = os.path.join(base_dir, "val.npz")
    test_path = os.path.join(base_dir, "test.npz")

    train_arrays = load_split_arrays(train_path)
    input_dim = int(train_arrays.X.shape[2])

    _, train_loader = build_dataloader(train_path, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    _, val_loader = build_dataloader(val_path, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
    _, test_loader = build_dataloader(test_path, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    results = {}
    for model_name, model_config in (
        ("transformer", {"d_model": 64, "num_heads": 4, "num_layers": 2, "ffn_dim": 128, "dropout": 0.1}),
        ("lstm", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1}),
        ("gru", {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1}),
    ):
        set_seed(seed)
        model = build_model(model_name, input_dim=input_dim, model_config=model_config)
        train_result = train_model(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=TRAIN_LEARNING_RATE,
            weight_decay=TRAIN_WEIGHT_DECAY,
            max_epochs=epochs,
            patience=min(TRAIN_PATIENCE, epochs),
        )
        val_eval = evaluate_model(model_name, train_result["model"], val_loader, device=train_result["device"])
        test_eval = evaluate_model(model_name, train_result["model"], test_loader, device=train_result["device"])
        results[model_name] = {
            "best_epoch": train_result["best_epoch"],
            "best_val_rmse": train_result["best_val_rmse"],
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
        }
    return results


if __name__ == "__main__":
    smoke_results = run_smoke()
    print("=" * 60)
    print("Desktop thesis_rebuild: Chapter 3 smoke test completed")
    for model_name, info in smoke_results.items():
        print("-" * 60)
        print(model_name)
        print(f"best_epoch: {info['best_epoch']}")
        print(f"best_val_rmse: {info['best_val_rmse']}")
        print(f"val_metrics: {info['val_metrics']}")
        print(f"test_metrics: {info['test_metrics']}")
