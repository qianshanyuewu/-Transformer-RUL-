"""Unified training utilities for Chapter 3 models."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from torch import nn

from thesis_rebuild.config import MAPE_EPSILON
from thesis_rebuild.modeling.metrics import regression_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward_batch(model_name, model, batch, device):
    x = batch["X"].to(device)
    if model_name in ("transformer", "paper_transformer"):
        decoder_input = batch["decoder_input"].to(device)
        return model(x, decoder_input)
    return model(x)


def _restore_rul(preds, batch, target_mode):
    if target_mode != "life_ratio":
        return preds, batch["y"].detach().cpu().numpy().astype(np.float64)

    total_samples = batch["total_samples"].detach().cpu().numpy().astype(np.float64)
    actual_rul = batch["rul_label"].detach().cpu().numpy().astype(np.float64)
    restored_pred = np.asarray(preds, dtype=np.float64) * np.maximum(total_samples - 1.0, 1.0)
    restored_pred = np.maximum(restored_pred, 0.0)
    return restored_pred, actual_rul


def _run_one_epoch(
    model_name,
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool,
    grad_clip_norm: float | None = None,
    transformer_eval_mode: str = "autoregressive",
):
    model.train(train)
    losses = []
    all_target_labels = []
    all_pred_labels = []
    all_target_rul = []
    all_pred_rul = []
    target_mode = getattr(loader.dataset, "target_mode", "rul")

    for batch in loader:
        y = batch["y"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if model_name in ("transformer", "paper_transformer") and not train:
            decoder_input = batch["decoder_input"].to(device)
            if transformer_eval_mode == "teacher_forcing":
                preds = model(batch["X"].to(device), decoder_input)
            elif transformer_eval_mode == "autoregressive":
                preds = model.autoregressive_predict(
                    batch["X"].to(device),
                    start_tokens=decoder_input[:, :1, :],
                    prediction_steps=max(int(decoder_input.size(1)) - 1, 0),
                )
            else:
                raise ValueError(f"unsupported transformer_eval_mode: {transformer_eval_mode}")
        else:
            preds = _forward_batch(model_name, model, batch, device)
        loss = criterion(preds, y)

        if train:
            loss.backward()
            if grad_clip_norm is not None and float(grad_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        pred_label = preds.detach().cpu().numpy()
        true_label = y.detach().cpu().numpy()
        pred_rul, true_rul = _restore_rul(pred_label, batch, target_mode=target_mode)
        all_target_labels.append(true_label)
        all_pred_labels.append(pred_label)
        all_target_rul.append(true_rul)
        all_pred_rul.append(pred_rul)

    if not losses:
        return {
            "loss": None,
            "metrics": None,
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "target_true": np.array([]),
            "target_pred": np.array([]),
        }

    y_true = np.concatenate(all_target_rul).astype(np.float64)
    y_pred = np.concatenate(all_pred_rul).astype(np.float64)
    target_true = np.concatenate(all_target_labels).astype(np.float64)
    target_pred = np.concatenate(all_pred_labels).astype(np.float64)
    return {
        "loss": float(np.mean(losses)),
        "metrics": regression_metrics(y_true, y_pred, mape_epsilon=MAPE_EPSILON),
        "y_true": y_true,
        "y_pred": y_pred,
        "target_true": target_true,
        "target_pred": target_pred,
    }


def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device=None,
    grad_clip_norm: float | None = 1.0,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    scheduler_min_lr: float = 1e-5,
    transformer_eval_mode: str = "autoregressive",
):
    device = device or choose_device()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        min_lr=float(scheduler_min_lr),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_score = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_result = _run_one_epoch(
            model_name,
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            grad_clip_norm=grad_clip_norm,
            transformer_eval_mode=transformer_eval_mode,
        )
        with torch.no_grad():
            val_result = _run_one_epoch(
                model_name,
                model,
                val_loader,
                optimizer,
                criterion,
                device,
                train=False,
                transformer_eval_mode=transformer_eval_mode,
            )

        if val_result["metrics"] is None:
            raise ValueError("validation loader produced no samples")

        val_rmse = val_result["metrics"]["rmse"]
        scheduler.step(val_rmse)
        improved = val_rmse < best_score
        if improved:
            best_score = val_rmse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": None if train_result["loss"] is None else round(train_result["loss"], 6),
                "train_metrics": train_result["metrics"],
                "val_loss": None if val_result["loss"] is None else round(val_result["loss"], 6),
                "val_metrics": val_result["metrics"],
                "learning_rate": round(float(optimizer.param_groups[0]["lr"]), 8),
            }
        )

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_rmse": round(float(best_score), 6),
        "history": history,
    }


@torch.no_grad()
def evaluate_model(model_name, model, loader, device=None, transformer_eval_mode: str = "autoregressive"):
    device = device or choose_device()
    model = model.to(device)
    result = _run_one_epoch(
        model_name,
        model,
        loader,
        optimizer=None,
        criterion=nn.MSELoss(),
        device=device,
        train=False,
        transformer_eval_mode=transformer_eval_mode,
    )
    return {
        "loss": None if result["loss"] is None else round(float(result["loss"]), 6),
        "metrics": result["metrics"],
        "y_true": result["y_true"].tolist(),
        "y_pred": result["y_pred"].tolist(),
        "target_true": result["target_true"].tolist(),
        "target_pred": result["target_pred"].tolist(),
    }
