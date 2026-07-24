import csv
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _write_history_csv(history: dict, out_csv_path: str | Path) -> None:
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k in history if k != "epoch"]
    with out_csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *keys])
        n = len(history.get(keys[0], [])) if keys else 0
        for i in range(n):
            w.writerow([i + 1, *[history[k][i] for k in keys]])


def _run_one_phase(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    steps_per_epoch: int | None,
    checkpoint_dir: Path | None,
    checkpoint_prefix: str,
    best_json_path: Path | None,
    monitor: str = "val_accuracy",
    mode: str = "max",
    device=None,
) -> dict:
    if device is None:
        device = _get_device()

    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    best_val = -float("inf") if mode == "max" else float("inf")
    best_weights = None
    best_epoch = 0

    def _is_better(new, old):
        return new > old if mode == "max" else new < old

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        train_iter = iter(train_loader)
        n_batches = steps_per_epoch if steps_per_epoch else len(train_loader)
        print(f"  Epoch {epoch}/{epochs} — training ({n_batches} batches)...")
        log_every = max(1, n_batches // 5)

        for batch_idx in range(n_batches):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            run_correct += (logits.argmax(1) == yb).sum().item()
            run_total += xb.size(0)

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_batches:
                running_acc = run_correct / max(run_total, 1)
                running_loss = run_loss / max(run_total, 1)
                print(f"    [{batch_idx + 1}/{n_batches}] loss: {running_loss:.4f}  acc: {running_acc:.4f}")

        train_loss = run_loss / max(run_total, 1)
        train_acc = run_correct / max(run_total, 1)

        print(f"  Epoch {epoch}/{epochs} — validating...")
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_total += xb.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        metric_val = val_acc if monitor == "val_accuracy" else val_loss
        print(
            f"  Epoch {epoch}/{epochs} — loss: {train_loss:.4f}  acc: {train_acc:.4f}"
            f"  val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )

        if _is_better(metric_val, best_val):
            best_val = metric_val
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

            if checkpoint_dir:
                best_path = checkpoint_dir / f"{checkpoint_prefix}_best_model.pt"
                torch.save(model, best_path)
                print(f"  [Checkpoint] {monitor} improved to {best_val:.4f} — saved {best_path}")

                if best_json_path:
                    best_json_path.write_text(
                        json.dumps({
                            "monitor": monitor,
                            "mode": mode,
                            "best_epoch_1_based": best_epoch,
                            "best_value": best_val,
                        }, indent=2),
                        encoding="utf-8",
                    )

        if checkpoint_dir:
            epoch_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
            torch.save(model.state_dict(), epoch_path)

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history


def load_and_retrain(
    model_path,
    train_generator: DataLoader,
    val_generator: DataLoader,
    epochs: int = 25,
    learning_rate: float = 1e-6,
    momentum: float = 0.9,
    steps_per_epoch: int | None = None,
    *,
    fine_tune: bool = False,
    fine_tune_epochs: int = 5,
    fine_tune_learning_rate: float = 1e-7,
    unfreeze_last_n: int = 30,
    checkpoint_dir: str | Path | None = None,
    monitor: str = "val_accuracy",
    mode: str = "max",
):
    device = _get_device()
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict):
        raise TypeError(
            f"Checkpoint at {model_path!r} is a state_dict (dict), not a full model. "
            "Re-run training so best_model.pt is saved as a full model object, "
            "or use build_model() + model.load_state_dict() to reconstruct manually."
        )
    model = loaded.to(device)

    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum,
    )

    print(f"[Retrain] Phase 1: retrain ({epochs} epochs, lr={learning_rate})")
    history1 = _run_one_phase(
        model, optimizer, train_generator, val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="retrain",
        best_json_path=checkpoint_dir / "best_epoch.json" if checkpoint_dir else None,
        monitor=monitor,
        mode=mode,
        device=device,
    )

    if checkpoint_dir:
        _write_history_csv(history1, checkpoint_dir / "retrain_history.csv")

    if not fine_tune:
        return model, history1

    # Phase 2: fine-tune — unfreeze last N backbone params
    backbone = getattr(model, "backbone", model)
    named = list(backbone.named_parameters())
    for _, p in named:
        p.requires_grad = False
    for name, p in named[-unfreeze_last_n:]:
        is_bn = any(k in name for k in ("bn", "batch_norm", "running_mean", "running_var", "num_batches"))
        if is_bn:
            continue
        p.requires_grad = True

    ft_optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=fine_tune_learning_rate,
        momentum=momentum,
    )

    print(f"[Retrain] Phase 2: fine-tune ({fine_tune_epochs} epochs, lr={fine_tune_learning_rate})")
    history2 = _run_one_phase(
        model, ft_optimizer, train_generator, val_generator,
        epochs=fine_tune_epochs,
        steps_per_epoch=steps_per_epoch,
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="finetune",
        best_json_path=checkpoint_dir / "best_epoch_finetune.json" if checkpoint_dir else None,
        monitor=monitor,
        mode=mode,
        device=device,
    )

    if checkpoint_dir:
        _write_history_csv(history2, checkpoint_dir / "finetune_history.csv")

    return model, {"retrain": history1, "fine_tune": history2}
