import os
import csv
import copy
import json
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

_EPOCH_LOG_HEADER = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "timestamp", "elapsed_s"]


class CrashLogger:
    """
    Writes a JSON heartbeat file after every batch using an atomic
    write-then-rename so the file is always valid JSON even if the
    process is killed mid-write (GPU crash / TDR / OOM reboot).

    Fields logged every batch:
      - epoch / batch / total_batches
      - running loss and accuracy
      - GPU memory (used / total)
      - system RAM used
      - wall-clock time per batch
      - last_update ISO timestamp

    On clean completion, status changes to "completed".
    On exception, call .crash(exc) to record the error before exit.
    """

    def __init__(self, log_path: str, model_name: str, dataset_path: str, config):
        self.log_path = log_path
        self._tmp_path = log_path + ".tmp"
        self._epoch_log_path = os.path.join(os.path.dirname(log_path), "epoch_log.csv")
        self._t_batch = time.time()
        self._t_start = time.time()
        # Write header if the file doesn't exist yet (new run)
        if not os.path.exists(self._epoch_log_path):
            try:
                with open(self._epoch_log_path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(_EPOCH_LOG_HEADER)
            except Exception:
                pass
        self._state = {
            "status": "starting",
            "model": model_name,
            "dataset": dataset_path,
            "epochs_total": int(config.epochs),
            "batch_size": int(config.batch_size),
            "epoch": 0,
            "batch": 0,
            "total_batches_this_epoch": 0,
            "train_loss": None,
            "train_acc": None,
            "val_loss": None,
            "val_acc": None,
            "gpu_memory_used_gb": None,
            "gpu_memory_total_gb": None,
            "system_ram_used_gb": None,
            "batch_time_s": None,
            "elapsed_s": None,
            "last_update": None,
            "error": None,
        }
        self._write()

    def _gpu_mem(self):
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return round(used, 3), round(total, 1)
        return None, None

    def _ram_gb(self):
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 1e9, 2)
        except ImportError:
            return None

    def _write(self):
        self._state["last_update"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._state["elapsed_s"] = round(time.time() - self._t_start, 1)
        try:
            with open(self._tmp_path, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(self._tmp_path, self.log_path)
        except Exception:
            pass  # never let logging crash training

    def update_batch(self, epoch: int, batch: int, n_batches: int,
                     loss: float, acc: float):
        now = time.time()
        gpu_used, gpu_total = self._gpu_mem()
        self._state.update({
            "status": "training",
            "epoch": epoch,
            "batch": batch,
            "total_batches_this_epoch": n_batches,
            "train_loss": round(loss, 6),
            "train_acc": round(acc, 6),
            "gpu_memory_used_gb": gpu_used,
            "gpu_memory_total_gb": gpu_total,
            "system_ram_used_gb": self._ram_gb(),
            "batch_time_s": round(now - self._t_batch, 3),
        })
        self._t_batch = now
        self._write()

    def update_epoch(self, epoch: int, train_loss: float, train_acc: float,
                     val_loss: float, val_acc: float):
        self._state.update({
            "status": "epoch_complete",
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        })
        self._write()
        # Append one row to the persistent epoch log — survives BSOD since each
        # row is flushed immediately after the epoch completes.
        try:
            with open(self._epoch_log_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch,
                    round(train_loss, 6),
                    round(train_acc, 6),
                    round(val_loss, 6),
                    round(val_acc, 6),
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    round(time.time() - self._t_start, 1),
                ])
        except Exception:
            pass

    def crash(self, exc: Exception):
        self._state["status"] = "CRASHED"
        self._state["error"] = f"{type(exc).__name__}: {exc}"
        self._write()

    def complete(self):
        self._state["status"] = "completed"
        self._write()


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _attach_sync_hooks(model: nn.Module) -> list:
    """
    Register torch.cuda.synchronize() forward AND backward hooks on every Conv2d.

    Used during burn-in to break the first real training batches into short GPU
    segments. Without this, the forward + backward pass submits 94+ kernels each
    as one uninterrupted GPU submission — the same TDR-triggering pattern that
    warmup exists to prevent. Both passes must be covered: the backward through
    all Conv2d gradients is as long as the forward and equally likely to TDR.
    Hooks are removed after burn-in so normal training runs at full speed.
    """
    handles = []
    def _sync_fwd(module, inp, out):
        torch.cuda.synchronize()
    def _sync_bwd(module, grad_in, grad_out):
        torch.cuda.synchronize()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(_sync_fwd))
            handles.append(m.register_full_backward_hook(_sync_bwd))
    return handles


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config,
    version: str | None = None,
    steps_per_epoch: int | None = None,
    history_filename: str = "history.csv",
    optimizer=None,
    burn_in_epochs: int = 0,
    crash_log_path: str | None = None,
    grad_accumulation_steps: int = 4,
    val_steps: int | None = None,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
):
    """
    Train model with best-checkpoint saving and early stopping (patience=5).

    Args:
        optimizer: if None, uses RMSprop with config.learning_rate
        steps_per_epoch: if set, only this many batches are used per epoch
                         (for balanced/infinite-style DataLoaders)
        burn_in_epochs: number of epochs to run with per-Conv2d sync hooks
                        active. Set to 0 (default) when warmup.py has been run
                        first — kernels are already cached so hooks are redundant
                        and the sustained synchronize() load destabilises WSL2.
                        Set to 1 only if training without a prior warmup run.
    """
    device = _get_device()
    model = model.to(device)

    save_path = config.save_path
    if version:
        save_path = os.path.join(save_path, version)
    os.makedirs(save_path, exist_ok=True)

    if optimizer is None:
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
        )

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    print(f"[Loss] CrossEntropyLoss(label_smoothing={label_smoothing}, "
          f"class_weights={'set' if class_weights is not None else 'none'})")

    best_val_acc = -1.0
    best_weights = None
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(save_path, "best_model.pt")

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    log_path = crash_log_path or os.path.join(save_path, "crash_log.json")
    dataset_path = getattr(config, "base_dirs", {}).get(
        getattr(config, "training_type", ""), "unknown"
    )
    clog = CrashLogger(log_path, getattr(config, "model_name", "unknown"), dataset_path, config)
    print(f"[CrashLog] Writing heartbeat to: {log_path}")

    # One-time label sanity check — out-of-bounds labels cause CrossEntropyLoss
    # to read outside shared memory, which shows up as cudaErrorLaunchFailure.
    try:
        num_classes = model.head[-1].out_features
        xb_probe, yb_probe = next(iter(train_loader))
        bad_labels = (yb_probe >= num_classes) | (yb_probe < 0)
        if bad_labels.any():
            raise ValueError(
                f"Label out of bounds: got {yb_probe[bad_labels].tolist()}, "
                f"num_classes={num_classes}"
            )
        print(f"[Sanity] Labels OK — num_classes={num_classes}, "
              f"first batch range [{yb_probe.min()}, {yb_probe.max()}]")
    except (AttributeError, StopIteration):
        pass  # model head doesn't have out_features or loader is empty

    try:
        for epoch in range(1, int(config.epochs) + 1):
            # --- Train ---
            model.train()
            if hasattr(model, "backbone"):
                if getattr(model, "backbone_frozen", False):
                    # Full backbone eval: disables StochasticDepth (random residual drops
                    # in EfficientNet MBConv produce different CUDA kernel configurations
                    # each batch, eventually hitting an unstable kernel on Blackwell sm_120a).
                    model.backbone.eval()
                else:
                    for m in model.backbone.modules():
                        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                            m.eval()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            train_iter = iter(train_loader)
            n_batches = steps_per_epoch if steps_per_epoch else len(train_loader)

            # Burn-in: keep per-Conv2d sync hooks active for all batches of the
            # first burn_in_epochs epochs. Protects every forward+backward pass
            # from TDR on Blackwell/WSL2 — 5-batch burn-in was insufficient because
            # the first unprotected batch (batch 6) triggered cudaErrorUnknown.
            epoch_burn_in = epoch <= burn_in_epochs and device.type == "cuda"
            if epoch_burn_in and epoch == 1:
                print(f"[Train] Epoch {epoch}: burn-in active (per-Conv2d sync hooks on all batches).")
            elif burn_in_epochs > 0 and epoch == burn_in_epochs + 1 and device.type == "cuda":
                print(f"[Train] Epoch {epoch}: burn-in complete — training at full speed.")

            pbar = tqdm(
                range(n_batches),
                desc=f"Epoch {epoch}/{config.epochs} [train]",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch_idx in pbar:
                if epoch_burn_in:
                    # Disable inplace activations before attaching backward hooks.
                    # register_full_backward_hook wraps tensors in a view; subsequent
                    # inplace ops (e.g. EfficientNet's silu_) on those views are
                    # forbidden by autograd and raise RuntimeError.
                    _inplace_off = [m for m in model.modules() if getattr(m, 'inplace', False)]
                    for m in _inplace_off:
                        m.inplace = False
                    _hooks = _attach_sync_hooks(model)

                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    if epoch_burn_in:
                        torch.cuda.synchronize()
                        for h in _hooks:
                            h.remove()
                        for m in _inplace_off:
                            m.inplace = True
                    break
                xb, yb = xb.to(device), yb.to(device)

                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n[WARN] NaN/Inf loss at epoch {epoch} batch {batch_idx} — skipping batch", flush=True)
                    optimizer.zero_grad()
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if epoch_burn_in:
                    torch.cuda.synchronize()
                    for h in _hooks:
                        h.remove()
                    for m in _inplace_off:
                        m.inplace = True

                running_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                running_correct += (preds == yb).sum().item()
                running_total += xb.size(0)

                batch_loss = running_loss / max(running_total, 1)
                batch_acc  = running_correct / max(running_total, 1)
                pbar.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

                clog.update_batch(
                    epoch, batch_idx + 1, n_batches,
                    batch_loss, batch_acc,
                )

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)

            # Drain the CUDA command queue before switching to eval — prevents queue
            # backup on slow PCIe/Thunderbolt eGPU links causing TDR or corrupted state.
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # --- Validate ---
            model.eval()
            val_loss_sum = 0.0
            val_correct = 0
            val_total = 0
            n_val_batches = val_steps if val_steps else len(val_loader)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                val_iter = iter(val_loader)
                for _ in tqdm(range(n_val_batches), desc=f"Epoch {epoch}/{config.epochs} [val]  ",
                              unit="batch", dynamic_ncols=True, leave=False):
                    try:
                        xb, yb = next(val_iter)
                    except StopIteration:
                        break
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    loss = criterion(logits, yb)
                    val_loss_sum += loss.item() * xb.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == yb).sum().item()
                    val_total += xb.size(0)

            val_loss = val_loss_sum / max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            clog.update_epoch(epoch, train_loss, train_acc, val_loss, val_acc)

            print(
                f"Epoch {epoch}/{config.epochs} — "
                f"loss: {train_loss:.4f}  acc: {train_acc:.4f}  "
                f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
            )

            # Flush GPU before checkpoint to avoid TDR timeout on Blackwell/WSL2
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())
                model.cpu()
                torch.save(model, best_model_path)
                model.to(device)
                print(f"  [Checkpoint] val_accuracy improved to {best_val_acc:.4f} — saved {best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [EarlyStopping] No improvement for {patience} epochs. Stopping.")
                    break

            if device.type == "cuda":
                torch.cuda.empty_cache()

    except Exception as exc:
        clog.crash(exc)
        raise

    clog.complete()

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save history CSV
    out_csv = os.path.join(save_path, history_filename or "history.csv")
    keys = list(history.keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *keys])
        for i in range(len(history["loss"])):
            w.writerow([i + 1, *[history[k][i] for k in keys]])
    print(f"[Train] Wrote training history CSV: {out_csv}")

    return history
