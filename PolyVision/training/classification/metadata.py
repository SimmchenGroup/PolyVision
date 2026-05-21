# training/classification/metadata.py
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import json
import platform
import sys
import datetime as _dt

def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x

def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def write_run_metadata(
    run_dir: str | Path,
    *,
    config: Any,
    model: Any,
    stage: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Writes run metadata into run_dir.

    Files produced (per stage):
      - <stage>__run_config.json
      - <stage>__run_config.txt
      - <stage>__model_summary.txt
      - <stage>__model_architecture.json
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Model summary as text ---
    summary_lines: list[str] = []
    try:
        model.summary(print_fn=summary_lines.append)
    except Exception as e:
        summary_lines.append(f"[WARN] model.summary() failed: {type(e).__name__}: {e}")
    _write_text(run_dir / f"{stage}__model_summary.txt", "\n".join(summary_lines) + "\n")

    # --- Model architecture (JSON) ---
    arch_json = None
    try:
        arch_json = model.to_json()
        _write_text(run_dir / f"{stage}__model_architecture.json", arch_json)
    except Exception as e:
        _write_text(
            run_dir / f"{stage}__model_architecture.json",
            json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2),
        )

    # --- Optimizer config (if present) ---
    optimizer_cfg = None
    try:
        opt = getattr(model, "optimizer", None)
        if opt is not None and hasattr(opt, "get_config"):
            optimizer_cfg = opt.get_config()
    except Exception:
        optimizer_cfg = None

    # --- Preprocess/augmentation metadata (best-effort) ---
    preprocess_meta: dict[str, Any] = {}
    try:
        from .data import _get_model_io, _make_augmenter  # local import to avoid broader coupling

        model_name = getattr(config, "model_name", None)
        target_size, preprocess_fn = _get_model_io(model_name)
        preprocess_meta["model_name"] = model_name
        preprocess_meta["target_size"] = list(target_size) if isinstance(target_size, tuple) else target_size
        preprocess_meta["preprocess_fn"] = getattr(preprocess_fn, "__name__", str(preprocess_fn))

        aug = _make_augmenter()
        preprocess_meta["augmenter"] = [
            {
                "class": layer.__class__.__name__,
                "name": getattr(layer, "name", None),
                "config": (layer.get_config() if hasattr(layer, "get_config") else None),
            }
            for layer in getattr(aug, "layers", [])
        ]
    except Exception as e:
        preprocess_meta["error"] = f"{type(e).__name__}: {e}"

    # --- Environment metadata ---
    env = {
        "timestamp_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        import tensorflow as tf
        env["tensorflow"] = getattr(tf, "__version__", None)
    except Exception:
        env["tensorflow"] = None

    payload: dict[str, Any] = {
        "stage": stage,
        "config": _jsonable(config),
        "preprocessing": preprocess_meta,
        "optimizer": optimizer_cfg,
        "extra": _jsonable(extra or {}),
        "env": env,
    }

    (run_dir / f"{stage}__run_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Human-readable TXT (easy to skim)
    pretty = []
    pretty.append(f"STAGE: {stage}")
    pretty.append("")
    pretty.append("=== CONFIG ===")
    pretty.append(json.dumps(_jsonable(config), indent=2, ensure_ascii=False))
    pretty.append("")
    pretty.append("=== PREPROCESSING ===")
    pretty.append(json.dumps(preprocess_meta, indent=2, ensure_ascii=False))
    pretty.append("")
    pretty.append("=== OPTIMIZER ===")
    pretty.append(json.dumps(optimizer_cfg, indent=2, ensure_ascii=False))
    pretty.append("")
    pretty.append("=== EXTRA ===")
    pretty.append(json.dumps(_jsonable(extra or {}), indent=2, ensure_ascii=False))
    pretty.append("")
    pretty.append("=== ENV ===")
    pretty.append(json.dumps(env, indent=2, ensure_ascii=False))
    pretty.append("")

    _write_text(run_dir / f"{stage}__run_config.txt", "\n".join(pretty))