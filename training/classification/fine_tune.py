"""
Fine-tuning stage: unfreeze the top of the (previously frozen) backbone so the
network can adapt to microplastic morphology, while keeping BatchNorm layers frozen
for stability on the modest dataset. Called after head-only pretraining in
pipeline.py. Several arguments are retained only for API compatibility with the
earlier TensorFlow code path and are unused in the PyTorch implementation.
"""
import torch.nn as nn


def fine_tune_model(
    model: nn.Module,
    base_model=None,          # unused, kept for API compatibility
    unfreeze_from="mixed6",   # unused in PyTorch path
    learning_rate: float = 1e-4,
    momentum: float = 0.9,
    model_name: str = "inception",
    unfreeze_last_n: int = 20,
):
    """
    Unfreeze the last N parameter tensors of model.backbone, then return the
    updated model. Caller is responsible for creating a new optimizer that
    captures the newly unfrozen parameters.

    BatchNorm weight/bias params are kept frozen to prevent running-stat drift
    during fine-tuning (standard best practice).
    """
    model_name = (model_name or "inception").strip().lower()

    backbone = getattr(model, "backbone", model)
    named = list(backbone.named_parameters())

    # Freeze everything first
    for _, p in named:
        p.requires_grad = False

    # Unfreeze last N, skipping BN params
    n = max(0, int(unfreeze_last_n))
    for name, p in named[-n:]:
        is_bn = any(k in name for k in ("bn", "batch_norm", "running_mean", "running_var", "num_batches"))
        if is_bn:
            continue
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FineTune] Unfroze last {n} backbone params. Trainable params: {trainable:,}")

    if hasattr(model, 'backbone_frozen'):
        model.backbone_frozen = False

    return model


def make_sgd_optimizer(model: nn.Module, learning_rate: float, momentum: float = 0.9):
    import torch.optim as optim
    return optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        momentum=momentum,
    )
