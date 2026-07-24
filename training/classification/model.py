"""
Classification model definition — the architecture shared by the Local and Global
classifiers (they are identical networks; only their training inputs differ).

An ImageNet-pretrained backbone (EfficientNet-B0 by default; InceptionV3 / ResNet50
also selectable) has its original classifier replaced by `nn.Identity`, so it acts as
a feature extractor. Its global-average-pooled feature vector feeds a small custom
head — Linear(feat_dim -> 512) -> ReLU -> Linear(512 -> num_classes) — producing class
logits (softmax is applied at inference; training uses CrossEntropyLoss).

Training is two-phase: the backbone is frozen for head-only pretraining, then its top
layers are unfrozen for fine-tuning (see fine_tune.py / pipeline.py). `_freeze_all`
and `_unfreeze_last_n_params` implement that, keeping BatchNorm frozen during
fine-tuning for stability.
"""
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    Inception_V3_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)


def get_model_input_size(model_name):
    """Return the (H, W, C) input size for a backbone name (299 for Inception, 224 for EfficientNet-B0, 380 for B4)."""
    model_name = model_name.lower().strip()
    if model_name in {"inception", "inceptionv3"}:
        return (299, 299, 3)
    elif model_name in {"efficient", "efficientnet", "efficientb0"}:
        return (224, 224, 3)
    elif model_name in {"efficientb4", "efficientnetb4"}:
        return (380, 380, 3)
    elif model_name in {"res", "resnet", "resnet50"}:
        return (224, 224, 3)
    else:
        raise ValueError(f"Unknown model_name={model_name!r}")


class MicroplasticClassifier(nn.Module):
    """
    Pretrained backbone with a custom 3-layer classification head.
    Backbone produces a flat feature vector; head maps it to num_classes logits.
    Use CrossEntropyLoss (no softmax in forward).
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        """Wrap a feature-extractor `backbone` with the custom head Linear(feature_dim->512)->ReLU->Linear(512->num_classes)."""
        super().__init__()
        self.backbone = backbone
        self.backbone_frozen = True  # flipped to False by fine_tune_model() when top layers unfreeze
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features (in no_grad while frozen) -> classification head -> class logits."""
        if self.backbone_frozen:
            with torch.no_grad():
                feats = self.backbone(x)
        else:
            feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        return self.head(feats)


def _freeze_all(module: nn.Module):
    """Freeze every parameter of `module` (requires_grad = False)."""
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_last_n_params(module: nn.Module, n: int, keep_bn_frozen: bool = True):
    """Unfreeze the last n parameter tensors in module, optionally keeping BN frozen."""
    named = list(module.named_parameters())
    for _, p in named:
        p.requires_grad = False
    for name, p in named[-n:]:
        is_bn = any(k in name for k in ("bn", "batch_norm", "running_mean", "running_var", "num_batches"))
        if keep_bn_frozen and is_bn:
            continue
        p.requires_grad = True


def _disable_all_inplace(backbone: nn.Module, name: str):
    """
    Disable inplace ops on Blackwell (sm_120a) where relu_/silu_ PTX kernels crash.

    Two passes:
    1. Module-level: flip inplace=False on any nn.ReLU / nn.SiLU / etc. that expose it.
    2. BasicConv2d patch: InceptionV3 calls F.relu(x, inplace=True) directly in forward()
       which is invisible to pass 1 — monkey-patch those instances.
    """
    # Pass 1: module attributes
    n_mod = 0
    for m in backbone.modules():
        if getattr(m, 'inplace', False):
            m.inplace = False
            n_mod += 1

    # Pass 2: InceptionV3 BasicConv2d functional calls
    n_bconv = 0
    try:
        from torchvision.models.inception import BasicConv2d as _BConv
        def _bconv_fwd(self, x):
            """Non-inplace BasicConv2d forward (conv -> bn -> relu); patched in to avoid in-place ops that break autograd."""
            return F.relu(self.bn(self.conv(x)), inplace=False)
        for m in backbone.modules():
            if isinstance(m, _BConv):
                m.forward = types.MethodType(_bconv_fwd, m)
                n_bconv += 1
    except Exception:
        pass  # not an Inception model or torchvision layout changed

    print(f"[Model] {name}: disabled inplace on {n_mod} modules, "
          f"patched {n_bconv} BasicConv2d forward() calls.")


def build_model(config, num_classes: int, model_name: str = "inception"):
    """
    Build a classification model with a pretrained backbone and custom head.

    Returns (model, None) — the second element kept for API compatibility
    with the old Keras build_model which returned (model, base_model).
    """
    model_name = (model_name or "inception").strip().lower()

    if model_name in {"inception", "inceptionv3"}:
        # aux_logits must stay True when loading pretrained weights; disable after
        backbone = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        backbone.aux_logits = False  # forward now returns plain tensor in all modes
        backbone.AuxLogits = None
        backbone.fc = nn.Identity()
        backbone.transform_input = False  # data.py already normalizes to [-1, 1]
        _disable_all_inplace(backbone, "InceptionV3")
        _freeze_all(backbone)
        feature_dim = 2048
        config.learning_rate = getattr(config, "learning_rate", 1e-5)

    elif model_name in {"efficient", "efficientnet", "efficientnetb0", "effb0"}:
        backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        _disable_all_inplace(backbone, "EfficientNet-B0")
        backbone.classifier = nn.Identity()
        _freeze_all(backbone)
        # Full freeze in pretrain — backbone_frozen=True wraps forward in no_grad() so
        # cuDNN selects inference-mode algorithms. Any requires_grad=True weight in the
        # backbone causes cuDNN to pick training-mode kernels (crashes on Blackwell sm_120a)
        # even inside no_grad(). Top layers unfreeze in fine-tune via fine_tune_model().
        feature_dim = 1280
        config.learning_rate = 1e-6

    elif model_name in {"efficientb4", "efficientnetb4", "effb4"}:
        backbone = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        _disable_all_inplace(backbone, "EfficientNet-B4")
        backbone.classifier = nn.Identity()
        _freeze_all(backbone)
        feature_dim = 1792
        config.learning_rate = 3e-5

    elif model_name in {"res", "resnet", "resnet50"}:
        backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        _disable_all_inplace(backbone, "ResNet50")
        backbone.fc = nn.Identity()
        _freeze_all(backbone)
        # Full freeze — same Blackwell sm_120a reason as EfficientNet above.
        # layer4 unfreeze happens in fine_tune_model().
        feature_dim = 2048
        config.learning_rate = 1e-4

    else:
        raise ValueError(
            f"Unknown model_name={model_name!r}. "
            "Use one of: inception, efficient, efficientb4, res"
        )

    model = MicroplasticClassifier(backbone, feature_dim, num_classes)
    return model, None  # second element kept for API compatibility
