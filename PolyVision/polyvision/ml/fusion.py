"""
Multi-scale fusion for microplastics classification.
Combines YOLO detection, local particle classifier, and global image classifier.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class ParticleFusionClassifier:
    """
    Combines predictions from three models:
    1. YOLO detection (bbox + initial class)
    2. Local particle classifier (crop-level features)
    3. Global image classifier (whole-image context)
    """

    def __init__(self,
                 weights: Tuple[float, float, float] = (0.3, 0.5, 0.2),
                 num_classes: int = 9,
                 use_meta_model: bool = False,
                 meta_model_path: Optional[str] = None):
        """
        Args:
            weights: (w_yolo, w_local, w_global) for late fusion
            num_classes: Number of plastic types
            use_meta_model: Whether to use trained meta-classifier
            meta_model_path: Path to saved meta-classifier weights
        """
        self.weights = weights
        self.num_classes = num_classes
        self.use_meta_model = use_meta_model

        if use_meta_model:
            if meta_model_path is None:
                raise ValueError("meta_model_path required when use_meta_model=True")
            self.meta_model = self._load_meta_model(meta_model_path)
        else:
            self.meta_model = None

    def predict(self,
                yolo_output: Dict,
                local_output: Dict,
                global_output: Dict,
                bbox_coords: Optional[np.ndarray] = None,
                sample_context: Optional[str] = None) -> Tuple[int, float, np.ndarray]:
        """
        Fuse predictions from all three models.

        Args:
            yolo_output: {'probs': np.array, 'conf': float, 'class_id': int}
            local_output: {'probs': np.array, 'features': np.array (optional)}
            global_output: {'probs': np.array, 'features': np.array (optional)}
            bbox_coords: [x_center, y_center, width, height] normalized (0-1)
            sample_context: 'pure' | 'contaminated' | 'environmental' | None

        Returns:
            final_class: Predicted class ID (int)
            final_confidence: Confidence score (float)
            final_probs: Full probability distribution (np.ndarray)
        """
        yolo_probs = self._ensure_normalized(yolo_output['probs'])
        local_probs = self._ensure_normalized(local_output['probs'])
        global_probs = self._ensure_normalized(global_output['probs'])

        if self.use_meta_model and self.meta_model is not None:
            # Use trained meta-classifier
            final_probs = self._meta_predict(
                yolo_probs,
                local_output.get('features', local_probs),
                global_output.get('features', global_probs),
                bbox_coords
            )
        else:
            # Weighted late fusion
            final_probs = self._late_fusion(
                yolo_probs,
                local_probs,
                global_probs,
                sample_context
            )

        final_class = int(np.argmax(final_probs))
        final_confidence = float(final_probs[final_class])

        return final_class, final_confidence, final_probs

    def _late_fusion(self,
                     yolo_probs: np.ndarray,
                     local_probs: np.ndarray,
                     global_probs: np.ndarray,
                     sample_context: Optional[str] = None) -> np.ndarray:
        """Weighted average fusion with context-aware weights."""

        # Adjust weights based on sample context
        if sample_context == "pure":
            # Trust global context more (sample should be homogeneous)
            weights = (0.2, 0.5, 0.3)
        elif sample_context == "contaminated":
            # Balanced weighting
            weights = (0.3, 0.5, 0.2)
        elif sample_context == "environmental":
            # Trust local classifier more (ignore background)
            weights = (0.2, 0.6, 0.2)
        else:
            # Default weights
            weights = self.weights

        w_yolo, w_local, w_global = weights

        final_probs = (
                w_yolo * yolo_probs +
                w_local * local_probs +
                w_global * global_probs
        )

        # Ensure normalized
        return self._ensure_normalized(final_probs)

    def _meta_predict(self,
                      yolo_probs: np.ndarray,
                      local_feats: np.ndarray,
                      global_feats: np.ndarray,
                      bbox_coords: Optional[np.ndarray]) -> np.ndarray:
        """Use trained meta-classifier for fusion."""

        if bbox_coords is None:
            bbox_coords = np.array([0.5, 0.5, 0.1, 0.1])  # default centered

        # Prepare input tensor
        inputs = [
            torch.from_numpy(yolo_probs).float(),
            torch.from_numpy(local_feats).float(),
            torch.from_numpy(global_feats).float(),
            torch.from_numpy(bbox_coords).float(),
        ]

        # Concatenate and add batch dimension
        x = torch.cat(inputs).unsqueeze(0)

        # Forward pass
        with torch.no_grad():
            logits = self.meta_model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        return probs

    def _ensure_normalized(self, probs: np.ndarray) -> np.ndarray:
        """Ensure probability vector sums to 1.0."""
        total = probs.sum()
        if total > 0:
            return probs / total
        else:
            # Uniform distribution if all zeros
            return np.ones(self.num_classes) / self.num_classes

    def _load_meta_model(self, path: str) -> nn.Module:
        """Load pre-trained meta-classifier."""
        model = MetaClassifier(num_classes=self.num_classes)
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model


class MetaClassifier(nn.Module):
    """
    Neural network that learns optimal fusion of multi-scale predictions.

    Input: Concatenated features from YOLO, local classifier, global classifier, bbox
    Output: Final class logits
    """

    def __init__(self,
                 num_classes: int = 9,
                 yolo_dim: int = 9,
                 local_dim: int = 512,
                 global_dim: int = 256,
                 bbox_dim: int = 4):
        super().__init__()

        self.yolo_dim = yolo_dim
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.bbox_dim = bbox_dim

        total_dim = yolo_dim + local_dim + global_dim + bbox_dim

        self.fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, total_dim) concatenated features

        Returns:
            logits: (batch, num_classes)
        """
        return self.fc(x)


def bayesian_fusion(yolo_probs: np.ndarray,
                    local_probs: np.ndarray,
                    global_probs: np.ndarray,
                    sample_type: str = "pure") -> np.ndarray:
    """
    Bayesian fusion using global context as prior.

    Args:
        sample_type: 'pure' | 'contaminated' | 'environmental'

    Returns:
        posterior: (num_classes,) probability distribution
    """
    num_classes = len(local_probs)

    # Define prior based on sample type
    if sample_type == "pure":
        # Expect single dominant class
        dominant_class = np.argmax(global_probs)
        prior = np.ones(num_classes) * 0.01
        prior[dominant_class] = 0.91  # 91% prior on dominant class

    elif sample_type == "contaminated":
        # Use global probs as soft prior (soften peaks)
        prior = global_probs ** 0.5
        prior /= prior.sum()

    else:  # environmental or unknown
        # Uniform prior
        prior = np.ones(num_classes) / num_classes

    # Combine YOLO and local as likelihood
    likelihood = 0.4 * yolo_probs + 0.6 * local_probs

    # Bayesian update: posterior ∝ likelihood × prior
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return posterior


# Utility function for quick fusion
def fuse_predictions(yolo_probs: np.ndarray,
                     local_probs: np.ndarray,
                     global_probs: np.ndarray,
                     weights: Tuple[float, float, float] = (0.3, 0.5, 0.2)) -> Tuple[int, float]:
    """
    Quick fusion without creating classifier object.

    Returns:
        class_id: int
        confidence: float
    """
    w_yolo, w_local, w_global = weights

    final_probs = w_yolo * yolo_probs + w_local * local_probs + w_global * global_probs
    final_probs /= final_probs.sum()

    class_id = int(np.argmax(final_probs))
    confidence = float(final_probs[class_id])

    return class_id, confidence