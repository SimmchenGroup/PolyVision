"""
Local particle classifier - analyzes individual cropped particles.
Keras implementation (loads .keras models).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
from keras.models import load_model
from keras import Model
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess


class LocalParticleClassifier:
    """
    Classifies individual particle crops.

    This version loads a Keras `.keras` model from disk.

    Notes:
    - `device` is kept for API compatibility but not used (backend decides device).
    - Preprocessing mirrors the previous Torch pipeline:
        resize to 224x224, RGB, ImageNet mean/std normalization.
    """

    def __init__(self, model_path: str, num_classes: int = 9, device: str = "cpu"):
        self.device = device  # kept for compatibility; not used
        self.num_classes = int(num_classes)
        self.model_path = str(model_path)

        p = Path(self.model_path)
        if p.suffix.lower() != ".keras":
            raise ValueError(f"LocalParticleClassifier expects a .keras model, got: {p}")

        self.model = load_model(self.model_path)
        self._feature_model: Model | None = None

        # Mirror the Torch defaults
        self.input_size = (224, 224)

    def predict(self, crop_img: np.ndarray, return_features: bool = False):
        """
        Args:
            crop_img: (H, W) or (H, W, 3) grayscale or RGB image
            return_features: Return feature vector from penultimate layer

        Returns:
            dict with keys:
                'probs': (num_classes,) probability distribution
                'class_id': predicted class
                'confidence': max probability
                'features': (optional) feature vector
        """
        x = self._preprocess(crop_img)  # (1, H, W, 3) float32

        probs = self.model.predict(x, verbose=0)
        probs = np.asarray(probs).squeeze()

        # Support either softmax output (C,) or logits-like; if not normalized, normalize
        if probs.ndim != 1:
            probs = probs.reshape(-1)
        if probs.size != self.num_classes:
            # If the model output doesn't match expected class count, don't guess silently
            raise ValueError(f"Model output size {probs.size} != num_classes {self.num_classes}")

        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or s > 1.5:
            # likely logits; apply softmax
            e = np.exp(probs - np.max(probs))
            probs = e / np.sum(e)

        result = {
            "probs": probs.astype(np.float32),
            "class_id": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
        }

        if return_features:
            feats = self._predict_features(x)
            result["features"] = feats

        return result

    def _predict_features(self, x: np.ndarray) -> np.ndarray:
        """
        Returns a 1D feature vector using the penultimate layer output.
        """
        if self._feature_model is None:
            if len(self.model.layers) < 2:
                raise ValueError("Model has too few layers to extract penultimate features.")
            penultimate = self.model.layers[-2].output
            self._feature_model = Model(inputs=self.model.input, outputs=penultimate)

        feats = self._feature_model.predict(x, verbose=0)
        feats = np.asarray(feats).squeeze()
        return feats.astype(np.float32)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Convert to RGB, resize, preprocess with EfficientNet preprocess_input.
        Output: (1, 224, 224, 3) float32
        """
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected crop_img shape: {img.shape}")

        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)  # keep 0..255
        img = efficientnet_preprocess(img)  # EfficientNetB0 training preprocess
        img = np.expand_dims(img, axis=0)
        return img