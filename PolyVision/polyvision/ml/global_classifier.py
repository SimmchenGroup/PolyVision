"""
Global image classifier - analyzes entire images for sample-level context.
Keras implementation (loads .keras models).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2
from keras.models import load_model
from keras import Model
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

class GlobalImageClassifier:
    """
    Classifies entire images to understand sample context.

    This version loads a Keras `.keras` model from disk.

    Notes:
    - `device` is kept for API compatibility but not used.
    - Preprocessing mirrors the previous Torch pipeline:
        resize to 384x384, RGB, ImageNet mean/std normalization.
    """

    def __init__(self, model_path: str, num_classes: int = 9, device: str = "cpu"):
        self.device = device  # kept for compatibility; not used
        self.num_classes = int(num_classes)
        self.model_path = str(model_path)

        p = Path(self.model_path)
        if p.suffix.lower() != ".keras":
            raise ValueError(f"GlobalImageClassifier expects a .keras model, got: {p}")

        self.model = load_model(self.model_path)
        self._feature_model: Model | None = None

        self.input_size = (224, 224)
        # self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict(self, full_img: np.ndarray, return_features: bool = False):
        """
        Args:
            full_img: (H, W) or (H, W, 3)
            return_features: Return feature vector

        Returns:
            dict with keys:
                'probs': (num_classes,) expected class distribution
                'dominant_class': most likely class in sample
                'purity_score': confidence that sample is pure (0-1)
                'features': (optional) feature vector
        """
        x = self._preprocess(full_img)  # (1, 384, 384, 3)

        probs = self.model.predict(x, verbose=0)
        probs = np.asarray(probs).squeeze()

        if probs.ndim != 1:
            probs = probs.reshape(-1)
        if probs.size != self.num_classes:
            raise ValueError(f"Model output size {probs.size} != num_classes {self.num_classes}")

        s = float(np.sum(probs))
        if not np.isfinite(s) or s <= 0 or s > 1.5:
            e = np.exp(probs - np.max(probs))
            probs = e / np.sum(e)

        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
        max_entropy = float(np.log(self.num_classes))
        purity_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        result = {
            "probs": probs.astype(np.float32),
            "dominant_class": int(np.argmax(probs)),
            "purity_score": float(purity_score),
        }

        if return_features:
            feats = self._predict_features(x)
            result["features"] = feats

        return result

    def _predict_features(self, x: np.ndarray) -> np.ndarray:
        if self._feature_model is None:
            if len(self.model.layers) < 2:
                raise ValueError("Model has too few layers to extract penultimate features.")
            penultimate = self.model.layers[-2].output
            self._feature_model = Model(inputs=self.model.input, outputs=penultimate)

        feats = self._feature_model.predict(x, verbose=0)
        feats = np.asarray(feats).squeeze()
        return feats.astype(np.float32)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unexpected full_img shape: {img.shape}")

        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)  # keep 0..255
        img = efficientnet_preprocess(img)
        img = np.expand_dims(img, axis=0)
        return img