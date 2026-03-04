# AI Microplastics Detection

This repository contains a modular deep learning framework for detecting and classifying 
microplastic particles using microscopy images. The framework is built on **TensorFlow / 
Keras** and is designed for reproducible experiments, automatic versioning, checkpointing, 
fine-tuning, and layer analysis.

---

## Project Structure
microplastic_ai/
├── init.py
├── config.py # Experiment configuration
├── data.py # Dataset loading and generators
├── model.py # Model building (pretrained InceptionV3 + custom layers)
├── train.py # Training logic with checkpointing and early stopping
├── fine_tune.py # Fine-tuning utilities
├── retrain.py # Continue training from saved models
├── analysis.py # Particle area and prediction confidence analysis
├── layer_analysis.py # Layer usage metrics and visualization
├── visualization.py # Training metrics plotting
├── gradcam.py # Grad-CAM heatmaps
└── pipeline.py # Main experiment pipeline

Top-level runner scripts:
run_experiment.py # Train a new model from scratch with versioning
retrain_model.py # Continue training an existing saved model
analyze_layers.py # Layer usage diagnostics


---

## ⚡ Features

- **Modular architecture**: Clean separation of data, models, training, analysis, and visualization.
- **Pretrained backbone**: InceptionV3 (frozen initially, fine-tunable later).
- **Custom classifier head**: Additional Conv2D, Dense layers, and Dropout.
- **Automatic checkpointing**: Best model saved during training.
- **Early stopping**: Stops training when validation accuracy plateaus.
- **Structured versioning**: Automatically increments experiment versions (`v1.19`, `v1.20`, ...).
- **Layer usage analysis**: Measure neuron/filter activity, detect underused neurons.
- **Grad-CAM**: Visual explanations for predictions.
- **Reproducibility**: Save training history and experiment metadata.

---
