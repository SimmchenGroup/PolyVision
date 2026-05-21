from tensorflow.keras.optimizers import SGD

def fine_tune_model(
    model,
    base_model,
    unfreeze_from="mixed6",
    learning_rate=1e-4,
    momentum=0.9,
    model_name: str = "inception",
    unfreeze_last_n: int = 20,
):
    """
    Unfreeze layers after a specified layer name and recompile.

    If `unfreeze_from` doesn't exist (e.g. different backbone), fall back to unfreezing
    the last `unfreeze_last_n` layers of the base_model.
    """
    model_name = (model_name or "inception").strip().lower()

    # Pick sensible defaults per backbone (can still be overridden by caller)
    if unfreeze_from == "mixed6":
        if model_name in {"efficient", "efficientnet", "efficientnetb0", "effb0"}:
            unfreeze_from = None
        elif model_name in {"res", "resnet", "resnet50"}:
            unfreeze_from = None

    # Try name-based unfreeze first (only if requested)
    did_unfreeze = False
    if unfreeze_from:
        unfreeze = False
        found = False

        for layer in base_model.layers:
            if unfreeze:
                layer.trainable = True
            if layer.name == unfreeze_from:
                unfreeze = True
                found = True

        did_unfreeze = found

    # Fallback: unfreeze last N layers
    if not did_unfreeze:
        n = max(0, int(unfreeze_last_n))
        if n > 0:
            for layer in base_model.layers[-n:]:
                layer.trainable = True

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=SGD(
            learning_rate=learning_rate,
            momentum=momentum
        ),
        metrics=["accuracy"]
    )

    return model