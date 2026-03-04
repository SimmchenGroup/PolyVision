from tensorflow.keras.optimizers import SGD

def fine_tune_model(model, base_model, unfreeze_from="mixed6",
                    learning_rate=1e-4, momentum=0.9):
    """
    Unfreeze layers after a specified layer name and recompile.
    """

    unfreeze = False

    for layer in base_model.layers:
        if unfreeze:
            layer.trainable = True
        if layer.name == unfreeze_from:
            unfreeze = True

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(
            learning_rate=learning_rate,
            momentum=momentum
        ),
        metrics=['acc']
    )

    return model