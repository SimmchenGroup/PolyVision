from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD

def load_and_retrain(
    model_path,
    train_generator,
    val_generator,
    epochs=25,
    learning_rate=1e-6,
    momentum=0.9
):
    model = load_model(model_path)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(
            learning_rate=learning_rate,
            momentum=momentum
        ),
        metrics=['acc']
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=2
    )

    return model, history