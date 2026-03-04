from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, Model

def build_model(config, num_classes):
    base_model = InceptionV3(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights=None
    )

    base_model.load_weights(config.weights_path)

    for layer in base_model.layers:
        layer.trainable = False

    last_layer = base_model.get_layer('mixed7')
    x = last_layer.output

    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(base_model.input, x)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(learning_rate=config.learning_rate),
        metrics=['acc']
    )

    return model, base_model