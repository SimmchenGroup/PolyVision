from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers
import tensorflow as tf

def get_model_input_size(model_name):
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

def build_model(config, num_classes, model_name: str = "inception"):
    model_name = (model_name or "inception").strip().lower()

    if model_name in {"inception", "inceptionv3"}:
        input_shape = get_model_input_size(model_name)
        base_model = InceptionV3(
            input_shape=(*config.image_size, 3),
            include_top=False,
            weights=None
        )
        base_model.load_weights(config.weights_path)

        for layer in base_model.layers:
            layer.trainable = False

    elif model_name in {"efficient", "efficientnet", "efficientnetb0", "effb0"}:
        input_shape = get_model_input_size(model_name)
        config.learning_rate = 1e-6
        base_model = EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        x = preprocess_input(base_model.input)

    elif model_name in {"efficientb4", "efficientnetb4", "effb4"}:
        input_shape = get_model_input_size(model_name)
        config.learning_rate = 3e-5
        base_model = EfficientNetB4(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet"
        )

        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        x = preprocess_input(base_model.input)

    elif model_name in {"res", "resnet", "resnet50"}:
        input_shape = get_model_input_size(model_name)
        config.learning_rate = 1e-4
        base_model = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights=r"training\classification\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
        )

        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        x = preprocess_input(base_model.input)

    else:
        raise ValueError(
            f"Unknown model_name={model_name!r}. "
            "Use one of: inception, efficient, res"
        )

    # Common head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)

    # l2 = regularizers.l2(3e-4)

    x = layers.Dense(1024, activation="relu", )(x) #kernel_regularizer=l2
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation="relu", )(x) #kernel_regularizer=l2
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(base_model.input, x)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=RMSprop(learning_rate=config.learning_rate),
        metrics=["accuracy"]
    )

    return model, base_model