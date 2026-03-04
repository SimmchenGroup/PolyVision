import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_classes(config):
    if config.training_type == "microplastic":
        return config.microplastic_classes
    elif config.training_type == "whisky":
        return config.whisky_classes
    else:
        raise ValueError("Unknown training type")

def get_generators(config):
    base_dir = config.base_dirs[config.training_type]
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen