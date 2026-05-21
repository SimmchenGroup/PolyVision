import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

import tensorflow as tf

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def _train_val_dirs(config):
    base_dir = config.base_dirs[config.training_type]

    # Allow an alternate root for TRAIN only (e.g. a manually undersampled dataset copy)
    train_base = getattr(config, "train_base_dir_override", None) or base_dir

    train_dir = os.path.join(train_base, "train")
    val_dir = os.path.join(base_dir, "val")

    print(f"[DATA] base_dir={base_dir}")
    print(f"[DATA] train_base_dir_override={getattr(config, 'train_base_dir_override', None)}")
    print(f"[DATA] train_dir={train_dir}")
    print(f"[DATA] val_dir={val_dir}")

    return train_dir, val_dir


def count_images_in_directory(root_dir: str) -> int:
    """
    Count image files under root_dir/<class_name>/**.*
    Works with local paths and TF-supported filesystems.
    """
    root_dir = str(root_dir)
    if not tf.io.gfile.exists(root_dir):
        return 0

    total = 0
    for dirpath, _dirnames, filenames in tf.io.gfile.walk(root_dir):
        for fn in filenames:
            # fn is just the filename; check extension only
            if fn.lower().endswith(_IMAGE_EXTS):
                total += 1

    return int(total)

def compute_steps_per_epoch(config) -> int:
    train_dir, _val_dir = _train_val_dirs(config)
    n_train = count_images_in_directory(train_dir)
    bs = int(config.batch_size)

    print(f"[steps] train_dir={train_dir}")
    print(f"[steps] n_train={n_train} batch_size={bs}")

    if bs <= 0:
        raise ValueError("config.batch_size must be > 0")
    if n_train <= 0:
        raise ValueError(f"No training images found under: {train_dir}")
    return max(1, n_train // bs)

def get_classes(config):
    if config.training_type == "microplastic":
        return config.microplastic_classes
    elif config.training_type == "whisky":
        return config.whisky_classes
    else:
        raise ValueError("Unknown training type")


def _get_model_io(model_name: str):
    """
    Returns (target_size, preprocess_fn) matching the old ImageDataGenerator behavior.
    """
    model_name = (model_name or "inception").lower().strip()

    if model_name in {"inception", "inceptionv3"}:
        return (299, 299), inception_preprocess
    if model_name in {"efficient", "efficientnet", "efficientb0"}:
        return (224, 224), efficientnet_preprocess
    if model_name in {"efficientb4", "efficientnetb4"}:
        return (380, 380), efficientnet_preprocess
    if model_name in {"res", "resnet", "resnet50"}:
        return (224, 224), efficientnet_preprocess

    raise ValueError(f"Unknown model_name={model_name!r}")


def _make_augmenter():
    """
    GPU-friendly augmentation pipeline (close to your ImageDataGenerator settings).
    Notes:
      - shear_range + fill_mode='reflect' from ImageDataGenerator are not replicated here.
      - start with this; only add more complex transforms if needed.
    """
    return tf.keras.Sequential(
        [
            # tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            # tf.keras.layers.RandomRotation(0.25),      # ~90 degrees (0.25 turns)
            # # tf.keras.layers.RandomTranslation(0.2, 0.2),
            # tf.keras.layers.RandomZoom(0.3),
            # tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomRotation(0),  # ~90 degrees (0.25 turns)
            # tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomZoom(0),
            #tf.keras.layers.RandomBrightness(0),
        ],
        name="augmenter",
    )


def _apply_preprocess_and_aug(ds: tf.data.Dataset, preprocess_fn, augmenter=None, training: bool = True):
    AUTOTUNE = tf.data.AUTOTUNE

    def _map(x, y):
        x = tf.cast(x, tf.float32)
        if training and augmenter is not None:
            x = augmenter(x, training=True)
        x = preprocess_fn(x)
        return x, y

    return ds.map(_map, num_parallel_calls=AUTOTUNE)


# def _make_balanced_train_ds(train_dir: str, image_size: tuple[int, int], batch_size: int, num_classes: int, seed: int):
#     """
#     Balanced sampling (oversamples minority classes) without class weights.
#     Produces an *infinite* dataset: you must pass steps_per_epoch to model.fit.
#     """
#     AUTOTUNE = tf.data.AUTOTUNE
#
#     base = tf.keras.utils.image_dataset_from_directory(
#         train_dir,
#         labels="inferred",
#         label_mode="int",
#         image_size=image_size,
#         batch_size=batch_size,
#         shuffle=True,
#         seed=seed,
#     )
#
#     # Split by class -> sample uniformly across classes
#     base_u = base.unbatch()
#     per_class = []
#     for k in range(int(num_classes)):
#         dsk = base_u.filter(lambda x, y, k=k: tf.equal(y, k)).repeat()
#         per_class.append(dsk)
#
#     balanced_u = tf.data.Dataset.sample_from_datasets(
#         per_class,
#         weights=[1.0 / float(num_classes)] * int(num_classes),
#         seed=seed,
#     )
#
#     return balanced_u.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

def _make_balanced_train_ds(
    train_dir: str,
    image_size: tuple[int, int],
    batch_size: int,
    class_names: list[str],
    seed: int,
):
    """
    Balanced sampling (oversamples minority classes) without class weights.
    Produces an *infinite* dataset: you must pass steps_per_epoch to model.fit.

    Streaming implementation: avoids image_dataset_from_directory() indexing
    huge file lists in Python (can MemoryError on large datasets).
    """
    AUTOTUNE = tf.data.AUTOTUNE
    image_h, image_w = int(image_size[0]), int(image_size[1])

    def _is_image_file(path: tf.Tensor) -> tf.Tensor:
        lower = tf.strings.lower(path)
        ok = tf.constant(False)
        for ext in _IMAGE_EXTS:
            ok = tf.logical_or(ok, tf.strings.ends_with(lower, ext))
        return ok

    def _load_and_resize(path: tf.Tensor, label: tf.Tensor):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, [image_h, image_w], method="bilinear")
        img = tf.cast(img, tf.uint8)
        img.set_shape([image_h, image_w, 3])
        return img, label

    per_class = []
    num_classes = len(class_names)

    for k, cname in enumerate(class_names):
        pattern = os.path.join(train_dir, cname, "*")
        ds = tf.data.Dataset.list_files(pattern, shuffle=True, seed=seed)
        ds = ds.filter(_is_image_file)
        ds = ds.map(lambda p, kk=tf.constant(k, tf.int32): (p, kk), num_parallel_calls=AUTOTUNE)
        ds = ds.map(_load_and_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.repeat()
        per_class.append(ds)

    balanced_u = tf.data.Dataset.sample_from_datasets(
        per_class,
        weights=[1.0 / float(num_classes)] * int(num_classes),
        seed=seed,
    )

    return balanced_u.batch(int(batch_size), drop_remainder=True).prefetch(AUTOTUNE)

def _apply_preprocess_and_aug(ds: tf.data.Dataset, preprocess_fn, augmenter=None, training: bool = True):
    AUTOTUNE = tf.data.AUTOTUNE

    def _map(x, y):
        x = tf.cast(x, tf.float32)  # 0..255
        x01 = x / 255.0
        if training and augmenter is not None:
            x01 = augmenter(x01, training=True)
        x = x01 * 255.0
        x = preprocess_fn(x)
        return x, y

    ds = ds.map(_map, num_parallel_calls=1)
    return ds

def get_tfdata_datasets(
    config,
    model_name=None,
    balance_mode: str = "none",  # "none" | "balanced"
    seed: int = 1337,
):
    """
    tf.data replacement for get_generators().

    balance_mode:
      - "none": keep natural class imbalance (fast + simplest)
      - "balanced": oversample classes uniformly (must set steps_per_epoch)
    """
    model_name = (model_name or getattr(config, "model_name", "inception")).lower().strip()
    target_size, preprocess_fn = _get_model_io(model_name)

    train_dir, val_dir = _train_val_dirs(config)

    # --- DEBUG: verify identical class mapping across splits ---
    try:
        _train_tmp = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            image_size=target_size,
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=int(seed),
        )
        _val_tmp = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="int",
            image_size=target_size,
            batch_size=int(config.batch_size),
            shuffle=False,
        )

        print("[DEBUG] TRAIN class_names:", _train_tmp.class_names)
        print("[DEBUG] VAL   class_names:", _val_tmp.class_names)
        print("[DEBUG] class_names match:", _train_tmp.class_names == _val_tmp.class_names)
    except Exception as e:
        print(f"[DEBUG] class_names check failed: {type(e).__name__}: {e}")

    classes = get_classes(config)
    augmenter = _make_augmenter()

    if balance_mode == "balanced":
        train_ds = _make_balanced_train_ds(
            train_dir=train_dir,
            image_size=target_size,
            batch_size=int(config.batch_size),
            class_names=list(classes),
            seed=int(seed),
        )
    elif balance_mode == "none":
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            labels="inferred",
            label_mode="int",
            image_size=target_size,
            batch_size=int(config.batch_size),
            shuffle=True,
            seed=int(seed),
        )#.prefetch(tf.data.AUTOTUNE)
    else:
        raise ValueError("balance_mode must be 'none' or 'balanced'")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=target_size,
        batch_size=int(config.batch_size),
        shuffle=False,
    )#.prefetch(tf.data.AUTOTUNE)

    train_ds = _apply_preprocess_and_aug(train_ds, preprocess_fn, augmenter=augmenter, training=True)
    val_ds = _apply_preprocess_and_aug(val_ds, preprocess_fn, augmenter=None, training=False)

    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    # python
    try:
        xb, yb = next(iter(train_ds.take(1)))
        print("[DEBUG] train batch stats:", float(tf.reduce_min(xb)), float(tf.reduce_max(xb)),
              float(tf.reduce_mean(xb)))
        xvb, yvb = next(iter(val_ds.take(1)))
        print("[DEBUG] val batch stats  :", float(tf.reduce_min(xvb)), float(tf.reduce_max(xvb)),
              float(tf.reduce_mean(xvb)))
    except Exception as e:
        print(f"[DEBUG] batch stats check failed: {type(e).__name__}: {e}")

    return train_ds, val_ds


def get_generators(config, model_name=None, augment_minority=True):
    """
    Returns train/val generators adapted to the model's input size and preprocessing.
    If model_name is None, uses config.model_name.

    Args:
        augment_minority: If True, applies stronger augmentation to balance classes
    """
    model_name = (model_name or getattr(config, "model_name", "inception")).lower().strip()

    # --- Determine target image size and preprocessing function ---
    if model_name in {"inception", "inceptionv3"}:
        target_size = (299, 299)
        preprocess_fn = inception_preprocess
    elif model_name in {"efficient", "efficientnet", "efficientb0"}:
        target_size = (224, 224)
        preprocess_fn = efficientnet_preprocess
    elif model_name in {"efficientb4", "efficientnetb4"}:
        target_size = (380, 380)
        preprocess_fn = efficientnet_preprocess
    elif model_name in {"res", "resnet", "resnet50"}:
        target_size = (224, 224)
        preprocess_fn = efficientnet_preprocess
    else:
        raise ValueError(f"Unknown model_name={model_name!r}")

    # Use the same train/val dirs as tf.data (respects train_base_dir_override)
    train_dir, val_dir = _train_val_dirs(config)

    # More aggressive augmentation for minority classes
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        # rotation_range=90,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # zoom_range=0.3,
        # horizontal_flip=True,
        # vertical_flip=True,
        # shear_range=0.2,
        # brightness_range=[0.8, 1.2],
        # fill_mode='reflect',
        rotation_range = 0,
        width_shift_range = 0,
        height_shift_range = 0,
        zoom_range = 0,
        horizontal_flip = False,
        vertical_flip = False,
        shear_range = 0,
        # brightness_range = 0,
    )

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode='categorical'
    )

    return train_gen, val_gen

def get_test_generator(config, model_name=None):
    """
    Test generator for metrics like confusion matrix.
    shuffle=False so predictions align with gen.classes / gen.filepaths.
    If model_name is None, uses config.model_name.
    """
    model_name = (model_name or getattr(config, "model_name", "inception")).lower().strip()

    if model_name in {"inception", "inceptionv3"}:
        target_size = (299, 299)
        preprocess_fn = inception_preprocess
    elif model_name in {"efficient", "efficientnet", "efficientb0"}:
        target_size = (224, 224)
        preprocess_fn = efficientnet_preprocess
    elif model_name in {"efficientb4", "efficientnetb4"}:
        target_size = (380, 380)
        preprocess_fn = efficientnet_preprocess
    elif model_name in {"res", "resnet", "resnet50"}:
        target_size = (224, 224)
        preprocess_fn = efficientnet_preprocess
    else:
        raise ValueError(f"Unknown model_name={model_name!r}")

    base_dir = config.base_dirs[config.training_type]
    test_dir = os.path.join(base_dir, "test")

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return test_gen