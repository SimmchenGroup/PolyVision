import numpy as np
from tensorflow.keras.models import Model

def analyze_layer_usage(model, generator, layer_name, max_batches=10):
    layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(layer_name).output)

    all_activations = []
    batches_processed = 0

    for x_batch, _ in generator:
        activations = layer_model.predict(x_batch, verbose=0)

        if len(activations.shape) == 4:  # (batch, H, W, channels)
            activations = np.mean(activations, axis=(1, 2))  # (batch, channels)

        all_activations.append(activations)
        batches_processed += 1
        if max_batches is not None and batches_processed >= max_batches:
            break

    all_activations = np.concatenate(all_activations, axis=0)

    mean_activation = np.mean(all_activations, axis=0)
    nonzero_fraction = np.mean(all_activations > 1e-5, axis=0)
    top_indices = np.argsort(-mean_activation)

    return {
        'mean_activation': mean_activation,
        'nonzero_fraction': nonzero_fraction,
        'top_indices': top_indices
    }

def summarize_layer_usage(layer_usage, threshold=0.01):
    mean_activations = layer_usage['mean_activation']
    return {
        "near_zero": int((mean_activations < threshold).sum()),
        "min": float(mean_activations.min()),
        "median": float(np.median(mean_activations)),
        "max": float(mean_activations.max())
    }

def analyze_all_layers(model, generator, max_batches=10, visualize_conv=False, verbose=True, print_every=10):
    """
    Analyze all "interesting" layers and return a dict keyed by layer.name.
    Adds progress prints so long runs don't look stuck.
    """
    usage_summary = {}

    eligible_layers = []
    for layer in model.layers:
        if not hasattr(layer, "output"):
            continue
        if layer.__class__.__name__.lower().startswith("input"):
            continue
        eligible_layers.append(layer)

    if verbose:
        mb = "ALL" if max_batches is None else str(max_batches)
        print(f"[LayerAnalysis] Analyzing {len(eligible_layers)} layers (max_batches={mb})...")

    analyzed = 0
    skipped = 0

    for i, layer in enumerate(eligible_layers, start=1):
        lname = layer.name
        if verbose and (i == 1 or i % max(1, int(print_every)) == 0 or i == len(eligible_layers)):
            print(f"[LayerAnalysis] {i}/{len(eligible_layers)}: {lname}")

        try:
            usage = analyze_layer_usage(model, generator, lname, max_batches=max_batches)
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"[LayerAnalysis]   - skipped ({type(e).__name__}): {e}")
            continue

        usage_summary[lname] = usage
        analyzed += 1

    if verbose:
        print(f"[LayerAnalysis] Done. analyzed={analyzed}, skipped={skipped}")

    return usage_summary