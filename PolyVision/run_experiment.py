import argparse
import sys
from pathlib import Path
from .microplastic_ai.pipeline import run_full_experiment, analyze_only
from .microplastic_ai.config import ExperimentConfig

# Allow running as a script: `python PolyVision/run_experiment.py ...`
# by ensuring the project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        training_type="microplastic",
        learning_rate=0.0001,
        base_dirs={
            "whisky": r"C:\Users\joshk\OneDrive - University of Strathclyde\AI Microplastics Data\datasets\whisky_webs\v1",
            "microplastic": r"C:\Users\joshk\OneDrive - University of Strathclyde\AI Microplastics Data\datasets\v1.14"
        },
        microplastic_classes=[
            'pet', 'rubber', 'pp', 'pvc', 'neoprene', 'nylon', 'pe',
            'pla', 'pmma', 'pu', 'ps'
        ],
        whisky_classes=[
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
            'Q','R','S','T','U','V','W','Y','Z','AA','AB','AC','AD'
        ],
        weights_path=r"C:\Users\joshk\OneDrive\Desktop\AI Microplastics Detection\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
        save_path="results",
        epochs=15
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze-only", action="store_true", help="Skip training and only run analysis on a saved model.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to a saved .keras/.h5 model to analyze.")
    parser.add_argument("--version", type=str, default=None, help="Optional output folder name (e.g. v1.25).")
    args = parser.parse_args()

    config = build_config()

    if args.analyze_only:
        if not args.model_path:
            raise SystemExit("Missing --model-path for --analyze-only")
        analyze_only(config, model_path=args.model_path, version=args.version)
        print("Analysis-only completed successfully!")
        return

    run_full_experiment(config)
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()

# Normal run (train + fine-tune + analysis)
# python PolyVision/run_experiment.py

# Analysis-only run (no training)
# python PolyVision/run_experiment.py --analyze-only --model-path PolyVision/results/v1.25/model.h5

# Run with custom config
# python PolyVision/run_experiment.py --config-path PolyVision/config.py

# Analysis-only but force output folder name
# python PolyVision/run_experiment.py --analyze-only --model-path "C:\path\to\best_model.keras" --version v1.99
