"""
Generate final model comparison table → CSV.

Usage:
    # After evaluating all models (results JSONs in outputs/results/):
    python scripts/compare_models.py --output outputs/results/comparison.csv

    # Or evaluate all models on-the-fly:
    python scripts/compare_models.py --checkpoints-dir outputs/checkpoints/ --output outputs/results/comparison.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Compare all models")
    parser.add_argument("--results-dir", default="outputs/results/",
                        help="Directory with per-model JSON results")
    parser.add_argument("--output", default="outputs/results/comparison.csv")
    args = parser.parse_args()

    from evaluation.comparison import ModelComparison, load_results_from_dir

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run evaluate.py for each model first.")
        return

    comparison = load_results_from_dir(str(results_dir))

    if not comparison.records:
        print("No result files found. Run evaluate.py for each model first.")
        print(f"Expected JSON files in: {results_dir}")
        return

    df = comparison.save_csv(args.output)
    comparison.print_table()


if __name__ == "__main__":
    main()
