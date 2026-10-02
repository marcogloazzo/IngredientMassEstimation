import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import (  # noqa: E402
    build_records,
    build_vocabulary,
    write_json,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Nutrition5k ingredient and per-ingredient mass labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "nutrition5k_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "default.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    records, summary = build_records(
        args.dataset_root.resolve(),
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    vocabulary = build_vocabulary(records, top_k=int(config["top_k"]))

    output_dir = args.output_dir.resolve()
    write_jsonl(output_dir / "records.jsonl", records)
    write_json(output_dir / "vocabulary.json", vocabulary)
    summary["vocabulary_size"] = len(vocabulary["entries"])
    summary["selected_ingredients"] = [
        entry["name"] for entry in vocabulary["entries"]
    ]
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPrepared records: {output_dir / 'records.jsonl'}")
    print(f"Vocabulary:       {output_dir / 'vocabulary.json'}")


if __name__ == "__main__":
    main()
