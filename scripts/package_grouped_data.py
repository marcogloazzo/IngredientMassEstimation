"""Package prepared labels only for Drive/Colab. No RGB, weights or training."""
import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingredient_mass.data import read_json, read_jsonl
from ingredient_mass.scan_groups import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed_100_grouped_provisional")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed_100_grouped_provisional.zip")
    args = parser.parse_args()
    source = args.processed_dir.resolve()
    summary = read_json(source / "summary.json")
    records = read_jsonl(source / "records.jsonl")
    if summary.get("mode") != "provisional" or summary.get("official_test_preserved") is not True:
        raise ValueError("Expected grouped provisional dataset with unchanged official test")
    seen = {}
    for record in records:
        group = record["scan_group_id"]
        if group in seen and seen[group] != record["split"]:
            raise ValueError("Group leakage in prepared data")
        seen[group] = record["split"]
    names = ("records.jsonl", "vocabulary.json", "summary.json", "scan_groups.json", "preparation_config.json",
             "train_ids.txt", "validation_ids.txt", "test_ids.txt")
    hashes = {name: file_digest(source / name) for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(source / name, "processed_100_grouped_provisional/" + name)
        archive.writestr("package_manifest.json", json.dumps({"schema_version": 1, "file_sha256": hashes}, indent=2))
    print(f"Upload this ZIP to Drive/v3/data/: {args.output.resolve()}")


if __name__ == "__main__":
    main()
