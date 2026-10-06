"""Prepare NEW grouped records/vocabulary; never train or overwrite V1/V2 data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingredient_mass.data import build_vocabulary, read_json, write_json, write_jsonl
from ingredient_mass.scan_groups import (conservative_groups, fingerprint, grouped_assignment,
                                        new_output_directory, prepared_records, reviewed_groups,
                                        split_report, verify_audit, verify_sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/scan_group_audit/audit_manifest.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/query2label_100_v2.json")
    parser.add_argument("--review", type=Path, nargs="+", default=[], help="Latest JSON export of each review page")
    parser.add_argument("--provisional", action="store_true", help="Explicit opt-in: group ALL candidates, without claiming confirmation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.provisional and args.review:
        parser.error("Use either --provisional or completed --review exports, not both.")
    if not args.provisional and not args.review:
        parser.error("Reviewed mode requires --review. Use --provisional explicitly for an UNVERIFIED conservative split.")
    audit, config = read_json(args.audit), read_json(args.config)
    verify_audit(audit)
    print("Verifying source metadata, official splits and RGB hashes...", flush=True)
    verify_sources(audit, args.dataset_root.resolve())
    mode = "provisional" if args.provisional else "reviewed"
    decisions = {}
    if args.provisional:
        mapping = conservative_groups(audit)
    else:
        mapping, decisions = reviewed_groups(audit, [read_json(path) for path in args.review])
    assignments = grouped_assignment(audit["nodes"], mapping, float(config["validation_fraction"]), int(config["seed"]))
    records = prepared_records(audit["nodes"], assignments, mapping)
    summary = split_report(records, audit["nodes"], assignments, mapping, audit, mode)
    vocabulary = build_vocabulary(records, int(config["top_k"]))
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("New train has too few ingredient classes for this config.")
    summary.update({"vocabulary_size": len(vocabulary["entries"]), "seed": config["seed"],
                    "validation_fraction_requested_for_groups": config["validation_fraction"],
                    "audit_id": audit["audit_id"], "vocabulary_sha256": fingerprint(vocabulary),
                    "records_sha256": fingerprint(records), "selected_ingredients": [e["name"] for e in vocabulary["entries"]]})
    # Grouping can move previously validated scans into train. Training profiles,
    # class weights, labels and train mean must be regenerated for a NEW run.
    output = new_output_directory(args.output_dir or ROOT / f"data/processed_100_grouped_{mode}",
                                  protected=(args.dataset_root, ROOT / "data/processed", ROOT / "data/processed_100",
                                             args.audit.parent, ROOT / "runs", ROOT / "artifacts"))
    write_jsonl(output / "records.jsonl", records)
    write_json(output / "vocabulary.json", vocabulary)
    write_json(output / "summary.json", summary)
    write_json(output / "scan_groups.json", {"schema_version": 1, "audit_id": audit["audit_id"], "mode": mode,
                                            "group_by_dish": mapping, "review_decisions": decisions,
                                            "source_sha256": audit["source_sha256"]})
    write_json(output / "preparation_config.json", config)
    for split in ("train", "validation", "test"):
        (output / f"{split}_ids.txt").write_text("".join(r["dish_id"] + "\n" for r in records if r["split"] == split), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("mode", "samples", "groups", "group_overlaps", "official_test_preserved",
                                                   "actual_validation_sample_fraction", "train_mean_mass_grams",
                                                   "candidate_links_crossing_new_train_validation", "certified_independence")}, indent=2))
    print(f"Prepared dataset: {output}")
    print("No training was run. Do NOT resume an old checkpoint on this split; use a NEW run directory.")
    if args.provisional:
        print("PROVISIONAL: not visually reviewed, may over-group plates and miss other related scans.")


if __name__ == "__main__":
    main()
