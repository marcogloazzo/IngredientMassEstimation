"""Synthetic and CLI checks; no torch, downloads, GPU or training."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingredient_mass.scan_groups import (candidate_link, connected_groups, conservative_groups, detect_links,
                                        fingerprint, grouped_assignment, new_output_directory, prepared_records,
                                        reviewed_groups, split_report, verify_audit, verify_sources)
from ingredient_mass.data import build_vocabulary


def node(timestamp, masses, split="train", image=True, cafe="cafe1"):
    identifier = f"dish_{timestamp}"
    ingredients = [{"ingredient_id": key, "name": key, "grams": grams,
                    "calories": grams * 2, "fat": 0., "carbohydrates": grams, "protein": 0.}
                   for key, grams in masses.items()]
    return {"dish_id": identifier, "timestamp": timestamp, "cafeteria": cafe, "official_split": split,
            "image": f"imagery/realsense_overhead/{identifier}/rgb.png" if image else None,
            "total_mass": sum(masses.values()), "ingredients": ingredients,
            "nutrients": {"calories": 2 * sum(masses.values()), "fat": 0., "carbohydrates": sum(masses.values()), "protein": 0.}}


def audit_for(nodes):
    links = detect_links(nodes)
    audit = {"schema_version": 1, "nodes": nodes, "links": links, "groups": connected_groups(nodes, links), "source_sha256": {}}
    audit["audit_id"] = fingerprint(audit)
    return audit


class ScanGroupTests(unittest.TestCase):
    def test_increment_single_start_and_multi_ingredient_recipe(self):
        a = node(100, {"berries": 50})
        b = node(120, {"berries": 50, "melon": 38})
        self.assertIsNotNone(candidate_link(a, b))
        c = node(140, {"berries": 51, "melon": 38, "yogurt": 30, "sugar": 2})
        self.assertIsNotNone(candidate_link(b, c))
        self.assertIsNone(candidate_link(b, node(145, {"berries": 20, "melon": 38})))
        self.assertIsNone(candidate_link(a, node(800, {"berries": 50, "melon": 38})))
        self.assertIsNone(candidate_link(a, node(120, {"berries": 50, "melon": 38}, cafe="cafe2")))

    def test_metadata_bridge_without_rgb(self):
        nodes = [node(100, {"a": 20}), node(500, {"a": 20, "b": 10}, "unassigned", image=False),
                 node(1000, {"a": 20, "b": 10, "c": 30})]
        audit = audit_for(nodes)
        self.assertEqual(len(audit["groups"]), 1)
        self.assertEqual(len(audit["groups"][0]["members"]), 3)

    def test_official_boundary_never_merged_even_via_missing_rgb(self):
        nodes = [node(100, {"a": 20}), node(500, {"a": 20, "b": 10}, "unassigned", image=False),
                 node(1000, {"a": 20, "b": 10, "c": 30}, "test")]
        audit = audit_for(nodes)
        mapping = conservative_groups(audit)
        self.assertNotEqual(mapping[nodes[0]["dish_id"]], mapping[nodes[2]["dish_id"]])
        self.assertTrue(any(link.get("excluded_from_grouping") for link in audit["links"]))

    def test_deterministic_group_split_and_original_test(self):
        nodes = [node(100, {"a": 20}), node(120, {"a": 20, "b": 10}),
                 node(2000, {"c": 40}), node(4000, {"d": 50}), node(6000, {"e": 60}, "test")]
        audit = audit_for(nodes); mapping = conservative_groups(audit)
        assignments = grouped_assignment(nodes, mapping, .3, 42)
        self.assertEqual(assignments, grouped_assignment(nodes[::-1], mapping, .3, 42))
        self.assertEqual(assignments["dish_100"], assignments["dish_120"])
        self.assertEqual(assignments["dish_6000"], "test")
        records = prepared_records(nodes, assignments, mapping)
        report = split_report(records, nodes, assignments, mapping, audit, "provisional")
        self.assertTrue(report["official_test_preserved"])
        self.assertFalse(report["certified_independence"])
        self.assertEqual(report["candidate_links_crossing_new_train_validation"], 0)
        self.assertTrue(all(not overlap for overlap in report["group_overlaps"].values()))

    def test_review_requires_complete_exact_coverage_and_notes(self):
        audit = audit_for([node(100, {"a": 20}), node(120, {"a": 20, "b": 10}), node(2000, {"c": 40})])
        review = {"schema_version": 1, "audit_id": audit["audit_id"], "decisions": {}}
        with self.assertRaisesRegex(ValueError, "Incomplete review"):
            reviewed_groups(audit, [review])
        for group in audit["groups"]:
            review["decisions"][group["group_id"]] = {"status": "reviewed", "note": "Photos inspected", "partitions": [group["members"]]}
        mapping, _ = reviewed_groups(audit, [review])
        self.assertEqual(set(mapping), {n["dish_id"] for n in audit["nodes"]})
        broken = copy.deepcopy(review)
        first = audit["groups"][0]["group_id"]
        broken["decisions"][first]["partitions"] = [["dish_100", "dish_100"]]
        with self.assertRaises(ValueError): reviewed_groups(audit, [broken])
        broken = copy.deepcopy(review); broken["decisions"][first]["note"] = ""
        with self.assertRaises(ValueError): reviewed_groups(audit, [broken])
        broken = copy.deepcopy(review); broken["audit_id"] = "another"
        with self.assertRaises(ValueError): reviewed_groups(audit, [broken])

    def test_review_can_split_false_matches_and_merge_missed_sequences(self):
        nodes = [node(100, {"a": 20}), node(120, {"a": 20, "b": 10}), node(2000, {"c": 40})]
        audit = audit_for(nodes)
        decisions = {g["group_id"]: {"status": "reviewed", "note": "Reviewed", "partitions": [[m] for m in g["members"]]}
                     for g in audit["groups"]}
        review = {"schema_version": 1, "audit_id": audit["audit_id"], "decisions": decisions}
        mapping, _ = reviewed_groups(audit, [review])
        self.assertNotEqual(mapping["dish_100"], mapping["dish_120"])
        review["merges"] = [{"members": ["dish_100", "dish_2000"], "note": "Manual continuation confirmed"}]
        mapping, _ = reviewed_groups(audit, [review])
        self.assertEqual(mapping["dish_100"], mapping["dish_2000"])
        self.assertNotEqual(mapping["dish_100"], mapping["dish_120"])

    def test_manual_merge_cannot_move_test_into_train(self):
        nodes = [node(100, {"a": 20}), node(2000, {"b": 30}), node(4000, {"c": 40}, "test")]
        audit = audit_for(nodes)
        decisions = {g["group_id"]: {"status": "reviewed", "note": "Reviewed", "partitions": [g["members"]]} for g in audit["groups"]}
        review = {"schema_version": 1, "audit_id": audit["audit_id"], "decisions": decisions,
                  "merges": [{"members": ["dish_100", "dish_4000"], "note": "Suspected sequence"}]}
        mapping, _ = reviewed_groups(audit, [review])
        with self.assertRaisesRegex(ValueError, "OFFICIAL"):
            grouped_assignment(nodes, mapping)

    def test_manifest_mutation_and_incomplete_group_map_rejected(self):
        nodes = [node(100, {"a": 20}), node(2000, {"b": 30})]
        audit = audit_for(nodes)
        altered = copy.deepcopy(audit); altered["nodes"][0]["total_mass"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            verify_audit(altered)
        with self.assertRaisesRegex(ValueError, "ALL metadata"):
            grouped_assignment(nodes, {"dish_100": "scan_100"})

    def test_vocabulary_profiles_use_new_train_only(self):
        nodes = [node(100, {"a": 10, "b": 20}), node(2000, {"a": 200, "c": 400}, "test")]
        nodes[1]["ingredients"][0]["calories"] = 99999
        records = prepared_records(nodes, {"dish_100": "train", "dish_2000": "test"}, {"dish_100": "g1", "dish_2000": "g2"})
        vocab = build_vocabulary(records, 3)
        entry = next(e for e in vocab["entries"] if e["key"] == "a")
        self.assertEqual(entry["nutrients_per_gram"]["calories"], 2)
        self.assertFalse(any(e["key"] == "c" for e in vocab["entries"]))

    def test_output_refuses_overwrite_and_protected_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); protected = root / "original"; protected.mkdir()
            with self.assertRaises(ValueError): new_output_directory(protected / "nested", (protected,))
            with self.assertRaises(ValueError): new_output_directory(root, (protected,))
            output = new_output_directory(root / "new", (protected,))
            (output / "keep.txt").write_text("user data")
            with self.assertRaises(FileExistsError): new_output_directory(output)
            self.assertEqual((output / "keep.txt").read_text(), "user data")

    def test_cli_tiny_dataset_provisional_and_reviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp); dataset = base / "dataset"
            (dataset / "metadata").mkdir(parents=True); (dataset / "dish_ids/splits").mkdir(parents=True)
            nodes = [node(100, {"ingr_a": 20}), node(120, {"ingr_a": 20, "ingr_b": 10}),
                     node(2000, {"ingr_c": 30, "ingr_d": 20}), node(4000, {"ingr_e": 50, "ingr_f": 10}),
                     node(6000, {"ingr_a": 40}, "test")]
            lines = []
            for n in nodes:
                rgb = dataset / n["image"]; rgb.parent.mkdir(parents=True)
                Image.new("RGB", (64, 48), (128, 64, 32)).save(rgb)
                fields = [n["dish_id"], n["nutrients"]["calories"], n["total_mass"], 0, n["total_mass"], 0]
                for c in n["ingredients"]:
                    fields.extend([c[k] for k in ("ingredient_id", "name", "grams", "calories", "fat", "carbohydrates", "protein")])
                lines.append(",".join(map(str, fields)))
            (dataset / "metadata/dish_metadata_cafe1.csv").write_text("\n".join(lines))
            (dataset / "dish_ids/dish_ids_cafe1.txt").write_text("\n".join(n["dish_id"] for n in nodes))
            (dataset / "dish_ids/splits/rgb_train_ids.txt").write_text("\n".join(n["dish_id"] for n in nodes if n["official_split"] == "train"))
            (dataset / "dish_ids/splits/rgb_test_ids.txt").write_text("dish_6000\n")
            config = base / "config.json"; config.write_text(json.dumps({"top_k": 3, "seed": 42, "validation_fraction": .3}))
            audit_dir = base / "audit"
            command = [sys.executable, str(ROOT / "scripts/audit_scan_groups.py"), "--dataset-root", str(dataset),
                       "--baseline-records", str(base / "missing/records.jsonl"), "--output-dir", str(audit_dir), "--page-size", "2"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((audit_dir / "index.html").is_file())
            self.assertTrue((audit_dir / "review_001.html").is_file())
            html = (audit_dir / "review_001.html").read_text(encoding="utf-8")
            self.assertIn("split(/\\n+/)", html)
            self.assertNotIn("split(/\n+/)", html)
            self.assertIn(".join('\\n')", html)
            self.assertNotIn(".join('\n')", html)
            audit = json.loads((audit_dir / "audit_manifest.json").read_text())
            verify_sources(audit, dataset)
            prepared = base / "prepared"
            prepare = [sys.executable, str(ROOT / "scripts/prepare_grouped_nutrition5k.py"), "--dataset-root", str(dataset),
                       "--audit", str(audit_dir / "audit_manifest.json"), "--config", str(config), "--output-dir", str(prepared)]
            result = subprocess.run(prepare + ["--provisional"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((prepared / "summary.json").read_text())
            self.assertEqual(summary["samples"]["test"], 1)
            self.assertEqual(summary["candidate_links_crossing_new_train_validation"], 0)
            result = subprocess.run(prepare + ["--provisional"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            review = {"schema_version": 1, "audit_id": audit["audit_id"], "decisions": {
                g["group_id"]: {"status": "reviewed", "partitions": [g["members"]], "note": "Synthetic fixture reviewed"}
                for g in audit["groups"]}}
            review_path = base / "review.json"; review_path.write_text(json.dumps(review))
            reviewed_dir = base / "reviewed"
            result = subprocess.run(prepare[:-2] + ["--output-dir", str(reviewed_dir), "--review", str(review_path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((reviewed_dir / "summary.json").read_text())["mode"], "reviewed")
            Image.new("RGB", (64, 48), (0, 0, 0)).save(dataset / nodes[0]["image"])
            with self.assertRaisesRegex(ValueError, "RGB changed"):
                verify_sources(audit, dataset)

    def test_import_does_not_require_torch(self):
        result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0,'src'); import ingredient_mass.scan_groups; assert 'torch' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lazy_import_preserves_public_model_export(self):
        script = "import sys,types; sys.path.insert(0,'src'); fake=types.ModuleType('ingredient_mass.model'); fake.IngredientMassHeads=type('Heads',(),{}); sys.modules['ingredient_mass.model']=fake; from ingredient_mass import IngredientMassHeads; assert IngredientMassHeads is fake.IngredientMassHeads"
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
