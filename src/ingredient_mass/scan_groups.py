"""Reconstruct *candidate* incremental scans, review them, then split groups.

No torch, pretrained weights or model inference. Metadata/time compatibility is
not proof of a common physical plate. Missing links remain a limitation even
after reviewing the detected candidates.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import find_rgb_image, load_all_dishes, read_split_ids


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_output_directory(path, protected=()):
    """Never overwrite previous datasets/reports or a protected tree."""
    path = Path(path).resolve()
    for item in protected:
        item = Path(item).resolve()
        if path == item or path in item.parents or item in path.parents:
            raise ValueError(f"Output overlaps protected path: {path} / {item}")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output is not empty: {path}. Choose a NEW directory; no files were deleted.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_inventory(dataset_root):
    """Use full metadata, including scans without RGB, to preserve bridges."""
    root = Path(dataset_root).resolve()
    dishes = load_all_dishes(root / "metadata")
    train, test = read_split_ids(root / "dish_ids/splits")
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError("Official split lists contain duplicated IDs.")
    train, test = set(train), set(test)
    if train & test:
        raise ValueError("Official train/test ID overlap.")
    missing = (train | test) - set(dishes)
    if missing:
        raise ValueError(f"Official IDs missing metadata: {sorted(missing)[:5]}")
    cafe_by_id = {}
    for path in sorted((root / "dish_ids").glob("dish_ids_cafe*.txt")):
        for identifier in path.read_text(encoding="utf-8-sig").split():
            cafe = path.stem.removeprefix("dish_ids_")
            if identifier in cafe_by_id and cafe_by_id[identifier] != cafe:
                raise ValueError(f"Conflicting cafeteria for {identifier}")
            cafe_by_id[identifier] = cafe
    nodes = []
    for identifier, dish in sorted(dishes.items()):
        if not identifier.startswith("dish_") or not identifier[5:].isdigit():
            raise ValueError(f"Invalid timestamp ID: {identifier}")
        cafe = cafe_by_id.get(identifier)
        if cafe is None:
            raise ValueError(f"Missing cafeteria membership for {identifier}; do not infer it from proximity.")
        components = [asdict(item) for item in dish.ingredients]
        values = [dish.total_mass, dish.total_calories, dish.total_fat, dish.total_carbohydrates, dish.total_protein]
        values += [c[key] for c in components for key in ("grams", "calories", "fat", "carbohydrates", "protein")]
        if not all(math.isfinite(v) and v >= 0 for v in values) or dish.total_mass <= 0:
            raise ValueError(f"Invalid masses/nutrients in {identifier}")
        rgb = find_rgb_image(root, identifier)
        nodes.append({"dish_id": identifier, "timestamp": int(identifier[5:]), "cafeteria": cafe,
                      "official_split": "train" if identifier in train else "test" if identifier in test else "unassigned",
                      "image": rgb.relative_to(root).as_posix() if rgb else None,
                      "total_mass": dish.total_mass,
                      "nutrients": {"calories": dish.total_calories, "fat": dish.total_fat,
                                    "carbohydrates": dish.total_carbohydrates, "protein": dish.total_protein},
                      "ingredients": components})
    sources = sorted((root / "metadata").glob("dish_metadata*.csv"))
    sources += sorted((root / "dish_ids").glob("dish_ids_cafe*.txt"))
    sources += sorted((root / "dish_ids/splits").glob("*.txt"))
    return nodes, {p.relative_to(root).as_posix(): file_digest(p) for p in sources}


def ingredient_masses(node):
    masses = defaultdict(float)
    for component in node["ingredients"]:
        if component["grams"] > 0:
            masses[component["ingredient_id"]] += component["grams"]
    return dict(masses)


def candidate_link(before, after, max_gap_seconds=600, gram_tolerance=0.1):
    """Heuristic, including single-item starts and added multi-ingredient recipes.

    Old ingredients must remain (grams may increase), total mass must not drop,
    and the two annotation totals must have a compatible increment. This is
    deliberately broad; identical/repeated recipes need visual disambiguation.
    """
    delta = after["timestamp"] - before["timestamp"]
    if before["cafeteria"] != after["cafeteria"] or not 0 < delta <= max_gap_seconds:
        return None
    left, right = ingredient_masses(before), ingredient_masses(after)
    if not left or not set(left).issubset(right):
        return None
    if any(right[key] < grams - gram_tolerance for key, grams in left.items()):
        return None
    dish_increment = after["total_mass"] - before["total_mass"]
    if dish_increment < -1:
        return None
    component_increment = sum(right.values()) - sum(left.values())
    mass_tolerance = max(2.0, gram_tolerance * len(set(left) | set(right)))
    if abs(dish_increment - component_increment) > mass_tolerance:
        return None
    unchanged = sum(abs(right[key] - grams) <= gram_tolerance for key, grams in left.items())
    added = sorted(set(right) - set(left))
    return {"before": before["dish_id"], "after": after["dish_id"], "seconds": delta,
            "mass_increment": dish_increment, "component_increment": component_increment,
            "shared_classes": len(left), "unchanged_classes": unchanged, "added_classes": added,
            "kind": "increment" if added or component_increment > gram_tolerance else "repeat_candidate",
            "official_boundary": {before["official_split"], after["official_split"]} >= {"train", "test"}}


def detect_links(nodes, max_gap_seconds=600, gram_tolerance=0.1):
    if max_gap_seconds <= 0 or gram_tolerance < 0 or not math.isfinite(gram_tolerance):
        raise ValueError("Invalid grouping tolerances.")
    nodes = sorted(nodes, key=lambda node: (node["cafeteria"], node["timestamp"], node["dish_id"]))
    links = []
    for index, before in enumerate(nodes):
        for after in nodes[index + 1:]:
            if after["cafeteria"] != before["cafeteria"] or after["timestamp"] - before["timestamp"] > max_gap_seconds:
                break
            link = candidate_link(before, after, max_gap_seconds, gram_tolerance)
            if link:
                links.append(link)
    return links


def connected_groups(nodes, links):
    parent = {node["dish_id"]: node["dish_id"] for node in nodes}
    official = {node["dish_id"]: ({node["official_split"]} if node["official_split"] in ("train", "test") else set()) for node in nodes}
    lookup = {node["dish_id"]: node for node in nodes}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    # Official train/test is a hard boundary, not another heuristic. Resolve
    # same-official-partition links first so missing-RGB/unassigned bridges can
    # never force a known train candidate apart or contaminate the test split.
    ordered = sorted(links, key=lambda link: (
        0 if lookup[link["before"]]["official_split"] == lookup[link["after"]]["official_split"] in ("train", "test") else 1,
        -link.get("unchanged_classes", 0), link.get("seconds", 0), link["before"], link["after"]))
    for link in ordered:
        left, right = find(link["before"]), find(link["after"])
        if {"train", "test"} <= official[left] | official[right]:
            link["excluded_from_grouping"] = "official_train_test_boundary"
            continue
        target, merged = min(left, right), max(left, right)
        parent[merged] = target
        official[target] |= official[merged]
    components = defaultdict(list)
    for key in parent:
        components[find(key)].append(key)
    result = []
    for members in sorted(components.values(), key=lambda members: min(members)):
        members.sort(key=lambda key: (lookup[key]["timestamp"], key))
        official = {lookup[key]["official_split"] for key in members}
        result.append({"group_id": "scan_" + members[0][5:], "members": members,
                       "status": "pending_review", "official_conflict": {"train", "test"} <= official,
                       "rgb_members": sum(lookup[key]["image"] is not None for key in members),
                       "used_rgb_members": sum(lookup[key]["image"] is not None and lookup[key]["official_split"] in ("train", "test") for key in members)})
    return result


def image_evidence(root, nodes, links, progress=print):
    """Image differences rank review; they NEVER accept/reject a link."""
    thumbnails = {}
    linked_ids = {link[key] for link in links for key in ("before", "after")}
    for index, node in enumerate(nodes):
        if node["image"]:
            path = Path(root) / node["image"]
            node["image_sha256"] = file_digest(path)
            if node["dish_id"] in linked_ids:
                with Image.open(path) as source:
                    rgb = source.convert("RGB")
                    width, height = rgb.size
                    crop = rgb.crop((int(width * .18), int(height * .12), int(width * .85), int(height * .90)))
                    thumbnails[node["dish_id"]] = np.asarray(crop.resize((32, 32)), dtype=np.float32)
        if (index + 1) % 500 == 0:
            progress(f"Image provenance: {index + 1}/{len(nodes)}", flush=True)
    for link in links:
        if link["before"] in thumbnails and link["after"] in thumbnails:
            difference = np.abs(thumbnails[link["before"]] - thumbnails[link["after"]])
            link["central_rgb_mae_255"] = round(float(difference.mean()), 4)
    return nodes, links


def verify_audit(audit):
    if audit.get("schema_version") != 1:
        raise ValueError("Unsupported audit schema.")
    payload = {key: value for key, value in audit.items() if key != "audit_id"}
    if fingerprint(payload) != audit.get("audit_id"):
        raise ValueError("Audit fingerprint mismatch: modified/incomplete manifest.")
    nodes = {node["dish_id"] for node in audit["nodes"]}
    members = [key for group in audit["groups"] for key in group["members"]]
    if len(nodes) != len(audit["nodes"]) or len(members) != len(set(members)) or set(members) != nodes:
        raise ValueError("Audit groups do not cover metadata exactly once.")


def reviewed_groups(audit, reviews):
    """A reviewer can split a proposed component into multiple real sequences."""
    verify_audit(audit)
    decisions = {}
    merges = []
    known_groups = {g["group_id"]: g for g in audit["groups"]}
    for review in reviews:
        if review.get("schema_version") != 1 or review.get("audit_id") != audit["audit_id"]:
            raise ValueError("Review belongs to another audit; do not mix exports.")
        for key, decision in review.get("decisions", {}).items():
            if key not in known_groups:
                raise ValueError(f"Unknown group in review: {key}")
            if key in decisions and decisions[key] != decision:
                raise ValueError(f"Conflicting review exports for {key}; export the latest decisions only.")
            decisions[key] = decision
        for merge in review.get("merges", []):
            if merge not in merges:
                merges.append(merge)
    group_by_dish = {}
    pending = []
    for group in audit["groups"]:
        decision = decisions.get(group["group_id"])
        if decision is None:
            if group["used_rgb_members"]:
                pending.append(group["group_id"])
            partitions = [group["members"]]
        else:
            if decision.get("status") != "reviewed" or not decision.get("note", "").strip():
                raise ValueError(f"Review needs status=reviewed and an evidence note: {group['group_id']}")
            partitions = decision.get("partitions", [])
            if not isinstance(partitions, list) or not partitions or any(not isinstance(p, list) or not p for p in partitions):
                raise ValueError(f"Invalid partitions for {group['group_id']}")
            flat = [identifier for part in partitions for identifier in part]
            if len(flat) != len(set(flat)) or set(flat) != set(group["members"]):
                raise ValueError(f"Partitions must cover every scan exactly once: {group['group_id']}")
        for part in partitions:
            identifier = "scan_" + min(part)[5:]
            for member in part:
                group_by_dish[member] = identifier
    if pending:
        raise ValueError(f"Incomplete review: {len(pending)} usable-RGB groups pending. First: {pending[:5]}")
    # A visual reviewer may discover continuations missed by the heuristics.
    parents = {group: group for group in group_by_dish.values()}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    for merge in merges:
        members = merge.get("members", [])
        if (not isinstance(members, list) or len(members) < 2 or len(set(members)) != len(members)
                or not set(members) <= set(group_by_dish) or not merge.get("note", "").strip()):
            raise ValueError("A manual merge needs known unique scan IDs and an evidence note.")
        roots = sorted({find(group_by_dish[key]) for key in members})
        for group in roots[1:]:
            parents[group] = roots[0]
    group_by_dish = {key: find(group) for key, group in group_by_dish.items()}
    return group_by_dish, {"groups": decisions, "merges": merges}


def conservative_groups(audit):
    verify_audit(audit)
    return {member: group["group_id"] for group in audit["groups"] for member in group["members"]}


def grouped_assignment(nodes, group_by_dish, validation_fraction=.15, seed=42):
    if not 0 < validation_fraction < .5:
        raise ValueError("Validation fraction must be in (0, .5).")
    if set(group_by_dish) != {node["dish_id"] for node in nodes}:
        raise ValueError("Group map must cover ALL metadata scans, not only scans with RGB.")
    official_by_group = defaultdict(set)
    cafeterias_by_group = defaultdict(set)
    train_groups = set()
    for node in nodes:
        group = group_by_dish[node["dish_id"]]
        official_by_group[group].add(node["official_split"])
        cafeterias_by_group[group].add(node["cafeteria"])
        if node["official_split"] == "train" and node["image"]:
            train_groups.add(group)
    conflicts = [key for key, values in official_by_group.items() if {"train", "test"} <= values]
    if conflicts:
        raise ValueError(f"Groups cross OFFICIAL train/test: {conflicts[:8]}. Review them; test is NEVER moved.")
    if any(len(values) > 1 for values in cafeterias_by_group.values()):
        raise ValueError("A physical-plate group cannot cross cafeterias.")
    if len(train_groups) < 2:
        raise ValueError("At least two usable official-train groups are needed.")
    shuffled = sorted(train_groups)
    random.Random(seed).shuffle(shuffled)
    count = max(1, min(len(shuffled) - 1, round(len(shuffled) * validation_fraction)))
    validation = set(shuffled[:count])
    assignments = {}
    for node in nodes:
        official = node["official_split"]
        assignments[node["dish_id"]] = ("validation" if group_by_dish[node["dish_id"]] in validation else "train") if official == "train" else official
    return assignments


def prepared_records(nodes, assignments, group_by_dish):
    records = []
    for node in sorted(nodes, key=lambda n: n["dish_id"]):
        if node["image"] and assignments[node["dish_id"]] in ("train", "validation", "test"):
            records.append({key: node[key] for key in ("dish_id", "image", "total_mass", "nutrients", "ingredients")} |
                           {"split": assignments[node["dish_id"]], "scan_group_id": group_by_dish[node["dish_id"]]})
    return records


def split_report(records, nodes, assignments, group_by_dish, audit, mode):
    groups = {split: {r["scan_group_id"] for r in records if r["split"] == split} for split in ("train", "validation", "test")}
    overlaps = {f"{a}/{b}": sorted(groups[a] & groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlaps.values()):
        raise ValueError("Group leakage after assignment.")
    expected_test = {node["dish_id"] for node in nodes if node["image"] and node["official_split"] == "test"}
    if {r["dish_id"] for r in records if r["split"] == "test"} != expected_test:
        raise ValueError("Official usable test changed.")
    residual = [link for link in audit["links"] if {assignments[link['before']], assignments[link['after']]} == {"train", "validation"}]
    counts = Counter(r["split"] for r in records)
    support = {split: Counter() for split in groups}
    mass = {}
    for split in groups:
        selected = [r for r in records if r["split"] == split]
        for record in selected:
            support[split].update(set(ingredient_masses(record)))
        values = np.asarray([r["total_mass"] for r in selected])
        mass[split] = {"mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max())} if len(values) else {}
    ingredient_support = [{"ingredient_id": key, **{split: support[split][key] for split in groups}}
                          for key in sorted(set().union(*(set(s) for s in support.values())))]
    report = {"mode": mode, "group_source": "heuristic_candidates" if mode == "provisional" else "reviewed_candidate_components",
              "certified_independence": False,
              "limitations": ["Candidate heuristics can miss related scans; zero overlap only covers the supplied group map.",
                              "Human review is not an official sequence annotation; audit missed links and sensitivity to tolerances."] +
                             (["Unreviewed candidate links are conservatively grouped; some groups may merge different plates."] if mode == "provisional" else []),
              "samples": dict(counts), "groups": {s: len(g) for s, g in groups.items()}, "group_overlaps": overlaps,
              "official_test_preserved": True, "actual_validation_sample_fraction": counts["validation"] / max(counts["train"] + counts["validation"], 1),
              "excluded_official_boundary_links": sum(bool(link.get("excluded_from_grouping")) for link in audit["links"]),
              "mass_distribution": mass, "train_mean_mass_grams": mass["train"]["mean"],
              "candidate_links_crossing_new_train_validation": len(residual),
              "crossing_links": residual, "ingredient_support": ingredient_support,
              "validation_ingredients_absent_from_train": sorted(set(support["validation"]) - set(support["train"])),
              "metadata_scans_without_rgb": sum(node["image"] is None for node in nodes),
              "unassigned_metadata_scans": sum(node["official_split"] == "unassigned" for node in nodes)}
    if mode == "provisional" and residual:
        raise ValueError("Provisional split separates candidate links.")
    return report


def verify_sources(audit, root):
    for relative, expected in audit["source_sha256"].items():
        if file_digest(Path(root) / relative) != expected:
            raise ValueError(f"Source changed since audit: {relative}. Generate a new audit.")
    for node in audit["nodes"]:
        current = find_rgb_image(Path(root), node["dish_id"])
        if bool(current) != bool(node["image"]):
            raise ValueError(f"RGB availability changed for {node['dish_id']}; regenerate audit.")
        if current and file_digest(current) != node["image_sha256"]:
            raise ValueError(f"RGB changed since audit: {node['dish_id']}")
