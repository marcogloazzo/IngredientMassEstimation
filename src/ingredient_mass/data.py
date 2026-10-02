from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NUTRIENT_NAMES = ("calories", "fat", "carbohydrates", "protein")
OTHER_KEY = "__other__"


@dataclass(frozen=True)
class IngredientComponent:
    ingredient_id: str
    name: str
    grams: float
    calories: float
    fat: float
    carbohydrates: float
    protein: float


@dataclass(frozen=True)
class DishRecord:
    dish_id: str
    total_calories: float
    total_mass: float
    total_fat: float
    total_carbohydrates: float
    total_protein: float
    ingredients: tuple[IngredientComponent, ...]


def _as_float(value: str, context: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value {value!r} in {context}") from error


def parse_dish_metadata_file(path: Path) -> list[DishRecord]:
    """Parse a variable-width Nutrition5k dish metadata CSV."""
    dishes: list[DishRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or not row[0].strip():
                continue
            if row[0].strip().lower() in {"dish_id", "id"}:
                continue
            if len(row) < 6:
                raise ValueError(f"{path}:{line_number} has fewer than 6 dish fields")

            context = f"{path.name}:{line_number}"
            # The public CSV files currently contain six dish fields followed
            # directly by groups of seven ingredient fields. Older documentation
            # also describes a num_ingrs column, so accept both layouts.
            remaining_fields = row[6:]
            has_explicit_count = bool(remaining_fields) and not remaining_fields[0].strip().startswith(
                "ingr_"
            )
            if has_explicit_count:
                try:
                    number_of_ingredients = int(float(remaining_fields[0]))
                except ValueError as error:
                    raise ValueError(
                        f"{context} has neither an ingredient ID nor a numeric count "
                        f"in field 7: {remaining_fields[0]!r}"
                    ) from error
                ingredient_fields = remaining_fields[1:]
                if number_of_ingredients == 0:
                    stride = 0
                else:
                    if len(ingredient_fields) % number_of_ingredients != 0:
                        raise ValueError(
                            f"{context} has {len(ingredient_fields)} ingredient fields for "
                            f"{number_of_ingredients} ingredients"
                        )
                    stride = len(ingredient_fields) // number_of_ingredients
            else:
                ingredient_fields = remaining_fields
                stride = 7
                if len(ingredient_fields) % stride != 0:
                    raise ValueError(
                        f"{context} has {len(ingredient_fields)} ingredient fields; "
                        "expected groups of 7"
                    )
                number_of_ingredients = len(ingredient_fields) // stride

            if number_of_ingredients > 0 and stride < 7:
                raise ValueError(f"{context} has unsupported ingredient stride {stride}")

            ingredients: list[IngredientComponent] = []
            for index in range(number_of_ingredients):
                fields = ingredient_fields[index * stride : (index + 1) * stride]
                ingredients.append(
                    IngredientComponent(
                        ingredient_id=fields[0].strip(),
                        name=fields[1].strip(),
                        grams=_as_float(fields[2], context),
                        calories=_as_float(fields[3], context),
                        fat=_as_float(fields[4], context),
                        carbohydrates=_as_float(fields[5], context),
                        protein=_as_float(fields[6], context),
                    )
                )

            dishes.append(
                DishRecord(
                    dish_id=row[0].strip(),
                    total_calories=_as_float(row[1], context),
                    total_mass=_as_float(row[2], context),
                    total_fat=_as_float(row[3], context),
                    total_carbohydrates=_as_float(row[4], context),
                    total_protein=_as_float(row[5], context),
                    ingredients=tuple(ingredients),
                )
            )
    return dishes


def load_all_dishes(metadata_dir: Path) -> dict[str, DishRecord]:
    paths = sorted(metadata_dir.glob("dish_metadata*.csv"))
    if not paths:
        raise FileNotFoundError(f"No dish_metadata*.csv files found in {metadata_dir}")
    dishes: dict[str, DishRecord] = {}
    for path in paths:
        for dish in parse_dish_metadata_file(path):
            if dish.dish_id in dishes:
                raise ValueError(f"Duplicate dish ID: {dish.dish_id}")
            dishes[dish.dish_id] = dish
    return dishes


def _find_split_file(splits_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        matches = list(splits_dir.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"None of {list(candidates)} was found below {splits_dir}"
    )


def read_split_ids(splits_dir: Path) -> tuple[list[str], list[str]]:
    train_path = _find_split_file(
        splits_dir, ("rgb_train_ids.txt", "train_ids.txt", "train.txt")
    )
    test_path = _find_split_file(
        splits_dir, ("rgb_test_ids.txt", "test_ids.txt", "test.txt")
    )

    def read_ids(path: Path) -> list[str]:
        return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]

    return read_ids(train_path), read_ids(test_path)


def split_train_validation(
    train_ids: list[str], validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    shuffled = list(dict.fromkeys(train_ids))
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_ids = set(shuffled[:validation_count])
    return set(shuffled[validation_count:]), validation_ids


def find_rgb_image(dataset_root: Path, dish_id: str) -> Path | None:
    expected = dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    return expected if expected.is_file() else None


def build_records(
    dataset_root: Path,
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict], dict]:
    dishes = load_all_dishes(dataset_root / "metadata")
    train_ids, test_ids = read_split_ids(dataset_root / "dish_ids" / "splits")
    train_set, validation_set = split_train_validation(
        train_ids, validation_fraction, seed
    )
    test_set = set(test_ids)

    requested = train_set | validation_set | test_set
    missing_metadata: list[str] = []
    missing_images: list[str] = []
    records: list[dict] = []

    for dish_id in sorted(requested):
        dish = dishes.get(dish_id)
        if dish is None:
            missing_metadata.append(dish_id)
            continue
        image_path = find_rgb_image(dataset_root, dish_id)
        if image_path is None:
            missing_images.append(dish_id)
            continue
        if dish_id in train_set:
            split = "train"
        elif dish_id in validation_set:
            split = "validation"
        else:
            split = "test"

        relative_image = image_path.relative_to(dataset_root).as_posix()
        records.append(
            {
                "dish_id": dish.dish_id,
                "split": split,
                "image": relative_image,
                "total_mass": dish.total_mass,
                "nutrients": {
                    "calories": dish.total_calories,
                    "fat": dish.total_fat,
                    "carbohydrates": dish.total_carbohydrates,
                    "protein": dish.total_protein,
                },
                "ingredients": [asdict(component) for component in dish.ingredients],
            }
        )

    summary = {
        "metadata_dishes": len(dishes),
        "requested_split_ids": len(requested),
        "usable_records": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "missing_metadata_count": len(missing_metadata),
        "missing_image_count": len(missing_images),
        "missing_metadata_examples": missing_metadata[:20],
        "missing_image_examples": missing_images[:20],
    }
    return records, summary


def build_vocabulary(records: list[dict], top_k: int) -> dict:
    if top_k < 3:
        raise ValueError("top_k must be at least 3")
    train_records = [record for record in records if record["split"] == "train"]
    frequency: Counter[str] = Counter()
    names: dict[str, Counter[str]] = defaultdict(Counter)
    grams_sum: Counter[str] = Counter()
    nutrient_sums: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float64))

    for record in train_records:
        seen: set[str] = set()
        for component in record["ingredients"]:
            key = component["ingredient_id"]
            names[key][component["name"]] += 1
            grams = max(0.0, float(component["grams"]))
            grams_sum[key] += grams
            nutrient_sums[key] += np.array(
                [
                    component["calories"],
                    component["fat"],
                    component["carbohydrates"],
                    component["protein"],
                ],
                dtype=np.float64,
            )
            seen.add(key)
        frequency.update(seen)

    ranked = sorted(
        frequency,
        key=lambda key: (frequency[key], grams_sum[key]),
        reverse=True,
    )
    selected = ranked[: top_k - 1]
    excluded = set(ranked[top_k - 1 :])

    entries: list[dict] = []
    for key in selected:
        total_grams = grams_sum[key]
        per_gram = nutrient_sums[key] / max(total_grams, 1e-12)
        entries.append(
            {
                "key": key,
                "name": names[key].most_common(1)[0][0],
                "support_dishes": frequency[key],
                "total_grams": total_grams,
                "nutrients_per_gram": dict(zip(NUTRIENT_NAMES, per_gram.tolist())),
            }
        )

    other_grams = sum(grams_sum[key] for key in excluded)
    other_nutrients = sum(
        (nutrient_sums[key] for key in excluded), np.zeros(4, dtype=np.float64)
    )
    other_per_gram = other_nutrients / max(other_grams, 1e-12)
    entries.append(
        {
            "key": OTHER_KEY,
            "name": "other",
            "support_dishes": sum(frequency[key] for key in excluded),
            "total_grams": other_grams,
            "nutrients_per_gram": dict(zip(NUTRIENT_NAMES, other_per_gram.tolist())),
        }
    )
    return {
        "nutrient_order": list(NUTRIENT_NAMES),
        "entries": entries,
        "selection": "top ingredients by training-dish frequency; remainder grouped as other",
    }


def vectorize_record(record: dict, vocabulary: dict) -> dict[str, np.ndarray | float]:
    entries = vocabulary["entries"]
    key_to_index = {entry["key"]: index for index, entry in enumerate(entries)}
    other_index = key_to_index[OTHER_KEY]
    masses = np.zeros(len(entries), dtype=np.float32)
    for component in record["ingredients"]:
        index = key_to_index.get(component["ingredient_id"], other_index)
        masses[index] += max(0.0, float(component["grams"]))
    presence = (masses > 0).astype(np.float32)
    component_total = float(masses.sum())
    shares = masses / max(component_total, 1e-12)
    nutrients = np.array(
        [record["nutrients"][name] for name in NUTRIENT_NAMES], dtype=np.float32
    )
    return {
        "presence": presence,
        "shares": shares.astype(np.float32),
        "masses": masses,
        "total_mass": float(record["total_mass"]),
        "nutrients": nutrients,
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def nutrient_matrix(vocabulary: dict) -> np.ndarray:
    order = vocabulary["nutrient_order"]
    return np.array(
        [
            [entry["nutrients_per_gram"][name] for name in order]
            for entry in vocabulary["entries"]
        ],
        dtype=np.float32,
    )
