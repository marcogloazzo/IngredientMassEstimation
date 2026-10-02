from __future__ import annotations

import numpy as np


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-12))


def ingredient_recognition_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = targets > 0.5
    true_positive = np.logical_and(predictions, truth).sum(axis=0).astype(float)
    false_positive = np.logical_and(predictions, ~truth).sum(axis=0).astype(float)
    false_negative = np.logical_and(~predictions, truth).sum(axis=0).astype(float)

    precision_per_class = true_positive / np.maximum(true_positive + false_positive, 1.0)
    recall_per_class = true_positive / np.maximum(true_positive + false_negative, 1.0)
    f1_per_class = 2 * precision_per_class * recall_per_class / np.maximum(
        precision_per_class + recall_per_class, 1e-12
    )
    supported = truth.sum(axis=0) > 0

    tp = float(true_positive.sum())
    fp = float(false_positive.sum())
    fn = float(false_negative.sum())
    micro_precision = _safe_ratio(tp, tp + fp)
    micro_recall = _safe_ratio(tp, tp + fn)
    micro_f1 = _safe_ratio(
        2 * micro_precision * micro_recall, micro_precision + micro_recall
    )
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_f1": float(f1_per_class[supported].mean()) if supported.any() else 0.0,
        "exact_match": float(np.all(predictions == truth, axis=1).mean()),
    }


def find_best_global_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    thresholds: np.ndarray | None = None,
    objective: str = "micro_f1",
) -> tuple[float, dict[str, float]]:
    """Select one robust presence threshold using validation labels only."""
    if objective not in {"micro_f1", "macro_f1"}:
        raise ValueError("objective must be 'micro_f1' or 'macro_f1'")
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181, dtype=np.float32)

    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    for candidate in thresholds:
        candidate_metrics = ingredient_recognition_metrics(
            probabilities, targets, float(candidate)
        )
        if best_metrics is None:
            is_better = True
        else:
            candidate_key = (
                candidate_metrics[objective],
                candidate_metrics["micro_precision"],
                -abs(float(candidate) - 0.5),
            )
            best_key = (
                best_metrics[objective],
                best_metrics["micro_precision"],
                -abs(best_threshold - 0.5),
            )
            is_better = candidate_key > best_key
        if is_better:
            best_threshold = float(candidate)
            best_metrics = candidate_metrics

    assert best_metrics is not None
    return best_threshold, best_metrics


def quantity_metrics(
    predicted_shares: np.ndarray,
    predicted_total_mass: np.ndarray,
    target_shares: np.ndarray,
    target_masses: np.ndarray,
    target_total_mass: np.ndarray,
    nutrient_matrix: np.ndarray,
    target_nutrients: np.ndarray,
) -> dict:
    predicted_masses = predicted_shares * predicted_total_mass[:, None]
    present = target_masses > 0
    mass_absolute_errors = np.abs(predicted_masses - target_masses)
    present_mass_mae = (
        float(mass_absolute_errors[present].mean()) if present.any() else 0.0
    )
    total_mass_absolute_errors = np.abs(predicted_total_mass - target_total_mass)

    predicted_nutrients = predicted_masses @ nutrient_matrix
    nutrient_errors = np.abs(predicted_nutrients - target_nutrients)
    nutrient_names = ("calories", "fat", "carbohydrates", "protein")
    nutrient_results = {}
    for index, name in enumerate(nutrient_names):
        nutrient_results[name] = {
            "mae": float(nutrient_errors[:, index].mean()),
            "pmae": _safe_ratio(
                float(nutrient_errors[:, index].sum()),
                float(np.abs(target_nutrients[:, index]).sum()),
            ),
        }

    return {
        "share_mae_percentage_points": float(
            100.0 * np.abs(predicted_shares - target_shares).mean()
        ),
        "ingredient_mass_mae_present_grams": present_mass_mae,
        "ingredient_mass_l1_per_plate_grams": float(mass_absolute_errors.sum(axis=1).mean()),
        "ingredient_mass_mae_all_classes_grams": float(mass_absolute_errors.mean()),
        "false_ingredient_mass_per_plate_grams": float((predicted_masses * ~present).sum(axis=1).mean()),
        "total_mass_mae_grams": float(total_mass_absolute_errors.mean()),
        "total_mass_pmae": _safe_ratio(
            float(total_mass_absolute_errors.sum()),
            float(np.abs(target_total_mass).sum()),
        ),
        "nutrients": nutrient_results,
    }


def full_metrics(
    presence_probabilities: np.ndarray,
    predicted_shares: np.ndarray,
    predicted_total_mass: np.ndarray,
    cache: dict[str, np.ndarray],
    nutrient_matrix: np.ndarray,
    threshold: float,
) -> dict:
    recognition = ingredient_recognition_metrics(
        presence_probabilities, cache["presence"], threshold
    )
    end_to_end = quantity_metrics(
        predicted_shares,
        predicted_total_mass,
        cache["shares"],
        cache["masses"],
        cache["total_mass"],
        nutrient_matrix,
        cache["nutrients"],
    )
    oracle_mass = quantity_metrics(
        predicted_shares,
        cache["total_mass"],
        cache["shares"],
        cache["masses"],
        cache["total_mass"],
        nutrient_matrix,
        cache["nutrients"],
    )
    return {
        "ingredient_recognition": recognition,
        "end_to_end_lightweight_mass": end_to_end,
        "composition_with_true_total_mass": oracle_mass,
    }
