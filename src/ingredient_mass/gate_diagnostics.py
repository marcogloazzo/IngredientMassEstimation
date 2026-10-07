"""Offline counterfactuals for conditional grams; no torch or model execution."""
from __future__ import annotations

import numpy as np

from .metrics import ingredient_recognition_metrics


def _mean(values):
    return float(np.mean(values)) if values.size else None


def mass_summary(predicted, truth, total_mass, matrix=None, nutrients=None):
    present = truth > 0
    error = np.abs(predicted - truth)
    total = predicted.sum(1)
    result = {
        "ingredient_l1_g_per_plate": float(error.sum(1).mean()),
        "present_l1_g_per_plate": float((error * present).sum(1).mean()),
        "false_g_per_plate": float((predicted * ~present).sum(1).mean()),
        "mae_present_g_per_ingredient": _mean(error[present]),
        "bias_present_g_per_ingredient": _mean((predicted - truth)[present]),
        "total_mass_mae_g": float(np.abs(total - total_mass).mean()),
        "total_mass_bias_g": float((total - total_mass).mean()),
    }
    if matrix is not None and nutrients is not None:
        errors = np.abs(predicted @ matrix - nutrients)
        result["nutrient_mae"] = dict(zip(("calories_kcal", "fat_g", "carbohydrates_g", "protein_g"), errors.mean(0).tolist()))
    return result


def diagnose_gate(arrays, details, vocabulary=None, thresholds=None):
    if details.get("split") != "validation":
        raise ValueError("Gate exploration/threshold selection is validation-only; never tune on test")
    required = ("probabilities", "conditional_grams", "masses", "predicted_shares",
                "predicted_total_mass", "total_mass", "dish_ids", "vocabulary_keys")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Missing arrays: {missing}; use saved V3/V4 evaluation predictions, not V2/intermediate tensors")
    p, c, truth = [np.asarray(arrays[key], dtype=np.float64) for key in ("probabilities", "conditional_grams", "masses")]
    if p.ndim != 2 or p.size == 0 or c.shape != p.shape or truth.shape != p.shape:
        raise ValueError("Expected matching nonempty N x C probabilities/conditional/target masses")
    n, classes = p.shape
    total = np.asarray(arrays["total_mass"], dtype=np.float64)
    predicted_total = np.asarray(arrays["predicted_total_mass"], dtype=np.float64)
    shares = np.asarray(arrays["predicted_shares"], dtype=np.float64)
    keys = np.asarray(arrays["vocabulary_keys"], dtype=str).tolist()
    ids = np.asarray(arrays["dish_ids"], dtype=str).tolist()
    if total.shape != (n,) or predicted_total.shape != (n,) or shares.shape != p.shape or len(keys) != classes or len(ids) != n:
        raise ValueError("Inconsistent sample/class dimensions")
    if len(set(ids)) != n or len(set(keys)) != classes:
        raise ValueError("Duplicate dish IDs or vocabulary keys")
    if details.get("samples", n) != n or details.get("vocabulary_keys", keys) != keys:
        raise ValueError("Evaluation details do not match prediction samples/vocabulary order")
    if not all(np.isfinite(a).all() for a in (p, c, truth, total, predicted_total, shares)):
        raise ValueError("Non-finite values")
    if (p < 0).any() or (p > 1).any() or any((a < 0).any() for a in (c, truth, shares, predicted_total, total)):
        raise ValueError("Probabilities outside [0,1] or negative masses/shares")
    baseline = shares * predicted_total[:, None]
    product = p * c
    is_v6 = details.get("model_type") == "convnext_small_query2label_spatial_clip_asl_v6"
    mass_filter = details.get("mass_filter",{})
    raw_baseline = baseline
    if is_v6:
        if "unfiltered_ingredient_masses" not in arrays or "threshold" not in mass_filter:
            raise ValueError("V6 diagnostics require saved unfiltered masses and mass-filter details")
        raw_baseline = np.asarray(arrays["unfiltered_ingredient_masses"],dtype=np.float64)
        if raw_baseline.shape != p.shape or not np.allclose(raw_baseline,product,atol=1e-3,rtol=1e-4):
            raise ValueError("V6 raw saved masses do not match p*c")
        cutoff = float(mass_filter["threshold"])
        if not 0 <= cutoff <= 1:
            raise ValueError("Invalid V6 mass threshold")
        expected = product * (p >= cutoff) if mass_filter.get("enabled",True) else product
        if not np.allclose(baseline,expected,atol=1e-3,rtol=1e-4):
            raise ValueError("V6 deployed masses do not match the recorded cutoff")
    elif not np.allclose(baseline, product, atol=1e-3, rtol=1e-4):
        raise ValueError("Saved grams do not match p*c: do not analyze a different mass rule with this diagnostic")
    if "ingredient_masses" in arrays and not np.allclose(arrays["ingredient_masses"], baseline, atol=1e-3, rtol=1e-4):
        raise ValueError("Explicit ingredient masses differ from shares * total")
    present = truth > 0
    if "presence" in arrays and not np.array_equal(np.asarray(arrays["presence"]) > .5, present):
        raise ValueError("Presence targets disagree with positive target grams")
    threshold = float(details.get("threshold", .5))
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid report presence threshold")
    names, matrix, nutrients = keys, None, None
    if vocabulary is not None:
        from .data import nutrient_matrix
        if [entry["key"] for entry in vocabulary["entries"]] != keys:
            raise ValueError("Vocabulary keys/order differ from predictions")
        if vocabulary["nutrient_order"] != ["calories", "fat", "carbohydrates", "protein"]:
            raise ValueError("Unexpected nutrient order")
        names = [entry["name"] for entry in vocabulary["entries"]]
        matrix = nutrient_matrix(vocabulary).astype(np.float64)
        nutrients = np.asarray(arrays["nutrients"], dtype=np.float64)
        if nutrients.shape != (n, 4) or not np.isfinite(nutrients).all() or not np.isfinite(matrix).all():
            raise ValueError("Invalid nutrient targets/matrix")

    def scenario(name, masses, *, oracle=False, rule="", candidate_threshold=None):
        row = {"name": name, "oracle_not_deployable": oracle, "rule": rule,
               **mass_summary(masses, truth, total, matrix, nutrients)}
        if candidate_threshold is not None:
            row["threshold"] = float(candidate_threshold)
        return row

    scenarios = [
        scenario("baseline", baseline, rule="p*c with recorded mass cutoff" if is_v6 else "p*c (saved model outputs)"),
        scenario("oracle_remove_false_mass", baseline * present, oracle=True, rule="GT-presence * p*c"),
        scenario("oracle_presence_conditional", c * present, oracle=True, rule="GT-presence * c"),
        scenario("hard_product_report_threshold", raw_baseline * (p >= threshold), rule="1[p>=tau] * p*c", candidate_threshold=threshold),
        scenario("hard_conditional_report_threshold", c * (p >= threshold), rule="1[p>=tau] * c", candidate_threshold=threshold),
    ]
    if is_v6:
        scenarios.insert(1,scenario("unfiltered",raw_baseline,rule="p*c before mass cutoff"))
    safe = p.clip(1e-12, 1 - 1e-12)
    logits = np.log(safe) - np.log1p(-safe)
    for temperature in (.5, 2.):
        gate = 1 / (1 + np.exp(-logits / temperature))
        scenarios.append(scenario(f"temperature_{temperature:g}", gate * c, rule=f"sigmoid(logit(p)/{temperature:g}) * c; not fitted calibration"))
    candidates = np.asarray(np.linspace(0, 1, 41) if thresholds is None else thresholds, dtype=float)
    if candidates.ndim != 1 or not candidates.size or not np.isfinite(candidates).all() or (candidates < 0).any() or (candidates > 1).any():
        raise ValueError("Threshold grid must be a nonempty finite 1-D array inside [0,1]")
    sweep = []
    for candidate in candidates:
        recognized = p >= candidate
        for name, grams in (("hard_product", raw_baseline), ("hard_conditional", c)):
            row = scenario(name, grams * recognized, rule=f"{name}: validation-only sweep", candidate_threshold=candidate)
            row["micro_f1"] = ingredient_recognition_metrics(p, present, float(candidate))["micro_f1"]
            sweep.append(row)
    best = {name: min((row for row in sweep if row["name"] == name), key=lambda r: (r["ingredient_l1_g_per_plate"], abs(r["threshold"] - threshold)))
            for name in ("hard_product", "hard_conditional")}
    final_error, conditional_error = np.abs(baseline - truth), np.abs(c - truth)
    raw_error = np.abs(raw_baseline-truth)
    class_scales = np.asarray(arrays["class_gram_scales"],dtype=float) if is_v6 and "class_gram_scales" in arrays else None
    if class_scales is not None and (class_scales.shape != (classes,) or not np.isfinite(class_scales).all() or (class_scales<=0).any()):
        raise ValueError("Invalid saved V6 class scales")
    class_rows = []
    for i in range(classes):
        mask = present[:, i]
        support = int(mask.sum())
        class_rows.append({
            "key": keys[i], "name": names[i], "support": support,
            "conditional_mae_present_g": _mean(conditional_error[mask, i]),
            "final_mae_present_g": _mean(final_error[mask, i]),
            "gate_effect_mae_g": _mean((raw_error - conditional_error)[mask, i]),
            **({"filter_effect_mae_g":_mean((final_error-raw_error)[mask,i])} if is_v6 else {}),
            **({"unfiltered_mae_present_g":_mean(raw_error[mask,i]),
                "scale_grams":float(class_scales[i]) if class_scales is not None else None} if is_v6 else {}),
            "conditional_bias_present_g": _mean((c - truth)[mask, i]),
            "final_bias_present_g": _mean((baseline - truth)[mask, i]),
            "mean_gate_present": _mean(p[mask, i]),
            "mean_attenuated_present_g": _mean(((1 - p) * c)[mask, i]),
            "l1_contribution_g_per_plate": float(final_error[:, i].mean()),
            "false_g_per_plate": float((baseline[:, i] * ~mask).mean()),
        })
    buckets = []
    detected = p >= threshold
    for name, mask in (("TP", present & detected), ("FN", present & ~detected),
                       ("FP", ~present & detected), ("TN", ~present & ~detected)):
        buckets.append({"group": name, "ingredient_entries": int(mask.sum()), "mean_probability": _mean(p[mask]),
                        "mean_conditional_g": _mean(c[mask]), "conditional_mae_g": _mean(conditional_error[mask]),
                        "final_mae_g": _mean(final_error[mask]), "final_bias_g": _mean((baseline - truth)[mask])})
    report = {
        "schema_version": 1, "split": "validation", "samples": n, "classes": classes,
        "epoch": details.get("epoch"), "model_type": details.get("model_type"),
        "recognition_threshold": threshold, "threshold_source": "evaluation_details" if "threshold" in details else "explicit diagnostic default 0.5; original report threshold unavailable",
        "checkpoint_sha256": details.get("checkpoint_sha256"), "prepared_fingerprint": details.get("prepared_fingerprint"),
        "max_saved_vs_product_difference_g": float(np.abs(raw_baseline - product).max()),
        **({"mass_filter":mass_filter} if is_v6 else {}),
        "global_present": {
            "entries": int(present.sum()), "conditional_mae_g": _mean(conditional_error[present]),
            "final_mae_g": _mean(final_error[present]), "mean_gate": _mean(p[present]),
            **({"unfiltered_mae_g":_mean(raw_error[present])} if is_v6 else {}),
            "conditional_bias_g": _mean((c - truth)[present]), "final_bias_g": _mean((baseline - truth)[present]),
            "gate_worsens_error_fraction": _mean((raw_error[present] > conditional_error[present] + 1e-6).astype(float)),
        },
        "scenarios": scenarios, "best_validation_thresholds": best, "threshold_sweep": sweep,
        "classes_by_l1": sorted(class_rows, key=lambda r: r["l1_contribution_g_per_plate"], reverse=True),
        "recognition_buckets": buckets,
        "worst_plates": [{"dish_id": ids[i], "l1_g": float(final_error[i].sum()), "false_g": float((baseline[i] * ~present[i]).sum()),
                          "present_ingredients": int(present[i].sum()), "total_mass_bias_g": float(baseline[i].sum() - total[i])}
                         for i in np.argsort(final_error.sum(1))[::-1][:20]],
        "warnings": [
            "Oracle scenarios use true presence and are NOT deployable or guaranteed error lower bounds.",
            "Threshold optima are in-sample validation explorations, not independent improvements or test results.",
            "Changing F1 thresholds alone does not change grams. No checkpoint or deployed rule is modified here.",
            "Gate effect is a counterfactual with fixed c, not proof of architectural/gradient causality.",
            "Rare classes have uncertain averages; p*c and c may compensate each other.",
        ],
    }
    if "auxiliary_total_mass" in arrays:
        auxiliary = np.asarray(arrays["auxiliary_total_mass"], dtype=np.float64)
        if auxiliary.shape != (n,) or not np.isfinite(auxiliary).all() or (auxiliary < 0).any():
            raise ValueError("Invalid auxiliary total mass")
        report["auxiliary_total_mass"] = {"mae_g": float(np.abs(auxiliary - total).mean()),
                                           "bias_g": float((auxiliary - total).mean()),
                                           "gap_to_ingredient_sum_mae_g": float(np.abs(auxiliary - baseline.sum(1)).mean())}
    return report
