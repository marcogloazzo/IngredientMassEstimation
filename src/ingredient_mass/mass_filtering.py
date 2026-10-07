"""Offline V6 mass cutoff and constrained validation calibration (no torch)."""
import numpy as np

from .metrics import full_metrics


def apply_mass_filter(predictions, threshold, enabled=True):
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Mass threshold must be finite and inside [0,1]")
    probability = np.asarray(predictions["presence"])
    raw = np.asarray(predictions["unfiltered_ingredient_masses"])
    if probability.shape != raw.shape or raw.ndim != 2 or not raw.size or not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("Invalid raw probability/mass arrays")
    if not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any():
        raise ValueError("Invalid presence probabilities")
    retained = probability >= threshold if enabled else np.ones_like(probability,dtype=bool)
    masses = raw*retained
    total = masses.sum(1)
    return {**predictions,"ingredient_masses":masses,"mass":total,
            "shares":np.divide(masses,total[:,None],out=np.zeros_like(masses),where=total[:,None]>1e-8),
            "mass_retained":retained}


def calibrate_mass_threshold(predictions, targets, matrix, config, *, split):
    if split != "validation":
        raise ValueError("Mass threshold must be selected on validation only, never test")
    settings = config["mass_filter"]
    baseline = apply_mass_filter(predictions,0.)
    truth = targets["masses"]
    def scores(candidate):
        masses = candidate["ingredient_masses"]
        return {"ingredient_l1_g_per_plate":float(np.abs(masses-truth).sum(1).mean()),
                "total_mass_mae_g":float(np.abs(candidate["mass"]-targets["total_mass"]).mean()),
                "calorie_mae_kcal":float(np.abs((masses@matrix)[:,0]-targets["nutrients"][:,0]).mean()),
                "false_g_per_plate":float((masses*(truth==0)).sum(1).mean())}
    base = scores(baseline)
    if not settings["enabled"]:
        return 0.,{"split":"validation","method":"mass_filter_disabled","selected":{"threshold":0.,**base},
                    "baseline_unfiltered":base,"candidates":[]}
    mass_tolerance = float(settings["calibration_max_mass_increase"])
    calorie_tolerance = float(settings["calibration_max_calorie_increase"])
    if not np.isfinite([mass_tolerance,calorie_tolerance]).all() or min(mass_tolerance,calorie_tolerance)<0:
        raise ValueError("Calibration guard tolerances must be finite/nonnegative")
    candidates = np.unique(np.r_[0.,float(settings["threshold"]),np.linspace(.025,.95,38)])
    rows = []
    for threshold in candidates:
        row = {"threshold":float(threshold),**scores(apply_mass_filter(predictions,threshold))}
        row["eligible"] = (row["total_mass_mae_g"] <= base["total_mass_mae_g"]*(1+mass_tolerance)+1e-6 and
                           row["calorie_mae_kcal"] <= base["calorie_mae_kcal"]*(1+calorie_tolerance)+1e-6)
        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    chosen = min(eligible,key=lambda row:(row["ingredient_l1_g_per_plate"],
                    abs(row["threshold"]-float(settings["threshold"]))))
    return chosen["threshold"],{"split":"validation","method":"min_l1_with_mass_calorie_guards",
        "baseline_unfiltered":base,"selected":chosen,"candidates":rows,
        "max_mass_increase_fraction":mass_tolerance,"max_calorie_increase_fraction":calorie_tolerance,
        "warning":"In-sample validation tuning, not test evidence. Cutoff may be 0 if filtering harms the guarded metrics."}


def v6_metric_details(predictions, targets, matrix, config):
    dense = apply_mass_filter(predictions,0.)
    dense_metrics = full_metrics(dense["presence"],dense["shares"],dense["mass"],targets,matrix,float(config["presence_threshold"]))
    filtered = predictions["ingredient_masses"]
    raw = predictions["unfiltered_ingredient_masses"]
    zero = predictions["mass"] <= 1e-8
    result = {"unfiltered_end_to_end":dense_metrics["end_to_end_lightweight_mass"],
              "mass_filter":{"enabled":bool(config["mass_filter"]["enabled"]),
                "threshold":float(config["mass_filter"]["threshold"]),
                "removed_g_per_plate":float((raw-filtered).sum(1).mean()),
                "empty_predicted_plates":int(zero.sum()),
                "training_losses_use_unfiltered_masses":True}}
    if zero.any():
        valid = ~zero
        if valid.any():
            subset = {key:value[valid] for key,value in targets.items()}
            comp = full_metrics(predictions["presence"][valid],predictions["shares"][valid],
                predictions["mass"][valid],subset,matrix,float(config["presence_threshold"]))["composition_with_true_total_mass"]
            comp.update(valid_samples=int(valid.sum()),excluded_zero_mass_samples=int(zero.sum()))
        else:
            comp = {"available":False,"valid_samples":0,"excluded_zero_mass_samples":int(zero.sum()),
                    "reason":"No surviving ingredient masses: oracle composition cannot be formed"}
        result["composition_with_true_total_mass"] = comp
    return result
