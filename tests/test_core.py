import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ingredient_mass.data import (
    nutrient_matrix,
    parse_dish_metadata_file,
    vectorize_record,
)
from ingredient_mass.metrics import find_best_global_threshold, full_metrics
from ingredient_mass.model import IngredientMassHeads
from ingredient_mass.features import create_frozen_encoder, validate_feature_cache


VOCABULARY = {
    "nutrient_order": ["calories", "fat", "carbohydrates", "protein"],
    "entries": [
        {
            "key": "rice",
            "name": "rice",
            "nutrients_per_gram": {
                "calories": 1.3,
                "fat": 0.0,
                "carbohydrates": 0.28,
                "protein": 0.03,
            },
        },
        {
            "key": "__other__",
            "name": "other",
            "nutrients_per_gram": {
                "calories": 2.0,
                "fat": 0.1,
                "carbohydrates": 0.1,
                "protein": 0.1,
            },
        },
    ],
}


def test_model_shares_sum_to_one():
    model = IngredientMassHeads(8, 3, hidden_dim=4, dropout=0.0)
    outputs = model(torch.randn(5, 8))
    assert torch.allclose(outputs["shares"].sum(dim=1), torch.ones(5), atol=1e-6)
    assert torch.all(outputs["total_mass"] > 0)


def test_metadata_parser_accepts_both_public_and_documented_layouts():
    public_row = (
        "dish_1,100,80,2,10,5,"
        "ingr_0000000001,rice,80,100,2,10,5\n"
    )
    documented_row = (
        "dish_2,100,80,2,10,5,1,"
        "ingr_0000000001,rice,80,100,2,10,5\n"
    )
    with patch.object(Path, "open", return_value=io.StringIO(public_row + documented_row)):
        dishes = parse_dish_metadata_file(Path("dish_metadata_test.csv"))
    assert len(dishes) == 2
    assert dishes[0].ingredients[0].name == "rice"
    assert dishes[1].ingredients[0].grams == 80.0


def test_vectorize_and_perfect_metrics():
    record = {
        "ingredients": [
            {
                "ingredient_id": "rice",
                "grams": 80.0,
            },
            {
                "ingredient_id": "unknown",
                "grams": 20.0,
            },
        ],
        "total_mass": 100.0,
        "nutrients": {
            "calories": 144.0,
            "fat": 2.0,
            "carbohydrates": 24.4,
            "protein": 4.4,
        },
    }
    target = vectorize_record(record, VOCABULARY)
    assert np.allclose(target["shares"], [0.8, 0.2])
    cache = {
        "presence": target["presence"][None],
        "shares": target["shares"][None],
        "masses": target["masses"][None],
        "total_mass": np.array([100.0], dtype=np.float32),
        "nutrients": target["nutrients"][None],
    }
    metrics = full_metrics(
        cache["presence"],
        cache["shares"],
        cache["total_mass"],
        cache,
        nutrient_matrix(VOCABULARY),
        0.5,
    )
    assert metrics["ingredient_recognition"]["micro_f1"] == 1.0
    assert metrics["end_to_end_lightweight_mass"]["total_mass_mae_grams"] == 0.0


def test_threshold_calibration_improves_f1():
    probabilities = np.array([[0.4, 0.2], [0.45, 0.1]], dtype=np.float32)
    targets = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    threshold, metrics = find_best_global_threshold(probabilities, targets)
    assert threshold <= 0.401
    assert metrics["micro_f1"] == 1.0


def test_convnext_encoder_keeps_normalization_without_downloads():
    fake_encoder = torch.nn.Module()
    normalization = torch.nn.LayerNorm(768)
    fake_encoder.classifier = torch.nn.Sequential(
        normalization, torch.nn.Flatten(1), torch.nn.Linear(768, 1000),
    )
    with patch("torchvision.models.convnext_small", return_value=fake_encoder) as builder:
        encoder, transform = create_frozen_encoder(encoder_name="convnext_small")
    builder.assert_called_once()
    assert encoder.classifier[0] is normalization
    assert isinstance(encoder.classifier[-1], torch.nn.Identity)
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert transform.crop_size == [224]


def test_new_cache_checks_encoder_and_ingredient_order():
    cache = {
        "features": np.zeros((1, 768), dtype=np.float32),
        "presence": np.zeros((1, 2), dtype=np.float32),
        "encoder_name": np.asarray("convnext_small"),
        "vocabulary_keys": np.asarray(["rice", "__other__"]),
    }
    validate_feature_cache(cache, VOCABULARY, "convnext_small")
    for change in (
        {"features": np.zeros((1, 1280), dtype=np.float32)},
        {"presence": np.zeros((1, 50), dtype=np.float32)},
        {"encoder_name": np.asarray("efficientnet_b0")},
        {"vocabulary_keys": np.asarray(["__other__", "rice"])},
    ):
        try:
            validate_feature_cache({**cache, **change}, VOCABULARY, "convnext_small")
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible feature cache was accepted")


def test_hundred_class_outputs():
    model = IngredientMassHeads(768, 100, hidden_dim=192, dropout=0.4)
    outputs = model(torch.zeros(2, 768))
    assert outputs["presence_logits"].shape == (2, 100)
    assert outputs["shares"].shape == (2, 100)
    assert torch.allclose(outputs["shares"].sum(1), torch.ones(2))


if __name__ == "__main__":
    test_model_shares_sum_to_one()
    test_metadata_parser_accepts_both_public_and_documented_layouts()
    test_vectorize_and_perfect_metrics()
    test_threshold_calibration_improves_f1()
    test_convnext_encoder_keeps_normalization_without_downloads()
    test_new_cache_checks_encoder_and_ingredient_order()
    test_hundred_class_outputs()
    print("All synthetic tests passed.")
