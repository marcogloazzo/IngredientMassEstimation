"""Ingredient recognition and mass-allocation utilities."""

__all__ = ["IngredientMassHeads"]


def __getattr__(name):
    # Dataset audit/preparation needs numpy/Pillow, not a torch import or GPU.
    # Preserve the public model export when it is actually requested.
    if name == "IngredientMassHeads":
        from .model import IngredientMassHeads
        globals()[name] = IngredientMassHeads
        return IngredientMassHeads
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
