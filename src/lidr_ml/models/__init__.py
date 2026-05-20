from lidr_ml.models.base import Model
from lidr_ml.models.logistic import LogisticRegressionModel

# name → Model class
MODEL_REGISTRY: dict[str, type[Model]] = {
    "logistic_regression": LogisticRegressionModel,
}


def build_model(spec: dict) -> Model:
    """Instantiate a model from a config spec like {type: ..., params: {...}}."""
    name = spec["type"]
    params = spec.get("params", {}) or {}
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)
