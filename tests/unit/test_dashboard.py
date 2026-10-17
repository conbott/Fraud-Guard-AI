"""Unit tests verifying Streamlit dashboard importability and module health."""

import importlib


def test_dashboard_imports_cleanly():
    """Verify that dashboard module can be imported without syntax, import, or circular dependency errors."""
    mod = importlib.import_module("src.dashboard.app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "get_model_bundle")
    assert hasattr(mod, "get_predictor")


def test_dashboard_model_bundle_loads():
    """Verify that dashboard model loader retrieves champion bundle."""
    from src.dashboard.app import get_model_bundle, get_predictor

    bundle = get_model_bundle()
    assert bundle is not None
    assert bundle.algorithm in ("logistic_regression", "random_forest", "dummy_baseline")

    predictor = get_predictor()
    assert predictor is not None
