"""Scaffold smoke test: the ml package imports and core dependencies resolve."""

import importlib


def test_ml_package_imports() -> None:
    assert importlib.import_module("ml") is not None


def test_core_dependencies_importable() -> None:
    for module in ("torch", "mlflow", "pandera"):
        assert importlib.import_module(module) is not None
