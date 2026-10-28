"""Feature contract and featurization shared by training and serving.

``Featurizer`` needs PyTorch and is imported on first use, so the serving
image (ONNX Runtime only) can import the contract without it.
"""

from ml.features.contract import FeatureContract

__all__ = ["FeatureContract", "Featurizer"]


def __getattr__(name: str):
    if name == "Featurizer":
        from ml.features.featurize import Featurizer

        return Featurizer
    raise AttributeError(name)
