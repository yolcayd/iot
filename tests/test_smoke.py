"""Smoke test: every third-party dependency and every project module can be imported."""

import importlib

import pytest

THIRD_PARTY = [
    "numpy",
    "pandas",
    "pyarrow",
    "sklearn",
    "xgboost",
    "torch",
    "onnx",
    "onnxruntime",
    "tensorflow",
    "scipy",
    "yaml",
    "matplotlib",
    "seaborn",
    "psutil",
]

PROJECT = [
    "src",
    "src.data",
    "src.data.loaders",
    "src.data.features",
    "src.data.windows",
    "src.data.splits",
    "src.data.sources",
    "src.data.dataset",
    "src.models",
    "src.models.cnn1d",
    "src.models.lstm_autoencoder",
    "src.models.conv_autoencoder",
    "src.models.classical",
    "src.train",
    "src.quantize",
    "src.evaluate",
    "src.benchmark",
    "src.run_grid",
    "src.sanity",
    "analysis",
    "analysis.aggregate",
    "analysis.stats",
    "analysis.figures",
]


@pytest.mark.parametrize("name", THIRD_PARTY + PROJECT)
def test_import(name):
    importlib.import_module(name)
