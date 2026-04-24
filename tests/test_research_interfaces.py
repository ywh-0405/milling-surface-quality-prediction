import pytest

torch = pytest.importorskip("torch")

from src.models_research import build_research_model


def test_build_research_model_has_two_heads():
    model = build_research_model(n_features=20, n_freq_bins=8)
    assert hasattr(model, "ra_head")
    assert hasattr(model, "freq_head")

