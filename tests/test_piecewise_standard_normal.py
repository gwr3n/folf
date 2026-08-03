import numpy as np
import pytest

from folf import (
    PARTITIONER,
    JensenMinimaxPartitioner,
    JensenUniformPartitioner,
    PiecewiseStandardNormalFirstOrderLossFunction,
)


@pytest.fixture(autouse=True)
def isolate_class_state(monkeypatch: pytest.MonkeyPatch):
    model = PiecewiseStandardNormalFirstOrderLossFunction
    monkeypatch.setattr(model, "_partitioner", PARTITIONER.MINIMAX)
    monkeypatch.setattr(model, "_error_cache", {})
    monkeypatch.setattr(model, "_probabilities_cache", {})
    monkeypatch.setattr(model, "_means_cache", {})


def test_metadata_accessors() -> None:
    model = PiecewiseStandardNormalFirstOrderLossFunction

    assert model.get_partitioner() == PARTITIONER.MINIMAX
    assert model.get_linearization_samples() == 0


@pytest.mark.parametrize(
    "getter",
    [
        PiecewiseStandardNormalFirstOrderLossFunction.get_error,
        PiecewiseStandardNormalFirstOrderLossFunction.get_probabilities,
        PiecewiseStandardNormalFirstOrderLossFunction.get_means,
    ],
)
def test_getters_reject_non_positive_partition_counts(getter) -> None:
    with pytest.raises(ValueError, match="partitions must be >= 1"):
        getter(0)


def test_getters_fill_and_reuse_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PiecewiseStandardNormalFirstOrderLossFunction
    expected = JensenMinimaxPartitioner().compute(5)
    calls = 0

    def compute(partitions: int):
        nonlocal calls
        calls += 1
        assert partitions == 4
        return expected

    monkeypatch.setattr(model, "_compute", compute)

    assert model.get_error(4) == expected.error
    assert model.get_error(4) == expected.error
    assert np.array_equal(model.get_probabilities(4), expected.p)
    assert np.array_equal(model.get_probabilities(4), expected.p)
    assert np.array_equal(model.get_means(4), expected.expect)
    assert np.array_equal(model.get_means(4), expected.expect)
    assert calls == 3


def test_probability_and_mean_results_are_defensive_copies() -> None:
    model = PiecewiseStandardNormalFirstOrderLossFunction

    probabilities = model.get_probabilities(4)
    means = model.get_means(4)
    probabilities[0] = -1.0
    means[0] = 999.0

    fresh_probabilities = model.get_probabilities(4)
    fresh_means = model.get_means(4)
    assert fresh_probabilities[0] > 0.0
    assert fresh_means[0] < 0.0


def test_compute_selects_uniform_partitioner(monkeypatch: pytest.MonkeyPatch) -> None:
    model = PiecewiseStandardNormalFirstOrderLossFunction
    monkeypatch.setattr(model, "_partitioner", PARTITIONER.UNIFORM)

    actual = model._compute(4)
    expected = JensenUniformPartitioner().compute(5)

    assert np.allclose(actual.p, expected.p)
    assert np.allclose(actual.expect, expected.expect)
    assert np.isclose(actual.error, expected.error)
