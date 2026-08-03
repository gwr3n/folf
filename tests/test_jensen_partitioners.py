import math

import numpy as np
import pytest

from folf import JensenMinimaxPartitioner, JensenUniformPartitioner


@pytest.mark.parametrize("partitioner_type", [JensenUniformPartitioner, JensenMinimaxPartitioner])
def test_rejects_fewer_than_two_segments(partitioner_type) -> None:
    with pytest.raises(ValueError, match="segments must be >= 2"):
        partitioner_type().compute(1)


@pytest.mark.parametrize("partitioner_type", [JensenUniformPartitioner, JensenMinimaxPartitioner])
@pytest.mark.parametrize("partitions", [1, 2, 3, 4, 7])
def test_partition_structure_is_normalized_ordered_and_symmetric(
    partitioner_type,
    partitions: int,
) -> None:
    result = partitioner_type().compute(partitions + 1)

    assert result.p.shape == (partitions,)
    assert result.expect.shape == (partitions,)
    assert np.all(result.p > 0.0)
    assert np.isclose(np.sum(result.p), 1.0, atol=1e-12)
    assert np.all(np.diff(result.expect) > 0.0)
    assert np.allclose(result.p, result.p[::-1], atol=1e-10)
    assert np.allclose(result.expect, -result.expect[::-1], atol=1e-10)
    assert np.isclose(result.p @ result.expect, 0.0, atol=1e-12)
    assert result.error > 0.0


def test_single_partition_matches_standard_normal_reference() -> None:
    expected_error = 1.0 / math.sqrt(2.0 * math.pi)

    for partitioner in (JensenUniformPartitioner(), JensenMinimaxPartitioner()):
        result = partitioner.compute(2)
        assert np.array_equal(result.p, np.array([1.0]))
        assert np.array_equal(result.expect, np.array([0.0]))
        assert np.isclose(result.error, expected_error, atol=1e-15)


@pytest.mark.parametrize("partitions", [2, 3, 4, 8, 16])
def test_uniform_partitioner_has_equal_probability_masses(partitions: int) -> None:
    result = JensenUniformPartitioner().compute(partitions + 1)

    assert np.allclose(result.p, np.full(partitions, 1.0 / partitions), atol=1e-12)


def test_minimax_matches_published_reference_errors() -> None:
    expected_errors = JensenMinimaxPartitioner.get_errors()[:6]
    actual_errors = np.array(
        [JensenMinimaxPartitioner().compute(partitions + 1).error for partitions in range(1, 7)]
    )

    assert np.allclose(actual_errors, expected_errors, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize("partitions", [3, 4, 5, 6])
def test_minimax_error_is_no_worse_than_uniform(partitions: int) -> None:
    minimax = JensenMinimaxPartitioner().compute(partitions + 1)
    uniform = JensenUniformPartitioner().compute(partitions + 1)

    assert minimax.error <= uniform.error + 1e-12


@pytest.mark.parametrize("partitioner_type", [JensenUniformPartitioner, JensenMinimaxPartitioner])
def test_error_decreases_as_partition_count_increases(partitioner_type) -> None:
    errors = [partitioner_type().compute(partitions + 1).error for partitions in range(1, 7)]

    assert np.all(np.diff(errors) < 0.0)


def test_uniform_stable_cdf_difference_handles_infinite_and_extreme_intervals() -> None:
    partitioner = JensenUniformPartitioner

    assert partitioner._cdf_diff_stable(-np.inf, np.inf) == 1.0
    assert np.isclose(partitioner._cdf_diff_stable(-np.inf, 0.0), 0.5)
    assert np.isclose(partitioner._cdf_diff_stable(0.0, np.inf), 0.5)
    assert partitioner._cdf_diff_stable(8.0, 8.000001) > 0.0
    assert partitioner._cdf_diff_stable(-1.0, 1.0) > 0.0


def test_uniform_stable_cdf_difference_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    partitioner = JensenUniformPartitioner
    original_phi = partitioner._phi

    monkeypatch.setattr(partitioner, "_Phi", classmethod(lambda cls, value: 0.5))
    monkeypatch.setattr(partitioner, "_phi", staticmethod(original_phi))

    assert partitioner._cdf_diff_stable(1.0, 1.1) > 0.0
    assert partitioner._cdf_diff_stable(-1.0, -0.9) > 0.0


def test_uniform_stable_density_difference_handles_tails_and_signs() -> None:
    partitioner = JensenUniformPartitioner

    assert partitioner._density_diff_stable(-np.inf, 0.0) < 0.0
    assert partitioner._density_diff_stable(0.0, np.inf) > 0.0
    assert partitioner._density_diff_stable(0.5, 1.0) > 0.0
    assert partitioner._density_diff_stable(-1.0, -0.5) < 0.0


def test_uniform_truncated_mean_handles_whole_and_one_sided_tails() -> None:
    partitioner = JensenUniformPartitioner

    assert partitioner._truncated_mean(-np.inf, np.inf, 1.0) == 0.0
    assert partitioner._truncated_mean(-np.inf, 0.0, 0.5) < 0.0
    assert partitioner._truncated_mean(0.0, np.inf, 0.5) > 0.0
    assert np.isfinite(partitioner._truncated_mean(-1.0, 1.0, 0.68))


def test_uniform_self_test_passes() -> None:
    assert JensenUniformPartitioner.self_test()
