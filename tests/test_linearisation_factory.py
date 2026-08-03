import math

import pytest

from folf import JensenMinimaxPartitioner, LinearisationFactory


def _amax(probabilities, conditional_means) -> float:
    return max(
        float(sum(probabilities[k] * conditional_means[k] for k in range(i, len(probabilities))))
        for i in range(len(probabilities))
    )


def test_rejects_non_positive_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon must be positive"):
        LinearisationFactory.choose_linearisation_parameters(0.0, 4.0, 10.0)

    with pytest.raises(ValueError, match="epsilon must be positive"):
        LinearisationFactory.choose_linearisation_parameters(-1.0, 4.0, 10.0)


def test_selected_loss_partition_count_is_feasible_and_minimal() -> None:
    epsilon = 5.0
    vmax = 4.0
    cost = 10.0
    segments, _ = LinearisationFactory.choose_linearisation_parameters(epsilon, vmax, cost)
    partitioner = JensenMinimaxPartitioner()
    rhs_loss = epsilon / (2.0 * cost * math.sqrt(vmax))

    selected = partitioner.compute(segments)
    assert selected.error <= rhs_loss

    partitions = segments - 1
    if partitions > 1:
        previous = partitioner.compute(segments - 1)
        assert previous.error > rhs_loss


def test_selected_sqrt_partition_count_is_feasible_and_minimal() -> None:
    epsilon = 5.0
    vmax = 4.0
    cost = 10.0
    segments, q = LinearisationFactory.choose_linearisation_parameters(epsilon, vmax, cost)
    result = JensenMinimaxPartitioner().compute(segments)
    rhs_delta = epsilon / (2.0 * cost * (_amax(result.p, result.expect) + result.error))

    assert math.sqrt(vmax / q) / 4.0 <= rhs_delta + 1e-12
    if q > 1:
        assert math.sqrt(vmax / (q - 1)) / 4.0 > rhs_delta


def test_tighter_tolerance_never_reduces_linearisation_sizes() -> None:
    loose_segments, loose_q = LinearisationFactory.choose_linearisation_parameters(5.0, 4.0, 10.0)
    tight_segments, tight_q = LinearisationFactory.choose_linearisation_parameters(2.0, 4.0, 10.0)

    assert tight_segments >= loose_segments
    assert tight_q >= loose_q


def test_parameter_selection_is_deterministic() -> None:
    first = LinearisationFactory.choose_linearisation_parameters(5.0, 4.0, 10.0)
    second = LinearisationFactory.choose_linearisation_parameters(5.0, 4.0, 10.0)

    assert first == second
