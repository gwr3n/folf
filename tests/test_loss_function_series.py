from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm

from folf import FirstOrderLossFunction, PiecewiseFirstOrderLossFunction
from folf.utilities.probability.sampling import SAMPLING


def test_first_order_loss_series_and_xy_aliases() -> None:
    model = FirstOrderLossFunction([norm(0.0, 1.0)], sampling_strategy=SAMPLING.LHS)

    distribution_x, distribution_y = model.get_distribution_xy_series(200, precision=0.1)
    complementary_x, complementary_y = model.get_complementary_first_order_loss_function_series(
        -2.0, 2.0, 200, 0.5
    )
    regular_x, regular_y = model.get_first_order_loss_function_series(-2.0, 2.0, 200, 0.5)

    assert np.all(np.diff(distribution_x) > 0.0)
    assert np.all(np.diff(distribution_y) > 0.0)
    assert 0.0 < distribution_y[0] <= distribution_y[-1] == 1.0
    assert complementary_x.shape == complementary_y.shape == (9,)
    assert regular_x.shape == regular_y.shape == (9,)
    assert np.all(complementary_y >= 0.0)
    assert np.all(regular_y >= 0.0)

    alias_x, alias_y = model.get_complementary_first_order_loss_function_xy_series(
        -2.0, 2.0, 200, 0.5
    )
    assert np.array_equal(alias_x, complementary_x)
    assert np.array_equal(alias_y, complementary_y)

    alias_x, alias_y = model.get_first_order_loss_function_xy_series(-2.0, 2.0, 200, 0.5)
    assert np.array_equal(alias_x, regular_x)
    assert np.array_equal(alias_y, regular_y)


def test_first_order_loss_plot_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model = FirstOrderLossFunction([norm(0.0, 1.0)])

    model.plot_empirical_distribution(100, 0.25, save_to_disk=True)
    model.plot_empirical_complementary_first_order_loss_function(
        -2.0, 2.0, 100, 0.5, save_to_disk=True
    )
    model.plot_empirical_first_order_loss_function(
        -2.0, 2.0, 100, 0.5, save_to_disk=True
    )

    for file_name in ("emp_graph.csv", "cfolf_graph.csv", "folf_graph.csv"):
        output = tmp_path / "latex" / file_name
        assert output.exists()
        assert output.read_text(encoding="utf-8").startswith("x,y")


def test_piecewise_rejects_invalid_probability_masses() -> None:
    model = PiecewiseFirstOrderLossFunction([norm(0.0, 1.0)])

    with pytest.raises(ValueError, match="positive and sum to 1.0"):
        model.get_conditional_expectations(np.array([0.5, 0.4]), 100)

    with pytest.raises(ValueError, match="positive and sum to 1.0"):
        model.get_conditional_expectations(np.array([0.5, 0.5, 0.0]), 100)


def test_piecewise_values_errors_and_series() -> None:
    model = PiecewiseFirstOrderLossFunction([norm(0.0, 1.0)], sampling_strategy=SAMPLING.LHS)
    masses = np.array([0.25, 0.5, 0.25])
    conditional_means = model.get_conditional_expectations(masses, 400)

    complementary = model.get_piecewise_complementary_first_order_loss_function_value(
        2, 0.5, masses, conditional_means
    )
    regular = model.get_piecewise_first_order_loss_function_value(
        1, 0.5, masses, conditional_means
    )
    assert complementary >= 0.0
    assert np.isclose(
        regular,
        float(np.sum((conditional_means[1:] - 0.5) * masses[1:])),
    )

    complementary_error = model.get_piecewise_complementary_first_order_loss_function_error_value(
        0.5, 400, masses, conditional_means
    )
    regular_error = model.get_piecewise_first_order_loss_function_error_value(
        0.5, 400, masses, conditional_means
    )
    assert complementary_error >= -1e-12
    assert regular_error >= -1e-12
    assert model.get_max_approximation_error(masses, 400) >= 0.0

    comp_x, comp_y = (
        model.get_piecewise_complementary_first_order_loss_function_xy_series_for_segment(
            2, masses, conditional_means, -2.0, 2.0, -10.0, 0.5
        )
    )
    reg_x, reg_y = model.get_piecewise_first_order_loss_function_xy_series_for_segment(
        1, masses, conditional_means, -2.0, 2.0, -10.0, 0.5
    )
    assert comp_x.shape == comp_y.shape == (9,)
    assert reg_x.shape == reg_y.shape == (9,)

    comp_error_x, comp_error_y = (
        model.get_piecewise_complementary_first_order_loss_function_error_xy_series(
            -2.0, 2.0, 400, masses, conditional_means, 0.5
        )
    )
    reg_error_x, reg_error_y = model.get_piecewise_first_order_loss_function_error_xy_series(
        -2.0, 2.0, 400, masses, conditional_means, 0.5
    )
    assert comp_error_x.shape == comp_error_y.shape == (9,)
    assert reg_error_x.shape == reg_error_y.shape == (9,)


def test_piecewise_plot_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model = PiecewiseFirstOrderLossFunction([norm(0.0, 1.0)])
    masses = np.array([0.5, 0.5])

    model.plot_piecewise_complementary_first_order_loss_function(
        -2.0, 2.0, -1.0, masses, 200, 0.5, save_to_disk=True
    )
    model.plot_piecewise_first_order_loss_function(
        -2.0, 2.0, -1.0, masses, 200, 0.5, save_to_disk=True
    )

    expected = (
        "pw_cfolf_graph.csv",
        "pw_cfolf_segments.csv",
        "pw_folf_graph.csv",
        "pw_folf_segments.csv",
    )
    for file_name in expected:
        output = tmp_path / "latex" / file_name
        assert output.exists()
        assert output.stat().st_size > 0


def test_to_primitive_conversions() -> None:
    assert PiecewiseFirstOrderLossFunction.to_primitive(None) is None
    assert np.array_equal(PiecewiseFirstOrderLossFunction.to_primitive([]), np.array([]))
    assert np.array_equal(
        PiecewiseFirstOrderLossFunction.to_primitive([1, 2.5]),
        np.array([1.0, 2.5]),
    )
