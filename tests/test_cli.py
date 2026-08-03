from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import gamma, norm, poisson

from folf import FirstOrderLossFunction, PiecewiseFirstOrderLossFunction
from folf.cli import main
from folf.utilities.probability.sampling import SAMPLING


def test_cli_plot_loss(tmp_path: Path) -> None:
    out_file = tmp_path / "plot.png"
    exit_code = main_with_args(
        [
            "plot-loss",
            "--distribution",
            "poisson:20",
            "--distribution",
            "norm:8:2",
            "--distribution",
            "gamma:4:1.5",
            "--samples",
            "300",
            "--x-min",
            "10",
            "--x-max",
            "50",
            "--precision",
            "1",
            "--piecewise-masses",
            "0.25,0.25,0.25,0.25",
            "--output",
            str(out_file),
        ]
    )
    assert exit_code == 0
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_complementary_piecewise_is_lower_bound_with_correct_tail_slope() -> None:
    distributions = [poisson(20), norm(8, 2), gamma(a=4, scale=1.5)]
    masses = np.array([0.25, 0.25, 0.25, 0.25])
    samples = 2_000
    model = FirstOrderLossFunction(distributions, sampling_strategy=SAMPLING.LHS)
    piecewise = PiecewiseFirstOrderLossFunction(distributions, sampling_strategy=SAMPLING.LHS)
    conditional_means = piecewise.get_conditional_expectations(masses, samples)
    xs = np.arange(10.0, 60.5, 0.5)

    empirical = np.array(
        [model.get_complementary_first_order_loss_function_value(x, samples) for x in xs]
    )
    lower_bound = np.array(
        [
            max(
                piecewise.get_piecewise_complementary_first_order_loss_function_value(
                    segment,
                    x,
                    masses,
                    conditional_means,
                )
                for segment in range(len(masses) + 1)
            )
            for x in xs
        ]
    )

    assert np.all(lower_bound <= empirical + 1e-12)
    assert np.isclose(lower_bound[-1] - lower_bound[-2], xs[-1] - xs[-2])


def test_piecewise_conditional_means_preserve_sample_mean() -> None:
    distributions = [poisson(20), norm(8, 2), gamma(a=4, scale=1.5)]
    masses = np.array([0.1, 0.2, 0.3, 0.4])
    samples = 2_000
    piecewise = PiecewiseFirstOrderLossFunction(distributions, sampling_strategy=SAMPLING.LHS)

    conditional_means = piecewise.get_conditional_expectations(masses, samples)
    empirical_mean = float(np.mean(piecewise.get_empirical_distribution(samples)))

    assert np.all(np.diff(conditional_means) >= 0.0)
    assert np.isclose(float(masses @ conditional_means), empirical_mean, atol=1e-10)


def test_regular_piecewise_is_lower_bound_with_correct_left_tail_slope() -> None:
    distributions = [poisson(20), norm(8, 2), gamma(a=4, scale=1.5)]
    masses = np.array([0.25, 0.25, 0.25, 0.25])
    samples = 2_000
    model = FirstOrderLossFunction(distributions, sampling_strategy=SAMPLING.LHS)
    piecewise = PiecewiseFirstOrderLossFunction(distributions, sampling_strategy=SAMPLING.LHS)
    conditional_means = piecewise.get_conditional_expectations(masses, samples)
    xs = np.arange(10.0, 60.5, 0.5)

    empirical = np.array([model.get_first_order_loss_function_value(x, samples) for x in xs])
    lower_bound = np.array(
        [
            max(
                piecewise.get_piecewise_first_order_loss_function_value(
                    segment,
                    x,
                    masses,
                    conditional_means,
                )
                for segment in range(len(masses))
            )
            for x in xs
        ]
    )

    assert np.all(lower_bound <= empirical + 1e-12)
    assert np.isclose(lower_bound[1] - lower_bound[0], -(xs[1] - xs[0]))


def main_with_args(args: list[str]) -> int:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["folf-cli", *args]
        return main()
    finally:
        sys.argv = old_argv
