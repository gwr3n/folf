from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, norm, poisson

from .first_order_loss_function import FirstOrderLossFunction
from .piecewise_first_order_loss_function import PiecewiseFirstOrderLossFunction
from .utilities.probability.sampling import SAMPLING


def _parse_distribution(spec: str):
    """Parse distribution specs like poisson:20, norm:8:2, gamma:4:1.5."""
    parts = spec.split(":")
    kind = parts[0].strip().lower()

    if kind == "poisson" and len(parts) == 2:
        return poisson(float(parts[1]))
    if kind == "norm" and len(parts) == 3:
        return norm(float(parts[1]), float(parts[2]))
    if kind == "gamma" and len(parts) == 3:
        return gamma(a=float(parts[1]), scale=float(parts[2]))

    msg = (
        f"Unsupported distribution spec '{spec}'. "
        "Use one of: poisson:<lambda>, norm:<mu>:<sigma>, gamma:<shape>:<scale>."
    )
    raise ValueError(msg)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="folf-cli",
        description="CLI for first-order loss function analysis and plotting.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plot_loss = subparsers.add_parser(
        "plot-loss",
        help="Plot empirical loss function and piecewise linearisation.",
    )
    plot_loss.add_argument(
        "--distribution",
        action="append",
        required=True,
        help=(
            "Distribution spec. Repeat this option for multiple distributions. "
            "Examples: poisson:20, norm:8:2, gamma:4:1.5"
        ),
    )
    plot_loss.add_argument("--sampling", choices=["SRS", "LHS"], default="SRS")
    plot_loss.add_argument("--samples", type=int, default=5000)
    plot_loss.add_argument("--x-min", type=float, required=True)
    plot_loss.add_argument("--x-max", type=float, required=True)
    plot_loss.add_argument("--precision", type=float, default=0.25)
    plot_loss.add_argument(
        "--piecewise-masses",
        type=str,
        default="0.25,0.25,0.25,0.25",
        help="Comma-separated probability masses (must sum to ~1).",
    )
    plot_loss.add_argument(
        "--loss-type",
        choices=["complementary", "regular"],
        default="complementary",
        help="Which loss function to plot.",
    )
    plot_loss.add_argument(
        "--output",
        type=str,
        default="folf_plot.png",
        help="Output image path.",
    )

    return parser


def _plot_loss(args: argparse.Namespace) -> int:
    distributions = [_parse_distribution(spec) for spec in args.distribution]
    sampling = SAMPLING[args.sampling]

    masses = np.array([float(x.strip()) for x in args.piecewise_masses.split(",")], dtype=float)
    if not np.isclose(np.sum(masses), 1.0, atol=1e-6):
        msg = f"piecewise masses must sum to 1.0, got {np.sum(masses)}"
        raise ValueError(msg)

    folf = FirstOrderLossFunction(distributions, sampling_strategy=sampling)
    pw = PiecewiseFirstOrderLossFunction(distributions, sampling_strategy=sampling)

    xs = np.arange(args.x_min, args.x_max + args.precision, args.precision)
    ce = pw.get_conditional_expectations(masses, args.samples)

    if args.loss_type == "complementary":
        ys = np.array(
            [
                folf.get_complementary_first_order_loss_function_value(float(x), args.samples)
                for x in xs
            ]
        )
        pw_ys = np.array(
            [
                max(
                    [
                        pw.get_piecewise_complementary_first_order_loss_function_value(
                            seg,
                            float(x),
                            masses,
                            ce,
                        )
                        for seg in range(len(masses) + 1)
                    ]
                )
                for x in xs
            ]
        )
        title = "Complementary First-Order Loss and Piecewise Linearisation"
        y_label = "CL(x)"
    else:
        ys = np.array(
            [folf.get_first_order_loss_function_value(float(x), args.samples) for x in xs]
        )
        pw_ys = np.array(
            [
                max(
                    [
                        pw.get_piecewise_first_order_loss_function_value(
                            seg,
                            float(x),
                            masses,
                            ce,
                        )
                        for seg in range(len(masses))
                    ]
                )
                for x in xs
            ]
        )
        title = "First-Order Loss and Piecewise Linearisation"
        y_label = "L(x)"

    plt.figure(figsize=(9, 5))
    plt.plot(xs, ys, label="Empirical loss", linewidth=2)
    plt.plot(xs, pw_ys, "--", label="Piecewise linearisation", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output)
    plt.close()

    print(f"Saved plot to {output}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "plot-loss":
        return _plot_loss(args)

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
