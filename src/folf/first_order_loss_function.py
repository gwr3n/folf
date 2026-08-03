from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy import stats

from .utilities.probability.sample_factory import SampleFactory
from .utilities.probability.sampling import SAMPLING


class FirstOrderLossFunction:
    """Empirical first-order loss function over independent distributions."""

    seed = (12345, 24513, 24531, 42531, 35124, 32451)

    def __init__(
        self,
        distributions: Sequence[stats.rv_continuous | stats.rv_discrete],
        sampling_strategy: SAMPLING = SAMPLING.SRS,
    ):
        self.distributions = tuple(distributions)
        self.sampling_strategy = sampling_strategy

    def _new_rng(self) -> np.random.Generator:
        # Java resets the stream before each sampling call.
        return np.random.default_rng(np.array(self.seed, dtype=np.uint64).sum())

    def sample(self, nb_samples: int) -> np.ndarray:
        rng = self._new_rng()
        if self.sampling_strategy == SAMPLING.LHS:
            return SampleFactory.get_next_lh_sample(self.distributions, nb_samples, rng)
        return SampleFactory.get_next_simple_random_sample(
            self.distributions,
            nb_samples,
            rng,
        )

    def get_empirical_distribution(self, nb_samples: int) -> np.ndarray:
        observations = self.sample(nb_samples).sum(axis=1)
        observations.sort()
        return observations

    def get_complementary_first_order_loss_function_value(self, x: float, nb_samples: int) -> float:
        obs = self.get_empirical_distribution(nb_samples)
        return float(np.mean(np.maximum(x - obs, 0.0)))

    def get_first_order_loss_function_value(self, x: float, nb_samples: int) -> float:
        obs = self.get_empirical_distribution(nb_samples)
        return float(np.mean(np.maximum(obs - x, 0.0)))

    def get_distribution_xy_series(
        self,
        nb_samples: int,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del precision  # Kept for Java API compatibility.
        obs = self.get_empirical_distribution(nb_samples)
        xs = np.unique(obs)
        ys = np.searchsorted(obs, xs, side="right") / len(obs)
        return xs, ys

    def get_complementary_first_order_loss_function_series(
        self, min_x: float, max_x: float, nb_samples: int, precision: float
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(min_x, max_x + precision, precision)
        ys = np.array(
            [self.get_complementary_first_order_loss_function_value(x, nb_samples) for x in xs]
        )
        return xs, ys

    def get_complementary_first_order_loss_function_xy_series(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.get_complementary_first_order_loss_function_series(
            min_x,
            max_x,
            nb_samples,
            precision,
        )

    def get_first_order_loss_function_series(
        self, min_x: float, max_x: float, nb_samples: int, precision: float
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(min_x, max_x + precision, precision)
        ys = np.array([self.get_first_order_loss_function_value(x, nb_samples) for x in xs])
        return xs, ys

    def get_first_order_loss_function_xy_series(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.get_first_order_loss_function_series(min_x, max_x, nb_samples, precision)

    def plot_empirical_distribution(
        self,
        nb_samples: int,
        precision: float,
        save_to_disk: bool,
    ) -> None:
        xs, ys = self.get_distribution_xy_series(nb_samples, precision)
        self._plot_series(
            xs,
            ys,
            "Empirical distribution",
            "Support",
            "Frequency",
            save_to_disk,
            "emp_graph.csv",
        )

    def plot_empirical_complementary_first_order_loss_function(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        precision: float,
        save_to_disk: bool,
    ) -> None:
        xs, ys = self.get_complementary_first_order_loss_function_xy_series(
            min_x,
            max_x,
            nb_samples,
            precision,
        )
        self._plot_series(
            xs,
            ys,
            "Empirical Complementary First Order Loss Function",
            "x",
            "CL(x)",
            save_to_disk,
            "cfolf_graph.csv",
        )

    def plot_empirical_first_order_loss_function(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        precision: float,
        save_to_disk: bool,
    ) -> None:
        xs, ys = self.get_first_order_loss_function_xy_series(min_x, max_x, nb_samples, precision)
        self._plot_series(
            xs,
            ys,
            "Empirical First Order Loss Function",
            "x",
            "L(x)",
            save_to_disk,
            "folf_graph.csv",
        )

    @staticmethod
    def _plot_series(
        xs: np.ndarray,
        ys: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        save_to_disk: bool,
        file_name: str,
    ) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plt.plot(xs, ys)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.tight_layout()
            plt.close()
        except Exception:
            pass

        if save_to_disk:
            latex_folder = Path("./latex")
            latex_folder.mkdir(exist_ok=True)
            out_path = latex_folder / file_name
            data = np.column_stack((xs, ys))
            np.savetxt(out_path, data, delimiter=",", header="x,y", comments="")
