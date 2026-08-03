from pathlib import Path

import numpy as np

from .first_order_loss_function import FirstOrderLossFunction


class PiecewiseFirstOrderLossFunction(FirstOrderLossFunction):
    """Piecewise linear approximations of FOLF and complementary FOLF."""

    def get_conditional_expectations(
        self,
        probability_masses: np.ndarray,
        nb_samples: int,
    ) -> np.ndarray:
        pm = np.asarray(probability_masses, dtype=float)
        if np.any(pm <= 0.0) or not np.isclose(np.sum(pm), 1.0, atol=1e-12):
            msg = "probability masses must be positive and sum to 1.0"
            raise ValueError(msg)

        obs = self.get_empirical_distribution(nb_samples)
        weighted_sums = np.zeros(pm.shape[0], dtype=float)
        segment_index = 0
        segment_remaining = float(pm[0])
        observation_mass = 1.0 / len(obs)

        for value in obs:
            mass_remaining = observation_mass
            while mass_remaining > 1e-15 and segment_index < len(pm):
                allocated_mass = min(mass_remaining, segment_remaining)
                weighted_sums[segment_index] += float(value) * allocated_mass
                mass_remaining -= allocated_mass
                segment_remaining -= allocated_mass

                if segment_remaining <= 1e-15:
                    segment_index += 1
                    if segment_index < len(pm):
                        segment_remaining = float(pm[segment_index])

        return weighted_sums / pm

    def _approximation_errors(self, probability_masses: np.ndarray, nb_samples: int) -> np.ndarray:
        pm = np.asarray(probability_masses, dtype=float)
        ce = self.get_conditional_expectations(pm, nb_samples)
        out = np.zeros_like(ce)
        for i in range(len(pm)):
            out[i] = (
                self.get_complementary_first_order_loss_function_value(
                    float(ce[i]),
                    nb_samples,
                )
                - self.get_piecewise_complementary_first_order_loss_function_value(
                    i,
                    float(ce[i]),
                    pm,
                    ce,
                )
            )
        return out

    def get_max_approximation_error(self, probability_masses: np.ndarray, nb_samples: int) -> float:
        return float(
            np.max(
                self._approximation_errors(
                    np.asarray(probability_masses, dtype=float),
                    nb_samples,
                )
            )
        )

    @staticmethod
    def get_piecewise_complementary_first_order_loss_function_value(
        segment_index: int,
        x: float,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
    ) -> float:
        value = 0.0
        for i in range(segment_index):
            value += (x - conditional_expectations[i]) * probability_masses[i]
        return float(value)

    @staticmethod
    def get_piecewise_first_order_loss_function_value(
        segment_index: int,
        x: float,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
    ) -> float:
        value = 0.0
        for i in range(segment_index, len(probability_masses)):
            value += (conditional_expectations[i] - x) * probability_masses[i]
        return float(value)

    def get_piecewise_complementary_first_order_loss_function_error_value(
        self,
        x: float,
        nb_samples: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
    ) -> float:
        loss_value = self.get_complementary_first_order_loss_function_value(x, nb_samples)

        max_value = 0.0
        for j in range(len(probability_masses) + 1):
            value = 0.0
            for i in range(j):
                value += (x - conditional_expectations[i]) * probability_masses[i]
            max_value = max(max_value, value)

        return float(loss_value - max_value)

    def get_piecewise_first_order_loss_function_error_value(
        self,
        x: float,
        nb_samples: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
    ) -> float:
        loss_value = self.get_first_order_loss_function_value(x, nb_samples)

        max_value = 0.0
        for j in range(len(probability_masses)):
            value = 0.0
            for i in range(j, len(probability_masses)):
                value += (conditional_expectations[i] - x) * probability_masses[i]
            max_value = max(max_value, value)

        return float(loss_value - max_value)

    def get_piecewise_complementary_first_order_loss_function_xy_series_for_segment(
        self,
        segment_index: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
        min_x: float,
        max_x: float,
        min_y_value: float,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        x = min_x
        while x <= max_x:
            value = self.get_piecewise_complementary_first_order_loss_function_value(
                segment_index,
                x,
                probability_masses,
                conditional_expectations,
            )
            if value >= min_y_value:
                xs.append(x)
                ys.append(value)
            x += precision
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def get_piecewise_first_order_loss_function_xy_series_for_segment(
        self,
        segment_index: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
        min_x: float,
        max_x: float,
        min_y_value: float,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs: list[float] = []
        ys: list[float] = []
        x = min_x
        while x <= max_x:
            value = self.get_piecewise_first_order_loss_function_value(
                segment_index,
                x,
                probability_masses,
                conditional_expectations,
            )
            if value >= min_y_value:
                xs.append(x)
                ys.append(value)
            x += precision
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def get_piecewise_complementary_first_order_loss_function_error_xy_series(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(min_x, max_x + precision, precision)
        ys = np.array(
            [
                self.get_piecewise_complementary_first_order_loss_function_error_value(
                    float(x),
                    nb_samples,
                    probability_masses,
                    conditional_expectations,
                )
                for x in xs
            ]
        )
        return xs, ys

    def get_piecewise_first_order_loss_function_error_xy_series(
        self,
        min_x: float,
        max_x: float,
        nb_samples: int,
        probability_masses: np.ndarray,
        conditional_expectations: np.ndarray,
        precision: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(min_x, max_x + precision, precision)
        ys = np.array(
            [
                self.get_piecewise_first_order_loss_function_error_value(
                    float(x),
                    nb_samples,
                    probability_masses,
                    conditional_expectations,
                )
                for x in xs
            ]
        )
        return xs, ys

    def plot_piecewise_complementary_first_order_loss_function(
        self,
        min_x: float,
        max_x: float,
        min_y_value: float,
        probability_masses: np.ndarray,
        nb_samples: int,
        precision: float,
        save_to_disk: bool,
    ) -> None:
        ce = self.get_conditional_expectations(probability_masses, nb_samples)
        xs, ys = self.get_complementary_first_order_loss_function_xy_series(
            min_x,
            max_x,
            nb_samples,
            precision,
        )
        self._plot_multi_series([(xs, ys)], "pw_cfolf_graph.csv", save_to_disk)

        segment_count = len(probability_masses) + 1
        segment_series: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(segment_count):
            sx, sy = (
                self.get_piecewise_complementary_first_order_loss_function_xy_series_for_segment(
                i,
                probability_masses,
                ce,
                min_x,
                max_x,
                min_y_value,
                precision,
                )
            )
            segment_series.append((sx, sy))
        self._plot_multi_series(segment_series, "pw_cfolf_segments.csv", save_to_disk)

    def plot_piecewise_first_order_loss_function(
        self,
        min_x: float,
        max_x: float,
        min_y_value: float,
        probability_masses: np.ndarray,
        nb_samples: int,
        precision: float,
        save_to_disk: bool,
    ) -> None:
        ce = self.get_conditional_expectations(probability_masses, nb_samples)
        xs, ys = self.get_first_order_loss_function_xy_series(min_x, max_x, nb_samples, precision)
        self._plot_multi_series([(xs, ys)], "pw_folf_graph.csv", save_to_disk)

        segment_count = len(probability_masses) + 1
        segment_series: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(segment_count):
            sx, sy = self.get_piecewise_first_order_loss_function_xy_series_for_segment(
                i,
                probability_masses,
                ce,
                min_x,
                max_x,
                min_y_value,
                precision,
            )
            segment_series.append((sx, sy))
        self._plot_multi_series(segment_series, "pw_folf_segments.csv", save_to_disk)

    @staticmethod
    def _plot_multi_series(
        series: list[tuple[np.ndarray, np.ndarray]],
        file_name: str,
        save_to_disk: bool,
    ) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            for xs, ys in series:
                if len(xs) > 0:
                    plt.plot(xs, ys)
            plt.tight_layout()
            plt.close()
        except Exception:
            pass

        if save_to_disk:
            latex_folder = Path("./latex")
            latex_folder.mkdir(exist_ok=True)
            out_path = latex_folder / file_name
            with out_path.open("w", encoding="utf-8") as handle:
                handle.write("series,x,y\n")
                for s_idx, (xs, ys) in enumerate(series):
                    for x_val, y_val in zip(xs, ys, strict=False):
                        handle.write(f"{s_idx},{x_val},{y_val}\n")

    @staticmethod
    def to_primitive(array: list[float] | None) -> np.ndarray | None:
        if array is None:
            return None
        if len(array) == 0:
            return np.array([], dtype=float)
        return np.array([float(v) for v in array], dtype=float)
