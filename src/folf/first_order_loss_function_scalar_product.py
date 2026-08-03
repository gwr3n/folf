from collections.abc import Sequence

import numpy as np
from scipy import stats

from .first_order_loss_function import FirstOrderLossFunction
from .utilities.probability.sampling import SAMPLING


class FirstOrderLossFunctionScalarProduct:
    """FOLF for scalar products between sampled random vectors and a given weight vector."""

    def __init__(
        self,
        distributions: Sequence[stats.rv_continuous | stats.rv_discrete],
        sampling_strategy: SAMPLING = SAMPLING.SRS,
    ):
        self._base = FirstOrderLossFunction(
            distributions,
            sampling_strategy=sampling_strategy,
        )

    def sample(self, nb_samples: int) -> np.ndarray:
        return self._base.sample(nb_samples)

    def get_empirical_distribution(
        self,
        nb_samples: int,
        x: np.ndarray,
    ) -> np.ndarray:
        sample_matrix = self.sample(nb_samples)
        observations = sample_matrix @ x
        observations.sort()
        return observations

    def get_complementary_first_order_loss_function_value(
        self,
        y: float,
        nb_samples: int,
        x: np.ndarray,
    ) -> float:
        obs = self.get_empirical_distribution(nb_samples, x)
        return float(np.mean(np.maximum(y - obs, 0.0)))

    def get_first_order_loss_function_value(
        self,
        y: float,
        nb_samples: int,
        x: np.ndarray,
    ) -> float:
        obs = self.get_empirical_distribution(nb_samples, x)
        return float(np.mean(np.maximum(obs - y, 0.0)))
