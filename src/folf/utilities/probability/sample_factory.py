from collections.abc import Sequence

import numpy as np
from scipy import stats


class SampleFactory:
    """Sampling utilities analogous to the Java SampleFactory."""

    @staticmethod
    def get_next_simple_random_sample(
        distributions: Sequence[stats.rv_continuous | stats.rv_discrete],
        samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        out = np.empty((samples, len(distributions)), dtype=float)
        for j, distribution in enumerate(distributions):
            u = rng.random(samples)
            out[:, j] = distribution.ppf(u)
        return out

    @staticmethod
    def get_next_lh_sample(
        distributions: Sequence[stats.rv_continuous | stats.rv_discrete],
        samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        if not distributions:
            return np.empty((samples, 0), dtype=float)

        dim = len(distributions)
        out = np.empty((samples, dim), dtype=float)

        for j, distribution in enumerate(distributions):
            base = np.arange(samples, dtype=float) / samples
            u = base + rng.random(samples) / samples
            u = SampleFactory._shuffle_like_java(u, rng)
            out[:, j] = distribution.ppf(u)

        return out

    @staticmethod
    def _shuffle_like_java(sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # Java version swaps each i with a uniformly random j in [0, n-1].
        out = sample.copy()
        n = len(out)
        for i in range(n):
            j = int(rng.integers(0, n))
            out[i], out[j] = out[j], out[i]
        return out
