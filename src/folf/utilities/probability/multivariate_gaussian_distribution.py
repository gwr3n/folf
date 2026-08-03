import numpy as np
from scipy.linalg import cholesky
from scipy.stats import norm


class MultivariateGaussianDistribution:
    """Multivariate Gaussian distribution with Monte Carlo CDF approximation."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        if mean.shape[0] != covariance.shape[0] or covariance.shape[0] != covariance.shape[1]:
            msg = "Mean vector and covariance matrix have different dimension"
            raise ValueError(msg)

        self.mu = mean
        self.sigma = covariance
        self.diagonal = False
        self._dim = mean.shape[0]
        self._sigma_l = cholesky(covariance, lower=True)
        self._free_parameters = self._dim + self._dim * (self._dim + 1) // 2

    def length(self) -> int:
        return self._free_parameters

    def mean(self) -> np.ndarray:
        return self.mu

    def cov(self) -> np.ndarray:
        return self.sigma

    @staticmethod
    def sub(y: np.ndarray, x: np.ndarray) -> None:
        if len(x) != len(y):
            msg = f"Arrays have different length: x[{len(x)}], y[{len(y)}]"
            raise ValueError(msg)
        y -= x

    def cdf(self, x: np.ndarray) -> float:
        rng = np.random.default_rng(123456)
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self._dim:
            msg = "Sample has different dimension"
            raise ValueError(msg)

        nmax = 10_000
        alph = norm.ppf(0.999)
        err_max = 0.001

        v = x.copy()
        self.sub(v, self.mu)

        e = np.zeros(self._dim, dtype=float)
        f = np.zeros(self._dim, dtype=float)
        e[0] = norm.cdf(v[0] / self._sigma_l[0, 0])
        f[0] = e[0]

        y = np.zeros(self._dim, dtype=float)
        p = 0.0
        var_sum = 0.0
        err = 2 * err_max

        n = 1
        while err > err_max and n <= nmax:
            w = rng.random(self._dim - 1)
            for i in range(1, self._dim):
                y[i - 1] = norm.ppf(w[i - 1] * e[i - 1])
                q = float(np.dot(self._sigma_l[i, :i], y[:i]))
                e[i] = norm.cdf((v[i] - q) / self._sigma_l[i, i])
                f[i] = e[i] * f[i - 1]

            delta = (f[self._dim - 1] - p) / n
            p += delta
            var_sum = (n - 2) * var_sum / n + delta * delta
            err = alph * np.sqrt(var_sum)
            n += 1

        return float(p)
