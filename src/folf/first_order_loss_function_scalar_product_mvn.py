import numpy as np
from scipy.stats import norm


class FirstOrderLossFunctionScalarProductMVN:
    """Closed-form FOLF for scalar product with multivariate normal random vector."""

    def __init__(
        self,
        mean: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
        independent_demand: bool = False,
        distribution: object | None = None,
    ):
        if distribution is not None:
            # Java accepts a MultiNormalDist object; support a duck-typed equivalent.
            if hasattr(distribution, "getMean") and hasattr(distribution, "getCovariance"):
                mean = np.asarray(distribution.getMean(), dtype=float)
                covariance = np.asarray(distribution.getCovariance(), dtype=float)
            elif hasattr(distribution, "mean") and hasattr(distribution, "cov"):
                mean = np.asarray(distribution.mean(), dtype=float)
                covariance = np.asarray(distribution.cov(), dtype=float)
            else:
                msg = "distribution must expose mean/covariance accessors"
                raise ValueError(msg)

        if mean is None or covariance is None:
            msg = "mean and covariance are required"
            raise ValueError(msg)

        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.independent_demand = independent_demand

    def get_complementary_first_order_loss_function_value(self, y: float, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        mu = float(self.mean @ x)

        if self.independent_demand:
            variance = float(np.sum(x**2 * np.diag(self.covariance)))
        else:
            variance = float(x @ self.covariance @ x)

        sigma = float(np.sqrt(variance))
        z = (y - mu) / sigma
        return float(sigma * norm.pdf(z) + (y - mu) * norm.cdf(z))

    def get_first_order_loss_function_value(self, y: float, x: np.ndarray) -> float:
        mu = float(np.asarray(x, dtype=float) @ self.mean)
        return self.get_complementary_first_order_loss_function_value(y, x) - (y - mu)
