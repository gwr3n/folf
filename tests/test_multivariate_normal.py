import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from folf import FirstOrderLossFunctionScalarProductMVN
from folf.utilities.probability import MultivariateGaussianDistribution


class _JavaStyleDistribution:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self._mean = mean
        self._covariance = covariance

    def getMean(self) -> np.ndarray:
        return self._mean

    def getCovariance(self) -> np.ndarray:
        return self._covariance


class _PythonStyleDistribution:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        self._mean = mean
        self._covariance = covariance

    def mean(self) -> np.ndarray:
        return self._mean

    def cov(self) -> np.ndarray:
        return self._covariance


def _expected_complementary_loss(y: float, mu: float, variance: float) -> float:
    sigma = np.sqrt(variance)
    z = (y - mu) / sigma
    return float(sigma * norm.pdf(z) + (y - mu) * norm.cdf(z))


def test_correlated_mvn_scalar_product_matches_univariate_projection() -> None:
    mean = np.array([4.0, 7.0, 2.0])
    covariance = np.array(
        [
            [4.0, 1.2, -0.4],
            [1.2, 9.0, 0.8],
            [-0.4, 0.8, 2.0],
        ]
    )
    weights = np.array([0.5, -0.25, 1.5])
    y = 3.25
    model = FirstOrderLossFunctionScalarProductMVN(mean, covariance, independent_demand=False)

    projected_mean = float(mean @ weights)
    projected_variance = float(weights @ covariance @ weights)
    expected = _expected_complementary_loss(y, projected_mean, projected_variance)

    assert np.isclose(
        model.get_complementary_first_order_loss_function_value(y, weights),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_independent_mvn_scalar_product_uses_squared_weights() -> None:
    mean = np.array([3.0, 5.0, 8.0])
    covariance = np.diag([4.0, 9.0, 16.0])
    weights = np.array([0.5, 2.0, -0.25])
    y = 10.0
    model = FirstOrderLossFunctionScalarProductMVN(mean, covariance, independent_demand=True)

    projected_mean = float(mean @ weights)
    projected_variance = float(np.sum(weights**2 * np.diag(covariance)))
    expected = _expected_complementary_loss(y, projected_mean, projected_variance)

    assert np.isclose(
        model.get_complementary_first_order_loss_function_value(y, weights),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_mvn_loss_identity_holds() -> None:
    mean = np.array([2.0, 6.0])
    covariance = np.array([[3.0, 0.7], [0.7, 5.0]])
    weights = np.array([1.25, -0.5])
    y = 1.75
    model = FirstOrderLossFunctionScalarProductMVN(mean, covariance, independent_demand=False)

    complementary = model.get_complementary_first_order_loss_function_value(y, weights)
    regular = model.get_first_order_loss_function_value(y, weights)
    projected_mean = float(mean @ weights)

    assert np.isclose(complementary - regular, y - projected_mean, atol=1e-12)
    assert complementary >= 0.0
    assert regular >= 0.0


def test_multivariate_gaussian_cdf_matches_scipy_reference() -> None:
    mean = np.array([1.0, -0.5])
    covariance = np.array([[2.0, 0.6], [0.6, 1.5]])
    point = np.array([1.5, 0.25])
    distribution = MultivariateGaussianDistribution(mean, covariance)

    actual = distribution.cdf(point)
    expected = float(multivariate_normal(mean=mean, cov=covariance).cdf(point))

    assert 0.0 <= actual <= 1.0
    assert abs(actual - expected) < 0.01


@pytest.mark.parametrize("distribution_type", [_JavaStyleDistribution, _PythonStyleDistribution])
def test_scalar_product_mvn_accepts_distribution_objects(distribution_type) -> None:
    mean = np.array([1.0, 3.0])
    covariance = np.array([[2.0, 0.4], [0.4, 5.0]])
    weights = np.array([0.75, -0.25])
    y = 0.5
    from_arrays = FirstOrderLossFunctionScalarProductMVN(mean, covariance)
    from_distribution = FirstOrderLossFunctionScalarProductMVN(
        distribution=distribution_type(mean, covariance)
    )

    assert np.isclose(
        from_distribution.get_complementary_first_order_loss_function_value(y, weights),
        from_arrays.get_complementary_first_order_loss_function_value(y, weights),
    )


def test_scalar_product_mvn_rejects_missing_or_unsupported_distribution() -> None:
    with pytest.raises(ValueError, match="mean and covariance are required"):
        FirstOrderLossFunctionScalarProductMVN()

    with pytest.raises(ValueError, match="distribution must expose"):
        FirstOrderLossFunctionScalarProductMVN(distribution=object())
