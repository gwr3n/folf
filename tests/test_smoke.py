import numpy as np
from scipy.stats import poisson

from folf import (
    FirstOrderLossFunction,
    FirstOrderLossFunctionScalarProduct,
    FirstOrderLossFunctionScalarProductMVN,
    JensenMinimaxPartitioner,
    JensenUniformPartitioner,
    LinearisationFactory,
    PiecewiseStandardNormalFirstOrderLossFunction,
)
from folf.utilities.hash import SHA
from folf.utilities.probability.sampling import SAMPLING


def test_partitioners_shapes():
    res_u = JensenUniformPartitioner().compute(6)
    res_m = JensenMinimaxPartitioner().compute(6)
    assert len(res_u.p) == 5
    assert len(res_u.expect) == 5
    assert len(res_m.p) == 5
    assert len(res_m.expect) == 5
    assert res_u.error > 0
    assert res_m.error > 0


def test_folf_values_non_negative():
    folf = FirstOrderLossFunction(
        [poisson(20), poisson(5), poisson(50)],
        sampling_strategy=SAMPLING.SRS,
    )
    cval = folf.get_complementary_first_order_loss_function_value(70.0, 200)
    fval = folf.get_first_order_loss_function_value(70.0, 200)
    assert cval >= 0
    assert fval >= 0


def test_scalar_product_variants():
    folf_sp = FirstOrderLossFunctionScalarProduct([poisson(2), poisson(3)])
    vec = np.array([1.5, -0.5])
    val = folf_sp.get_first_order_loss_function_value(1.0, 100, vec)
    assert val >= 0

    mvn = FirstOrderLossFunctionScalarProductMVN(
        mean=np.array([1.0, 2.0]),
        covariance=np.array([[1.0, 0.2], [0.2, 2.0]]),
        independent_demand=False,
    )
    cval = mvn.get_complementary_first_order_loss_function_value(2.5, np.array([1.0, 1.0]))
    assert cval >= 0


def test_piecewise_cache_accessors_and_factory():
    err = PiecewiseStandardNormalFirstOrderLossFunction.get_error(5)
    probs = PiecewiseStandardNormalFirstOrderLossFunction.get_probabilities(5)
    means = PiecewiseStandardNormalFirstOrderLossFunction.get_means(5)
    assert err > 0
    assert probs.shape == (5,)
    assert means.shape == (5,)

    w_segments, q = LinearisationFactory.choose_linearisation_parameters(0.5, 4.0, 10.0)
    assert w_segments >= 2
    assert q >= 1


def test_sha_wrapper():
    assert SHA.generate_sha256("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
