import math

from .jensen_minimax_partitioner import JensenMinimaxPartitioner
from .jensen_partitioner import JensenPartitioner
from .jensen_uniform_partitioner import JensenUniformPartitioner
from .partitioner import PARTITIONER
from .piecewise_standard_normal_first_order_loss_function import (
    PiecewiseStandardNormalFirstOrderLossFunction,
)


class LinearisationFactory:
    @staticmethod
    def choose_linearisation_parameters(epsilon: float, vmax: float, c: float) -> tuple[int, int]:
        if epsilon <= 0.0:
            msg = "epsilon must be positive"
            raise ValueError(msg)

        partitioner: JensenPartitioner
        if PiecewiseStandardNormalFirstOrderLossFunction.get_partitioner() == PARTITIONER.UNIFORM:
            partitioner = JensenUniformPartitioner()
        else:
            partitioner = JensenMinimaxPartitioner()

        smax = math.sqrt(vmax)
        rhs_loss = epsilon / (2.0 * c * smax)

        w = 1
        part = partitioner.compute(w + 1)
        while part.error > rhs_loss:
            w <<= 1
            part = partitioner.compute(w + 1)
            if w > (1 << 22):
                msg = "W exploded (> 4,000,000)"
                raise RuntimeError(msg)

        low, high = w >> 1, w
        while low + 1 < high:
            mid = (low + high) >> 1
            mid_part = partitioner.compute(mid + 1)
            if mid_part.error <= rhs_loss:
                high = mid
                part = mid_part
            else:
                low = mid

        w = high
        dot_err = part.error
        w_segments = w + 1

        p = part.p
        mu = part.expect
        amax = 0.0
        for i in range(w):
            s = float(sum(p[k] * mu[k] for k in range(i, w)))
            amax = max(amax, s)

        rhs_delta = epsilon / (2.0 * c * (amax + dot_err))
        q_min = vmax / (16.0 * rhs_delta * rhs_delta)
        q = max(1, math.ceil(q_min))

        return w_segments, q
