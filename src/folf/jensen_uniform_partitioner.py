import math

import numpy as np
from scipy.stats import norm

from .jensen_partitioner import JensenPartitioner
from .result import Result


class JensenUniformPartitioner(JensenPartitioner):
    """Uniform-probability Jensen partitioner for standard normal demand."""

    _LOG_SQRT_2PI = math.log(math.sqrt(2.0 * math.pi))

    @staticmethod
    def _phi(z: float) -> float:
        return float(norm.pdf(z))

    @staticmethod
    def _Phi(z: float) -> float:
        return float(norm.cdf(z))

    @staticmethod
    def _barPhi(z: float) -> float:
        return float(norm.cdf(-z))

    @classmethod
    def _log_phi(cls, z: float) -> float:
        return -0.5 * z * z - cls._LOG_SQRT_2PI

    @classmethod
    def _cdf_diff_stable(cls, a: float, b: float) -> float:
        if a == -np.inf:
            return 1.0 if b == np.inf else cls._Phi(b)
        if b == np.inf:
            return 1.0 if a == -np.inf else cls._barPhi(a)

        if a >= 0.0 and b >= 0.0:
            d = cls._Phi(-a) - cls._Phi(-b)
            if d <= 0.0:
                d = 0.0
            if d < 1e-16:
                m = 0.5 * (a + b)
                return (b - a) / 6.0 * (cls._phi(a) + 4.0 * cls._phi(m) + cls._phi(b))
            return d

        d = cls._Phi(b) - cls._Phi(a)
        if d <= 0.0:
            m = 0.5 * (a + b)
            return max(0.0, (b - a) / 6.0 * (cls._phi(a) + 4.0 * cls._phi(m) + cls._phi(b)))
        return d

    @classmethod
    def _density_diff_stable(cls, a: float, b: float) -> float:
        if a == -np.inf:
            return -cls._phi(b)
        if b == np.inf:
            return cls._phi(a)

        la = cls._log_phi(a)
        lb = cls._log_phi(b)
        if la >= lb:
            delta = lb - la
            return -math.exp(la) * math.expm1(delta)

        delta = la - lb
        return math.exp(lb) * math.expm1(delta)

    @classmethod
    def _mills_right(cls, a: float) -> float:
        num_log = cls._log_phi(a)
        den = cls._barPhi(a)
        return math.exp(num_log) / den

    @classmethod
    def _mills_left(cls, b: float) -> float:
        num_log = cls._log_phi(b)
        den = cls._Phi(b)
        return math.exp(num_log) / den

    @classmethod
    def _truncated_mean(cls, a: float, b: float, p: float) -> float:
        if a == -np.inf:
            if b == np.inf:
                return 0.0
            return -cls._mills_left(b)

        if b == np.inf:
            return cls._mills_right(a)

        num = cls._density_diff_stable(a, b)
        return num / p

    def compute(self, segments: int) -> Result:
        if segments < 2:
            msg = "segments must be >= 2"
            raise ValueError(msg)

        n = segments - 1
        if n == 1:
            return Result(
                p=np.array([1.0]),
                expect=np.array([0.0]),
                error=1.0 / math.sqrt(2.0 * math.pi),
            )

        breakpoints = norm.ppf(np.arange(1, n) / n)

        a = np.empty(n)
        b = np.empty(n)
        a[0] = -np.inf
        if n > 1:
            a[1:] = breakpoints
            b[:-1] = breakpoints
        b[-1] = np.inf

        p = np.empty(n)
        expect = np.empty(n)

        for i in range(n):
            if i == 0:
                p[i] = self._Phi(b[i])
            elif i == n - 1:
                p[i] = self._barPhi(a[i])
            else:
                p[i] = self._cdf_diff_stable(a[i], b[i])

            if p[i] < 0.0:
                p[i] = 0.0

            expect[i] = self._truncated_mean(a[i], b[i], p[i])

        if abs(float(np.sum(p)) - 1.0) > 1e-14:
            p = p / np.sum(p)

        err = 0.0
        for i in range(n):
            phi_b = 0.0 if i == n - 1 else self._phi(b[i])
            phi_big = 1.0 if i == n - 1 else self._Phi(b[i])
            loss = self._phi(expect[i]) + expect[i] * self._Phi(expect[i])
            lb = phi_big * expect[i] + phi_b
            err = max(err, float(loss - lb))

        return Result(p=p, expect=expect, error=err)

    @staticmethod
    def self_test() -> bool:
        test_w = [2, 4, 8, 16, 32, 64]
        eps = 1e-12
        partitioner = JensenUniformPartitioner()

        for w in test_w:
            r = partitioner.compute(w + 1)
            sum_p = float(np.sum(r.p))
            norm_ok = abs(sum_p - 1.0) < eps

            mono_ok = True
            for i in range(1, len(r.expect)):
                if r.expect[i] <= r.expect[i - 1]:
                    mono_ok = False
                    break

            sym_ok = True
            for i in range(len(r.p) // 2):
                if abs(r.p[i] - r.p[len(r.p) - 1 - i]) > 1e-6:
                    sym_ok = False
                    break
                if abs(r.expect[i] + r.expect[len(r.expect) - 1 - i]) > 1e-6:
                    sym_ok = False
                    break

            if not (norm_ok and mono_ok and sym_ok):
                return False
        return True
