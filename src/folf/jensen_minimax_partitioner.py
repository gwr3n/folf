import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .jensen_partitioner import JensenPartitioner
from .result import Result


class JensenMinimaxPartitioner(JensenPartitioner):
    """Minimax Jensen partitioner for standard normal demand."""

    @staticmethod
    def _logsumexp(loga: float, logb: float) -> float:
        if math.isinf(loga):
            return logb
        if math.isinf(logb):
            return loga
        m = max(loga, logb)
        return m + math.log(math.exp(loga - m) + math.exp(logb - m))

    @classmethod
    def _to_breakpoints(cls, y: np.ndarray) -> np.ndarray:
        out = np.empty_like(y)
        cum = -np.inf
        for i, yi in enumerate(y):
            cum = cls._logsumexp(cum, float(yi))
            out[i] = cum
        return out

    @classmethod
    def _objective(cls, y: np.ndarray, n: int, m_pos: int, add_zero: bool) -> float:
        bp_pos = cls._to_breakpoints(y)
        b_count = 2 * m_pos + (1 if add_zero else 0)
        b_int = np.empty(b_count)

        for j in range(m_pos):
            b_int[j] = -bp_pos[m_pos - 1 - j]
        if add_zero:
            b_int[m_pos] = 0.0
        for j in range(m_pos):
            b_int[m_pos + (1 if add_zero else 0) + j] = bp_pos[j]

        a = np.empty(n)
        b = np.empty(n)
        a[0] = -np.inf
        if n > 1:
            a[1:] = b_int
            b[:-1] = b_int
        b[-1] = np.inf

        e = np.empty(n)
        for i in range(n):
            phi_ai = 0.0 if i == 0 else norm.pdf(a[i])
            phi_bi = 0.0 if i == n - 1 else norm.pdf(b[i])
            Phi_ai = 0.0 if i == 0 else norm.cdf(a[i])
            Phi_bi = 1.0 if i == n - 1 else norm.cdf(b[i])
            p_i = Phi_bi - Phi_ai
            E_i = (phi_ai - phi_bi) / p_i
            loss = norm.pdf(E_i) + E_i * norm.cdf(E_i)
            lb = Phi_bi * E_i + phi_bi
            e[i] = loss - lb

        d = e[1:] - e[0]
        return float(np.dot(d, d))

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
        if n == 2:
            p = np.array([0.5, 0.5])
            phi0 = norm.pdf(0.0)
            e_i = -phi0 / 0.5
            expect = np.array([e_i, -e_i])
            Phi_e = norm.cdf(-e_i)
            loss = norm.pdf(e_i) + (-e_i) * Phi_e
            lb = 0.5 * (-e_i) + phi0
            return Result(p=p, expect=expect, error=float(loss - lb))

        m_pos = (n - 1) // 2
        add_zero = (n % 2) == 0

        y0 = np.log(np.arange(m_pos, dtype=float) + 0.5)
        initial_simplex = np.vstack((y0, y0 + np.eye(m_pos)))
        sol = minimize(
            fun=lambda y: self._objective(y, n, m_pos, add_zero),
            x0=y0,
            method="Nelder-Mead",
            options={
                "initial_simplex": initial_simplex,
                "maxiter": 1_000_000,
                "maxfev": 1_000_000,
                "xatol": 1e-12,
                "fatol": 1e-12,
            },
        )
        y = sol.x
        bp_pos = self._to_breakpoints(y)

        b_count = 2 * m_pos + (1 if add_zero else 0)
        b_int = np.empty(b_count)
        for j in range(m_pos):
            b_int[j] = -bp_pos[m_pos - 1 - j]
        if add_zero:
            b_int[m_pos] = 0.0
        for j in range(m_pos):
            b_int[m_pos + (1 if add_zero else 0) + j] = bp_pos[j]

        a = np.empty(n)
        b = np.empty(n)
        a[0] = -np.inf
        if n > 1:
            a[1:] = b_int
            b[:-1] = b_int
        b[-1] = np.inf

        p = np.empty(n)
        expect = np.empty(n)
        err = 0.0
        for i in range(n):
            phi_ai = 0.0 if i == 0 else norm.pdf(a[i])
            phi_bi = 0.0 if i == n - 1 else norm.pdf(b[i])
            Phi_ai = 0.0 if i == 0 else norm.cdf(a[i])
            Phi_bi = 1.0 if i == n - 1 else norm.cdf(b[i])

            p[i] = Phi_bi - Phi_ai
            expect[i] = (phi_ai - phi_bi) / p[i]

            loss = norm.pdf(expect[i]) + expect[i] * norm.cdf(expect[i])
            lb = Phi_bi * expect[i] + phi_bi
            err = max(err, float(loss - lb))

        return Result(p=p, expect=expect, error=err)

    @staticmethod
    def get_errors() -> np.ndarray:
        return np.array(
            [
                0.3989422804014327,
                0.1206560496714961,
                0.05784405029198253,
                0.033905164962384104,
                0.022270929512393414,
                0.01574607463566398,
                0.011721769576577057,
                0.00906528789647753,
                0.007219916411227892,
                0.005885974956458359,
            ],
            dtype=float,
        )
