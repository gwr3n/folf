from __future__ import annotations

import numpy as np

from .jensen_minimax_partitioner import JensenMinimaxPartitioner
from .jensen_uniform_partitioner import JensenUniformPartitioner
from .partitioner import PARTITIONER


class PiecewiseStandardNormalFirstOrderLossFunction:
    """Cached accessors for standard-normal piecewise linearization data."""

    partitioner = PARTITIONER.MINIMAX
    _partitioner = PARTITIONER.MINIMAX
    _linearization_samples = 0
    _error_cache: dict[int, float] = {}
    _probabilities_cache: dict[int, np.ndarray] = {}
    _means_cache: dict[int, np.ndarray] = {}

    @classmethod
    def get_partitioner(cls) -> PARTITIONER:
        return cls._partitioner

    @classmethod
    def get_linearization_samples(cls) -> int:
        return cls._linearization_samples

    @classmethod
    def get_error(cls, partitions: int) -> float:
        if partitions <= 0:
            msg = "partitions must be >= 1"
            raise ValueError(msg)
        if partitions not in cls._error_cache:
            cls._error_cache[partitions] = float(cls._compute(partitions).error)
        return cls._error_cache[partitions]

    @classmethod
    def get_probabilities(cls, partitions: int) -> np.ndarray:
        if partitions <= 0:
            msg = "partitions must be >= 1"
            raise ValueError(msg)
        if partitions not in cls._probabilities_cache:
            src = cls._compute(partitions).p
            cls._probabilities_cache[partitions] = np.array(src, dtype=float, copy=True)
        return np.array(cls._probabilities_cache[partitions], dtype=float, copy=True)

    @classmethod
    def get_means(cls, partitions: int) -> np.ndarray:
        if partitions <= 0:
            msg = "partitions must be >= 1"
            raise ValueError(msg)
        if partitions not in cls._means_cache:
            src = cls._compute(partitions).expect
            cls._means_cache[partitions] = np.array(src, dtype=float, copy=True)
        return np.array(cls._means_cache[partitions], dtype=float, copy=True)

    @classmethod
    def _compute(cls, partitions: int):
        if cls._partitioner == PARTITIONER.UNIFORM:
            return JensenUniformPartitioner().compute(partitions + 1)
        return JensenMinimaxPartitioner().compute(partitions + 1)
