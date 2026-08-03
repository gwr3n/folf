from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Result:
    """Container for partition probabilities, conditional expectations, and max error."""

    p: np.ndarray
    expect: np.ndarray
    error: float
