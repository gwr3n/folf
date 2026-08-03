from abc import ABC, abstractmethod

from .result import Result


class JensenPartitioner(ABC):
    @abstractmethod
    def compute(self, segments: int) -> Result:
        """Compute a Jensen partition result for the requested number of segments."""
