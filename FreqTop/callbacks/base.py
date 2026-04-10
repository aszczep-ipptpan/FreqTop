from abc import ABC, abstractmethod
import numpy as np


class Callback(ABC):
    """Abstract base for iteration callbacks (plotting, logging, etc.)."""

    @abstractmethod
    def __call__(
        self,
        loop: int,
        obj: float,
        xPhys: np.ndarray,
        change: float,
    ) -> None:
        """Called once per optimisation iteration.

        Parameters
        ----------
        loop : int
            Current iteration number (1-based).
        obj : float
            Current compliance objective value.
        xPhys : np.ndarray
            Current physical density field.
        change : float
            Inf-norm change between this and the previous design.
        """
        pass
