from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """Strategy interface for design-variable update schemes."""

    @abstractmethod
    def update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        volfrac: float,
    ) -> np.ndarray:
        """Return updated design variables.

        Parameters
        ----------
        x : np.ndarray
            Current design variables (shape ``(n,)``).
        dc : np.ndarray
            Filtered compliance sensitivities (shape ``(n,)``).
        dv : np.ndarray
            Filtered volume sensitivities (shape ``(n,)``).
        volfrac : float
            Target volume fraction.

        Returns
        -------
        x_new : np.ndarray
            Updated design variables.
        """
        ...
