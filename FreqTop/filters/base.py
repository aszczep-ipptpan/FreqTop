from abc import ABC, abstractmethod
import numpy as np


class Filter(ABC):
    """Abstract base class for density / sensitivity filters."""

    @abstractmethod
    def filter_sensitivities(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the filter to the raw sensitivities.

        Parameters
        ----------
        x : np.ndarray
            Current (unfiltered) design variables.
        dc : np.ndarray
            Raw compliance sensitivities.
        dv : np.ndarray
            Raw volume sensitivities.

        Returns
        -------
        dc_f, dv_f : filtered sensitivity arrays.
        """
        pass

    @abstractmethod
    def filter_design(self, x: np.ndarray) -> np.ndarray:
        """Map design variables *x* to physical densities *xPhys*."""
        pass
