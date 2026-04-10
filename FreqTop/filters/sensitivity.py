import numpy as np
from base import Filter
from _matrix import build_filter_matrix


class SensitivityFilter(Filter):
    """Sensitivity-based filter (``ft=0`` in the original code).

    The design variables are passed through unchanged as physical densities;
    only the sensitivities are smoothed.
    """

    def __init__(self, nelx: int, nely: int, rmin: float):
        self.H, self.Hs = build_filter_matrix(nelx, nely, rmin)

    def filter_sensitivities(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dc_f = np.asarray(
            (self.H * (x * dc))[np.newaxis].T / self.Hs
        )[:, 0] / np.maximum(0.001, x)
        # dv is not modified by the sensitivity filter
        return dc_f, dv

    def filter_design(self, x: np.ndarray) -> np.ndarray:
        # No projection — physical = design
        return x.copy()
