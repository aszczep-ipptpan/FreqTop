import numpy as np
from base import Filter
from _matrix import build_filter_matrix


class DensityFilter(Filter):
    """Density-based filter (``ft=1`` in the original code).

    Both sensitivities and the design-to-physical mapping pass through H.
    """

    def __init__(self, nelx: int, nely: int, rmin: float):
        self.H, self.Hs = build_filter_matrix(nelx, nely, rmin)

    def filter_sensitivities(
        self, x: np.ndarray, dc: np.ndarray, dv: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dc_f = np.asarray(self.H * (dc[np.newaxis].T / self.Hs))[:, 0]
        dv_f = np.asarray(self.H * (dv[np.newaxis].T / self.Hs))[:, 0]
        return dc_f, dv_f

    def filter_design(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.H * x[np.newaxis].T / self.Hs)[:, 0]
