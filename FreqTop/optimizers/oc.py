import numpy as np
from .base import Optimizer
import time


class OCOptimizer(Optimizer):
    """Optimality Criteria (OC) update scheme.

    This is the direct extraction of the ``oc()`` function from the
    original 165-line code, wrapped in the :class:`Optimizer` interface.

    Parameters
    ----------
    move : float
        Maximum allowable change in a single element density per iteration.
    bisect_tol : float
        Convergence tolerance for the Lagrange-multiplier bisection.
    """

    def __init__(self, move: float = 0.2, bisect_tol: float = 1e-3):
        self.move = move
        self.bisect_tol = bisect_tol


    def update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        volfrac: float,
    ) -> np.ndarray:
        start = time.perf_counter()
        g = float(np.dot(dv, x - volfrac))
        x_new, _ = self._find_lagrange_multiplier(x, dc, dv, g)
        end = time.perf_counter()
        total_time = end - start
        mem_mb = (x.nbytes + x_new.nbytes + dc.nbytes + dv.nbytes) / 1024**2
        return x_new, total_time, mem_mb


    def _find_lagrange_multiplier(self, x, dc, dv, g):
        EPS   = 1e-30
        lo    = np.maximum(0.0, x - self.move)
        hi    = np.minimum(1.0, x + self.move)
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + EPS) > self.bisect_tol:
            lam   = 0.5 * (l1 + l2)
            x_new = np.clip(x * np.sqrt(np.maximum(0.0, -dc / np.maximum(dv * lam, EPS))), lo, hi)
            if g + float(np.dot(dv, x_new - x)) > 0:
                l1 = lam
            else:
                l2 = lam
        return x_new, lam
