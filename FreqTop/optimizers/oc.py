import numpy as np
from .base import Optimizer


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
        self._g = 0.0  # Nguyen/Paulino internal state

    def update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        volfrac: float,
    ) -> np.ndarray:
        n = len(x)
        l1, l2 = 0.0, 1e9
        move = self.move
        xnew = np.zeros(n)

        while (l2 - l1) / (l1 + l2) > self.bisect_tol:
            lmid = 0.5 * (l2 + l1)
            xnew[:] = np.maximum(
                0.0,
                np.maximum(
                    x - move,
                    np.minimum(
                        1.0,
                        np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)),
                    ),
                ),
            )
            gt = self._g + np.sum(dv * (xnew - x))
            if gt > 0:
                l1 = lmid
            else:
                l2 = lmid

        self._g = gt  # carry state forward (Nguyen/Paulino approach)
        return xnew
