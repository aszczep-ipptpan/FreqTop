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
        self._g = 0.0  # Nguyen/Paulino internal state




    def update(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        volfrac: float,
    ) -> np.ndarray:
        """Compute the OC update for the design variables.

        Parameters
        ----------
        x : np.ndarray, shape (nelx*nely,)
            Current design variable vector.
        dc : np.ndarray, shape (nelx*nely,)
            Sensitivity of the objective function w.r.t. design variables.
        dv : np.ndarray, shape (nelx*nely,)
            Sensitivity of the volume constraint w.r.t. design variables.
        volfrac : float
            Target volume fraction (used to compute the constraint residual).   
        """
        start = time.perf_counter()
        self._g = np.sum(dv * (x - volfrac))
        lmbda = self._find_lagrange_multiplier(x, dc, dv)
        x_new = self._update_density(x, dc, dv, lmbda)
        end = time.perf_counter()
        total_time = end - start
        return x_new, total_time


    # =========================
    # AKTUALIZACJA ZMIENNYCH
    # =========================
    def _update_density(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
        lmbda: float,
    ) -> np.ndarray:
        """Wzór OC:
        x_new = x * sqrt(-dc / (dv * λ))
        z ograniczeniami move i [0,1]

        Guard: when dv*lmbda ≈ 0 (passive elements zeroed by solver),
        ratio defaults to 1.0 so x_candidate = x (element unchanged).
        """
        denom = dv * lmbda
        safe  = np.abs(denom) > 1e-30
        ratio = np.where(safe, -dc / np.where(safe, denom, 1.0), 1.0)
        x_candidate = x * np.sqrt(np.maximum(0.0, ratio))
        x_new = np.maximum(
            0.0,
            np.maximum(
                x - self.move,
                np.minimum(
                    1.0,
                    np.minimum(x + self.move, x_candidate),
                ),
            ),
        )
        return x_new

    # =========================
    # BISEKCJA DLA λ
    # =========================
    def _find_lagrange_multiplier(
        self,
        x: np.ndarray,
        dc: np.ndarray,
        dv: np.ndarray,
    ) -> float:
        """Rozwiązuje równanie constraintu:
        
        g(λ) = Σ dv * (x_new(λ) - x) + g_old = 0
        """
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2 + 1e-30) > self.bisect_tol:
            lmid = 0.5 * (l1 + l2)
            x_new = self._update_density(x, dc, dv, lmid)
            g_val = self._constraint_residual(x, x_new, dv)
            if g_val > 0:
                l1 = lmid
            else:
                l2 = lmid
        return 0.5 * (l1 + l2)

    # =========================
    # RESIDUUM OGRANICZENIA
    # =========================
    def _constraint_residual(
        self,
        x_old: np.ndarray,
        x_new: np.ndarray,
        dv: np.ndarray,
    ) -> float:
        """g(λ) – ile naruszamy constraint objętości"""
        return self._g + np.sum(dv * (x_new - x_old))