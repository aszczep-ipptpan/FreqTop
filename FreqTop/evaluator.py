"""
FreqTop/evaluator.py
=====================
TopOptEvaluator — wraps problem + FESolver + Filter into a single evaluator
that is interface-compatible with BenchmarkProblem.

The key insight is that both topology optimisation and benchmark problems
reduce to the same abstract interface::

    evaluate(x) -> (f, dc, g, dv)

where
    f   — scalar objective
    dc  — objective gradient  ∂f/∂x
    g   — constraint residual  (< 0 → feasible)
    dv  — constraint gradient  ∂g/∂x

TopOptEvaluator implements this by composing:
    1.  filter.filter_design(x)              → xPhys
    2.  fe.evaluate(xPhys)                   → (f, dc, dv)   [new FESolver method]
    3.  filter.filter_sensitivities(x,dc,dv) → (dc_f, dv_f)
    4.  g = mean(xPhys) − volfrac

This lets TopOptSolver use the same loop for both FE-based topology
optimisation and analytic benchmark problems.
"""

from __future__ import annotations

import numpy as np

from .fe.fe_solver  import FESolver
from .filters.base  import Filter
from .utils.base    import TopOptSolver as FEProblem


class TopOptEvaluator:
    """BenchmarkProblem-compatible evaluator for finite-element topology optimisation.

    Bundles together the three objects that ``run.py`` previously handled
    separately (``problem``, ``fe``, ``filt``) into a single evaluator with
    the same ``evaluate(x) -> (f, dc, g, dv)`` contract as
    ``BenchmarkProblem``.

    Passive-region handling (BeamDomain flanges) is encapsulated here so the
    solver loop stays clean.

    Parameters
    ----------
    problem : FEProblem (TopOptSolver ABC)
        Domain with BCs and loads.  May optionally expose passive-element
        methods (``has_passive_elements``, ``get_passive_elements``,
        ``get_active_elements``, ``correct_design_variable``).
    fe_solver : FESolver
        Assembled FE system.
    filter : Filter
        Density or sensitivity filter.
    volfrac : float
        Target volume fraction (overall, including any passive material).
    """

    def __init__(
        self,
        problem:   FEProblem,
        fe_solver: FESolver,
        filter:    Filter,
        volfrac:   float,
    ) -> None:
        self._problem  = problem
        self._fe       = fe_solver
        self._filt     = filter
        self._volfrac  = float(volfrac)

        # ── Passive-region bookkeeping ────────────────────────────────────
        self._has_passive = (
            hasattr(problem, "has_passive_elements")
            and problem.has_passive_elements()
        )
        if self._has_passive:
            self._passive_elems = problem.get_passive_elements()
            self._active_elems  = problem.get_active_elements()
            n_passive = len(self._passive_elems)
            n_active  = len(self._active_elems)
            n_total   = problem.nelx * problem.nely
            # Effective volfrac the optimizer targets — active region only
            self._opt_volfrac = (volfrac * n_total - n_passive) / n_active
        else:
            self._passive_elems = np.empty(0, dtype=int)
            self._active_elems  = None  # None = all elements
            self._opt_volfrac   = volfrac

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def volfrac(self) -> float:
        """Effective volfrac passed to the optimizer (excludes passive elements)."""
        return self._opt_volfrac

    @property
    def has_passive(self) -> bool:
        return self._has_passive

    @property
    def active_elems(self):
        """Active element indices or None (meaning all elements)."""
        return self._active_elems

    @property
    def label(self) -> str:
        return getattr(self._problem, "label", "")

    # ── Core interface (matches BenchmarkProblem) ───────────────────────────

    def x0(self) -> np.ndarray:
        """Initial uniform design variable vector, corrected for passive elements."""
        n = self._problem.nelx * self._problem.nely
        x = np.full(n, self._volfrac)
        if self._has_passive:
            x = self._problem.correct_design_variable(x, self._volfrac)
        return x

    def evaluate(
        self,
        x: np.ndarray,
        xPhys: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, float, np.ndarray]:
        """Evaluate compliance, filtered sensitivities, and constraint residual.

        Parameters
        ----------
        x : np.ndarray
            Design variables (pre-filter), shape ``(nelx*nely,)``.
        xPhys : np.ndarray, optional
            Physical (filtered) densities.  Computed from ``x`` when ``None``.
            Pass explicitly from the solver loop to avoid a redundant
            filter call (filter is called once at end of each iteration
            to produce xPhys for the *next* evaluate).

        Returns
        -------
        f  : float          — compliance objective
        dc : np.ndarray     — objective gradient (filtered)
        g  : float          — constraint residual = mean(xPhys) − volfrac_orig
        dv : np.ndarray     — constraint gradient (filtered)
        """
        if xPhys is None:
            xPhys = self.filter_design(x)

        # FE solve + compliance sensitivities (uses FESolver.evaluate())
        f, dc, dv = self._fe.evaluate(xPhys)

        # Filter sensitivities
        dc, dv = self._filt.filter_sensitivities(x, dc, dv)

        # Zero sensitivities on passive elements after filtering
        if self._has_passive:
            dc[self._passive_elems] = 0.0
            dv[self._passive_elems] = 0.0

        # Constraint residual (volume − target) for history tracking
        g = float(xPhys.mean()) - self._volfrac

        return f, dc, g, dv

    # ── Helpers used by the solver loop ────────────────────────────────────

    def filter_design(self, x: np.ndarray) -> np.ndarray:
        """Apply density filter and enforce passive-element densities."""
        xPhys = self._filt.filter_design(x)
        if self._has_passive:
            xPhys[self._passive_elems] = 1.0
        return xPhys

    def correct_design(self, x: np.ndarray) -> np.ndarray:
        """Apply passive-region rescaling to design variables after optimizer update."""
        if self._has_passive:
            return self._problem.correct_design_variable(x, self._volfrac)
        return x
