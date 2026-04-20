"""FreqTop/solver.py — topology optimisation loop."""

import numpy as np

from .utils.base import TopOptSolver as Problem
from .fe.fe_solver import FESolver
from .filters.base import Filter
from .optimizers.base import Optimizer


class TopOptSolver:
    """Orchestrates the topology optimisation loop.

    Parameters
    ----------
    problem : Problem
        Defines domain, BCs, loads, and optionally passive regions.
        Must expose ``nelx``, ``nely``, ``get_fixed_dofs()``,
        ``get_load_vector()``.  If it also exposes
        ``has_passive_elements()`` / ``get_passive_elements()`` /
        ``get_active_elements()`` / ``correct_design_variable()``,
        passive-flange logic is activated automatically (e.g. BeamDomain).
    fe_solver : FESolver
        Assembles and solves the FE system; computes sensitivities.
    filter : Filter
        Filters sensitivities and maps design vars to physical densities.
    optimizer : Optimizer
        Updates design variables given sensitivities.
    volfrac : float
        Target volume fraction (overall, including passive material).
    callbacks : sequence of callable
        Zero or more callbacks called each iteration with signature
        ``(loop, obj, xPhys, change, elapsed)``.
    max_iter : int
        Hard upper bound on number of iterations.
    tol : float
        Convergence tolerance on the inf-norm design change
        (measured on active elements only when passive regions exist).
    """

    def __init__(
        self,
        problem: Problem,
        fe_solver: FESolver,
        filter: Filter,
        optimizer: Optimizer,
        volfrac: float = 0.4,
        callbacks: tuple = (),
        max_iter: int = 2000,
        tol: float = 0.01,
        x_init: np.ndarray | None = None,
    ):
        self.problem   = problem
        self.fe_solver = fe_solver
        self.filter    = filter
        self.optimizer = optimizer
        self.volfrac   = volfrac
        self.callbacks = callbacks
        self.max_iter  = max_iter
        self.tol       = tol
        self.x_init    = x_init

    def run(self) -> np.ndarray:
        """Execute the optimisation loop.

        Returns
        -------
        xPhys : np.ndarray, shape (nelx*nely,)
            Final physical density field.
        """
        nelx, nely = self.problem.nelx, self.problem.nely

        # ------------------------------------------------------------------
        # Passive-region bookkeeping
        # ------------------------------------------------------------------
        has_passive = (
            hasattr(self.problem, "has_passive_elements")
            and self.problem.has_passive_elements()
        )
        if has_passive:
            passive_elems = self.problem.get_passive_elements()
            active_elems  = self.problem.get_active_elements()
            # Effective volume fraction that the optimizer should target —
            # excludes passive elements which are locked at density 1.0.
            # volfrac_active = (volfrac * nelxy - n_passive) / n_active
            n_passive    = len(passive_elems)
            n_active     = len(active_elems)
            opt_volfrac  = (self.volfrac * (nelx * nely) - n_passive) / n_active
        else:
            opt_volfrac = self.volfrac

        # ------------------------------------------------------------------
        # Initialise design variables
        # ------------------------------------------------------------------
        if self.x_init is not None:
            x = self.x_init.copy()
        else:
            x = np.full(nelx * nely, self.volfrac)
        if has_passive:
            x = self.problem.correct_design_variable(x, self.volfrac)

        xPhys = self.filter.filter_design(x)
        if has_passive:
            xPhys[passive_elems] = 1.0

        change = 1.0
        loop   = 0

        while change > self.tol and loop < self.max_iter:
            loop += 1

            # FE solve + sensitivity analysis
            u              = self.fe_solver.solve(xPhys)
            obj, dc, dv    = self.fe_solver.sensitivities(xPhys, u)

            dc, dv = self.filter.filter_sensitivities(x, dc, dv)

            # Zero passive-element sensitivities AFTER filtering.
            # Interior passive elements (all-passive neighbourhood) would get
            # dc=dv=0 from the filter anyway, but boundary passive elements
            # receive leaked contributions from active neighbours — zeroing
            # here ensures the optimiser never tries to move passive elements
            # and avoids 0/0 in the OC update formula.
            if has_passive:
                dc[passive_elems] = 0.0
                dv[passive_elems] = 0.0

            x_new, elapsed = self.optimizer.update(x, dc, dv, opt_volfrac)

            # Enforce passive densities and rescale active region
            if has_passive:
                x_new = self.problem.correct_design_variable(x_new, self.volfrac)

            xPhys = self.filter.filter_design(x_new)
            if has_passive:
                xPhys[passive_elems] = 1.0

            # Convergence measured on active elements only
            if has_passive:
                change = float(np.linalg.norm(x_new[active_elems] - x[active_elems], np.inf))
            else:
                change = float(np.linalg.norm(x_new - x, np.inf))

            x = x_new

            for cb in self.callbacks:
                cb(loop, obj, xPhys, change, elapsed)

        return xPhys
