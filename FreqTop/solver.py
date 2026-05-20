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

    def min_compliance(self) -> np.ndarray:
        """Execute the compliance-minimisation optimisation loop.

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

        change = float("inf")
        loop   = 0
        obj = float("inf")
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
                change = float(np.linalg.norm(x_new[active_elems] - x[active_elems], 2))
            else:
                change = float(np.linalg.norm(x_new - x, 2))

            x = x_new

            for cb in self.callbacks:
                cb(loop, obj, xPhys, change, elapsed)

        return xPhys

    def max_frequency(self) -> np.ndarray:
        """Execute the frequency-maximisation optimisation loop.

        Maximises the fundamental natural frequency omega_1 subject to a
        volume constraint, using SIMP material interpolation and the
        consistent Q4 mass matrix.

        Returns
        -------
        xPhys : np.ndarray, shape (nelx*nely,)
            Final physical density field.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import eigsh

        nelx, nely = self.problem.nelx, self.problem.nely
        n_el = nelx * nely
        ndof = self.problem.ndof

        # Consistent mass matrix for a Q4 unit-square element (unit density,
        # unit thickness).  Same DOF ordering as lk() / edofMat.
        ME = np.array([
            [4, 0, 2, 0, 1, 0, 2, 0],
            [0, 4, 0, 2, 0, 1, 0, 2],
            [2, 0, 4, 0, 2, 0, 1, 0],
            [0, 2, 0, 4, 0, 2, 0, 1],
            [1, 0, 2, 0, 4, 0, 2, 0],
            [0, 1, 0, 2, 0, 4, 0, 2],
            [2, 0, 1, 0, 2, 0, 4, 0],
            [0, 2, 0, 1, 0, 2, 0, 4],
        ], dtype=np.float64) / 36.0

        KE      = self.fe_solver.KE
        edofMat = self.fe_solver.edofMat
        iK      = self.fe_solver.iK
        jK      = self.fe_solver.jK
        free    = self.fe_solver.free
        penal   = self.fe_solver.penal
        Emin    = self.fe_solver.Emin
        Emax    = self.fe_solver.Emax
        rho_min = 1e-2
        rho_max = 1.0

        # ------------------------------------------------------------------
        # Passive-region bookkeeping (mirrors min_compliance)
        # ------------------------------------------------------------------
        has_passive = (
            hasattr(self.problem, "has_passive_elements")
            and self.problem.has_passive_elements()
        )
        if has_passive:
            passive_elems = self.problem.get_passive_elements()
            active_elems  = self.problem.get_active_elements()
            n_passive     = len(passive_elems)
            n_active      = len(active_elems)
            opt_volfrac   = (self.volfrac * n_el - n_passive) / n_active
        else:
            opt_volfrac = self.volfrac

        # ------------------------------------------------------------------
        # Initialise design variables
        # ------------------------------------------------------------------
        if self.x_init is not None:
            x = self.x_init.copy()
        else:
            x = np.full(n_el, self.volfrac)
        if has_passive:
            x = self.problem.correct_design_variable(x, self.volfrac)

        xPhys = self.filter.filter_design(x)
        if has_passive:
            xPhys[passive_elems] = 1.0

        change = float("inf")
        loop   = 0
        obj    = 0.0

        while change > self.tol and loop < self.max_iter:
            loop += 1

            # 1. from Flow Chart: FE solve + sensitivity analysis
            # SIMP interpolation
            Ee = Emin + xPhys**penal * (Emax - Emin)
            re = rho_min + xPhys * (rho_max - rho_min)

            # Assemble global K and M via COO (same pattern as FESolver.solve)
            sK = ((KE.flatten()[np.newaxis]).T * Ee).flatten(order="F")
            sM = ((ME.flatten()[np.newaxis]).T * re).flatten(order="F")
            K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsr()
            M = coo_matrix((sM, (iK, jK)), shape=(ndof, ndof)).tocsr()

            Kf = K[np.ix_(free, free)]
            Mf = M[np.ix_(free, free)]

            # Solve generalised eigenvalue problem for the first mode
            try:
                eigvals, eigvecs = eigsh(
                    Kf, k=1, M=Mf, sigma=0.0, which="LM", tol=1e-8, maxiter=4000
                )
            except Exception:
                continue

            lam1   = float(np.real(eigvals[0]))
            omega1 = float(np.sqrt(max(lam1, 0.0)))
            obj    = omega1

            # Computation of generalized gradients f_sk (from article OlhoffDu(2007))
            # Sensitivity of lambda_1 w.r.t. physical density (adjoint)
            v = np.real(eigvecs[:, 0])
            norm_m = float(v @ (Mf @ v))
            if norm_m > 1e-30:
                v /= np.sqrt(norm_m)

            phi        = np.zeros(ndof)
            phi[free]  = v
            pe         = phi[edofMat]          # (n_el, 8) — mode shape at element DOFs

            dKdx   = penal * (Emax - Emin) * xPhys**(penal - 1) * np.sum((pe @ KE) * pe, axis=1)
            dMdx   = (rho_max - rho_min) * np.sum((pe @ ME) * pe, axis=1)
            dlam_dx = dKdx - lam1 * dMdx

            if omega1 > 1e-10:
                domega_dx = dlam_dx / (2.0 * omega1)
            else:
                domega_dx = dlam_dx

            # Negate: optimizer minimises, we maximise omega
            dc = -domega_dx
            dv = np.ones(n_el)

            dc, dv = self.filter.filter_sensitivities(x, dc, dv)

            if has_passive:
                dc[passive_elems] = 0.0
                dv[passive_elems] = 0.0
            # 4. Update values of the design variables rho_e = rho_e + delta rho
            x_new, elapsed, *_ = self.optimizer.update(x, dc, dv, opt_volfrac)

            if has_passive:
                x_new = self.problem.correct_design_variable(x_new, self.volfrac)

            xPhys = self.filter.filter_design(x_new)
            if has_passive:
                xPhys[passive_elems] = 1.0
            #Stop criterion - if rho is converged ?
            if has_passive:
                change = float(np.linalg.norm(x_new[active_elems] - x[active_elems], 2))
            else:
                change = float(np.linalg.norm(x_new - x, 2))

            x = x_new

            for cb in self.callbacks:
                cb(loop, obj, xPhys, change, elapsed)

        return xPhys
