"""FreqTop/solver.py — topology optimisation loop."""

from collections import namedtuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

from .utils.base import TopOptSolver as Problem
from .fe.fe_solver import FESolver
from .filters.base import Filter
from .optimizers.base import Optimizer


# ---------------------------------------------------------------------------
# Material / element / FEM helpers
# ---------------------------------------------------------------------------

class MaterialProperties:
    """SIMP material interpolation parameters and physical material constants."""

    def __init__(
        self,
        Emin: float = 1e-9,
        Emax: float = 1.0,
        penal: float = 3.0,
        rho_min: float = 1e-2,
        rho_max: float = 1.0,
        E: float = 1.0,
        rho: float = 1.0,
    ):
        self.Emin    = Emin
        self.Emax    = Emax
        self.penal   = penal
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.E       = E    # physical Young's modulus [Pa]
        self.rho     = rho  # physical mass density [kg/m³]

    def read_json_params(self, json_path: str) -> None:
        """Read all material and SIMP parameters from a JSON parameter file.

        Reads physical constants (E, nu, rho) from ``materials.base`` and
        SIMP interpolation parameters (Emin, Emax, penal, rho_min, rho_max)
        from ``optimisation``.  Unit conversion is applied to the elastic
        modulus (GPa → Pa, MPa → Pa).
        """
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            root_params = json.load(f)

        mat    = root_params.get("materials").get("base")
        opt    = root_params.get("optimisation", {})
        units  = root_params.get("meta", {}).get("units", {}).get("elastic_modulus", "")

        self.E  = mat.get("E")
        if units == "GPa":
            self.E *= 1e9
        elif units == "MPa":
            self.E *= 1e6
        self.nu  = mat.get("nu")
        self.rho = mat.get("rho")

        self.penal   = opt.get("penalization",  self.penal)
        self.Emin    = opt.get("E_min",         self.Emin)
        self.Emax    = self.E
        self.rho_min = opt.get("rho_min",       self.rho_min)
        self.rho_max = opt.get("rho_max",       self.rho_max)


class Element:
    """QUAD4 element stiffness and consistent mass matrices."""

    def __init__(self, KE: np.ndarray):
        self._KE = KE

    def compute_QUAD4_element_stiffness(self, E: float) -> np.ndarray:
        """Return the physical element stiffness matrix scaled by Young's modulus E [Pa]."""
        return self._KE * E

    @staticmethod
    def compute_QUAD4_element_mass(rho: float, volume_element: float = 1.0) -> np.ndarray:
        """Return the physical consistent mass matrix for a Q4 element.

        Parameters
        ----------
        rho : float
            Mass density [kg/m³].
        volume_element : float
            Element volume (length_x * length_y / n_elements) [m³].
        """
        ME_unit = np.array([
            [4, 0, 2, 0, 1, 0, 2, 0],
            [0, 4, 0, 2, 0, 1, 0, 2],
            [2, 0, 4, 0, 2, 0, 1, 0],
            [0, 2, 0, 4, 0, 2, 0, 1],
            [1, 0, 2, 0, 4, 0, 2, 0],
            [0, 1, 0, 2, 0, 4, 0, 2],
            [2, 0, 1, 0, 2, 0, 4, 0],
            [0, 2, 0, 1, 0, 2, 0, 4],
        ], dtype=np.float64) / 36.0
        return ME_unit * rho * volume_element


_FreqResult = namedtuple("FreqResult", ["lam1", "omega1", "v", "eigvals", "eigvecs"])


class FEM:
    """FE assembly and fundamental-frequency computation for topology optimisation."""

    def __init__(
        self,
        fe_solver: FESolver,
        material: MaterialProperties,
        ndof: int,
        volume_element: float = 1.0,
    ):
        self.material = material
        element       = Element(fe_solver.KE)
        # Physical element matrices: E and rho * volume_element already baked in.
        self.KE       = element.compute_QUAD4_element_stiffness(material.E)
        self.ME       = Element.compute_QUAD4_element_mass(material.rho, volume_element)
        self.edofMat  = fe_solver.edofMat
        self.iK       = fe_solver.iK
        self.jK       = fe_solver.jK
        self.free     = fe_solver.free
        self.ndof     = ndof

    def assemble_stiffness_matrix(self, coeff: np.ndarray):
        """Assemble global K from per-element SIMP stiffness coefficients (dimensionless)."""
        sK = ((self.KE.flatten()[np.newaxis]).T * coeff).flatten(order="F")
        return coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsr()

    def assemble_mass_matrix(self, coeff: np.ndarray):
        """Assemble global M from per-element SIMP mass coefficients (dimensionless)."""
        sM = ((self.ME.flatten()[np.newaxis]).T * coeff).flatten(order="F")
        return coo_matrix((sM, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsr()

    def compute_freq(self, xPhys: np.ndarray, k: int = 1) -> "_FreqResult | None":
        """Assemble K and M, solve generalised EVP for k lowest modes.

        Returns k eigenvalues (ascending) and M-normalised eigenvectors, plus
        lam1/omega1/v for the fundamental mode.  Returns None on solver failure.
        """
        mat  = self.material
        Ee   = mat.Emin + xPhys ** mat.penal * (mat.Emax - mat.Emin)

        # Tcherniak (2002) mass modification — eliminates spurious localized
        # eigenmodes in near-void regions (Du & Olhoff 2007, Section 2.2, eq. 4a).
        # For ρₑ > 0.1: linear interpolation  re = rho_min + ρ*(rho_max-rho_min)
        # For ρₑ ≤ 0.1: penalised  re = c0 * ρ^6 * rho_max
        # Continuity at ρ=0.1 requires c0 = (rho_min + 0.1*(rho_max-rho_min)) / (rho_max * 0.1^6)
        _rho_thresh  = 0.1
        _r_void      = 6
        _re_thresh   = mat.rho_min + _rho_thresh * (mat.rho_max - mat.rho_min)
        _c0          = _re_thresh / (mat.rho_max * _rho_thresh ** _r_void)
        re = np.where(
            xPhys > _rho_thresh,
            mat.rho_min + xPhys * (mat.rho_max - mat.rho_min),
            _c0 * xPhys ** _r_void * mat.rho_max,
        )

        # KE and ME already carry physical units (E [Pa], rho [kg/m³], volume_element [m³])
        K    = self.assemble_stiffness_matrix(Ee)
        M    = self.assemble_mass_matrix(re)
        free = self.free
        Kf   = K[free, :][:, free].tocsr()
        Mf   = M[free, :][:, free].tocsr()
        try:
            eigvals, eigvecs = eigsh(
                Kf, k=k, M=Mf, sigma=0.0, which="LM", tol=1e-8, maxiter=4000
            )
        except Exception:
            return None
        # Sort ascending and take real parts
        idx     = np.argsort(np.real(eigvals))
        eigvals = np.real(eigvals[idx])
        eigvecs = np.real(eigvecs[:, idx])
        # M-normalise every eigenvector: vᵢᵀ M vᵢ = 1
        for i in range(k):
            norm_m = float(eigvecs[:, i] @ (Mf @ eigvecs[:, i]))
            if norm_m > 1e-30:
                eigvecs[:, i] /= np.sqrt(norm_m)
        lam1   = float(eigvals[0])
        omega1 = float(np.sqrt(max(lam1, 0.0)))
        v      = eigvecs[:, 0]
        return _FreqResult(lam1=lam1, omega1=omega1, v=v, eigvals=eigvals, eigvecs=eigvecs)


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
        nelx, nely = self.problem.nelx, self.problem.nely
        n_el = nelx * nely
        ndof = self.problem.ndof

        material = MaterialProperties(
            Emin  = self.fe_solver.Emin,
            Emax  = self.fe_solver.Emax,
            penal = self.fe_solver.penal,
            E     = getattr(self.problem, "E",   205e9),
            rho   = getattr(self.problem, "rho", 7850.0),
        )
        volume_element = getattr(self.problem, "volume_element", 1.0)
        fem = FEM(self.fe_solver, material, ndof, volume_element)

        KE      = fem.KE   # KE_unit * E  — physical stiffness
        ME      = fem.ME   # ME_unit * rho * volume_element — same matrix used in compute_freq
        edofMat = fem.edofMat
        free    = fem.free

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

        while loop < self.max_iter and change > self.tol:
            loop += 1

            xPhys_thresholded = xPhys.copy()

            # Assemble K, M and solve generalised EVP (bound formulation: k modes)
            N_MODES   = 3
            TOL_CLUSTER = 0.01   # modes within 1 % of beta are considered active

            freq_result = fem.compute_freq(xPhys_thresholded, k=N_MODES)
            if freq_result is None:
                continue

            beta   = freq_result.lam1          # lower bound on eigenvalues (= min λᵢ)
            omega1 = freq_result.omega1         # √β — for reporting only
            obj    = omega1

            # Identify active (clustered) modes: λᵢ ≤ (1 + tol) · β
            active_mask = freq_result.eigvals <= (1.0 + TOL_CLUSTER) * beta
            m_active    = int(active_mask.sum())
            mu          = 1.0 / m_active        # equal weights (simple multiplicity)

            # Bound-formulation sensitivity (OlhoffDu 2007):
            #   ∂β/∂xₑ = Σᵢ μᵢ · (vᵢᵀ ∂K/∂xₑ vᵢ  −  λᵢ · vᵢᵀ ∂M/∂xₑ vᵢ)
            dbeta_dx = np.zeros(n_el)
            for i_mode in np.where(active_mask)[0]:
                lam_i       = freq_result.eigvals[i_mode]
                phi_i       = np.zeros(ndof)
                phi_i[free] = freq_result.eigvecs[:, i_mode]
                pe_i        = phi_i[edofMat]   # (n_el, 8)
                dK_i = (
                    material.penal * (material.Emax - material.Emin)
                    * xPhys ** (material.penal - 1)
                    * np.sum((pe_i @ KE) * pe_i, axis=1)
                )
                dM_i = (
                    (material.rho_max - material.rho_min)
                    * np.sum((pe_i @ ME) * pe_i, axis=1)
                )
                dbeta_dx += mu * (dK_i - lam_i * dM_i)

            # Negate: MMA minimises, we maximise beta
            dc = -dbeta_dx
            dv = np.ones(n_el)

            dc, dv = self.filter.filter_sensitivities(x, dc, dv)

            if has_passive:
                dc[passive_elems] = 0.0
                dv[passive_elems] = 0.0

            x_new, elapsed, *_ = self.optimizer.update(x, dc, dv, opt_volfrac)

            if has_passive:
                x_new = self.problem.correct_design_variable(x_new, self.volfrac)

            xPhys = self.filter.filter_design(x_new)
            if has_passive:
                xPhys[passive_elems] = 1.0

            if has_passive:
                change = float(np.linalg.norm(x_new[active_elems] - x[active_elems], 2))
            else:
                change = float(np.linalg.norm(x_new - x, 2))

            x = x_new

            for cb in self.callbacks:
                cb(loop, obj, xPhys, change, elapsed)

        return xPhys
