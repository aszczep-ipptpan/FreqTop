"""FreqTop/fe/fe_solver.py — FE assembly and static solve."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from ..utils.base import TopOptSolver
from .elements import lk


class FESolver:
    """Assembles and solves the FE system for a given design field *xPhys*.

    Parameters
    ----------
    problem : TopOptSolver or BeamDomain
        Provides ``nelx``, ``nely``, ``ndof``, ``get_fixed_dofs()``,
        ``get_load_vector()``.  If it also exposes ``generate_edofMat()``
        (i.e. it is a MeshDomain / BeamDomain), that method is used
        directly instead of re-building the DOF map inline.
    penal : float
        SIMP penalisation exponent.
    Emin : float
        Minimum Young's modulus (void elements, avoids singularity).
    Emax : float
        Maximum Young's modulus (solid elements).
    """

    def __init__(
        self,
        problem: TopOptSolver,
        penal: float = 3.0,
        Emin:  float = 1e-9,
        Emax:  float = 1.0,
    ):
        self.problem = problem
        self.penal   = penal
        self.Emin    = Emin
        self.Emax    = Emax

        nelx, nely = problem.nelx, problem.nely
        ndof       = problem.ndof

        # Element stiffness matrix (constant for the whole run)
        self.KE = lk()

        # ------------------------------------------------------------------
        # Build element → DOF map (edofMat)
        # Delegate to MeshDomain.generate_edofMat() when available so the
        # numbering is always consistent with node_index / element_index.
        # ------------------------------------------------------------------
        if hasattr(problem, "generate_edofMat"):
            self.edofMat = problem.generate_edofMat()
        else:
            edofMat = np.zeros((nelx * nely, 8), dtype=int)
            for elx in range(nelx):
                for ely in range(nely):
                    el = ely + elx * nely
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edofMat[el, :] = [
                        2*n1+2, 2*n1+3,
                        2*n2+2, 2*n2+3,
                        2*n2,   2*n2+1,
                        2*n1,   2*n1+1,
                    ]
            self.edofMat = edofMat

        # COO index vectors for fast sparse K assembly
        self.iK = np.kron(self.edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(self.edofMat, np.ones((1, 8))).flatten()

        # Fixed / free DOFs and load vector
        self.fixed = problem.get_fixed_dofs()
        self.free  = np.setdiff1d(np.arange(ndof), self.fixed)
        self.f     = problem.get_load_vector()

    # ------------------------------------------------------------------
    def solve(self, xPhys: np.ndarray) -> np.ndarray:
        """Solve Ku = f for the physical density field *xPhys*.

        Returns
        -------
        u : np.ndarray, shape (ndof, 1)
            Global displacement vector.
        """
        ndof = self.problem.ndof
        sK = (
            (self.KE.flatten()[np.newaxis]).T
            * (self.Emin + xPhys ** self.penal * (self.Emax - self.Emin))
        ).flatten(order="F")

        K = coo_matrix(
            (sK, (self.iK, self.jK)), shape=(ndof, ndof)
        ).tocsc()
        K = K[self.free, :][:, self.free]

        u = np.zeros((ndof, 1))
        u[self.free, 0] = spsolve(K, self.f[self.free, 0])
        return u

    # ------------------------------------------------------------------
    def sensitivities(
        self, xPhys: np.ndarray, u: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute compliance objective and its sensitivities.

        Parameters
        ----------
        xPhys : np.ndarray, shape (nelx*nely,)
            Physical (filtered) densities.
        u : np.ndarray, shape (ndof, 1)
            Displacement field returned by :meth:`solve`.

        Returns
        -------
        obj : float
            Compliance (scalar objective).
        dc : np.ndarray, shape (nelx*nely,)
            Sensitivity of compliance w.r.t. element densities.
        dv : np.ndarray, shape (nelx*nely,)
            Sensitivity of volume w.r.t. element densities (all ones).
        """
        ue = u[self.edofMat].reshape(-1, 8)
        ce = (ue @ self.KE * ue).sum(axis=1)

        obj = float(
            ((self.Emin + xPhys ** self.penal * (self.Emax - self.Emin)) * ce).sum()
        )
        dc = (-self.penal * xPhys ** (self.penal - 1) * (self.Emax - self.Emin)) * ce
        dv = np.ones(self.problem.nelx * self.problem.nely)
        return obj, dc, dv
