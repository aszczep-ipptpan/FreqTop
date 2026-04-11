import numpy as np
from .base import TopOptSolver


class CantileverProblem(TopOptSolver):
    """Cantilever beam fixed on the left edge with a downward point load
    at the middle of the right edge — the problem baked into the original
    165-line code.

    Boundary conditions
    -------------------
    Fixed : all horizontal DOFs on the left edge  +  the bottom-right corner
            vertical DOF (prevents rigid-body rotation).
    Load  : unit downward force at the midpoint of the right edge.
    """

    def get_fixed_dofs(self) -> np.ndarray:
        nelx, nely = self.nelx, self.nely
        dofs = np.arange(self.ndof)
        # Left edge: every other DOF starting at 0 (x-displacement of each
        # node on the left column) → indices 0, 2, 4, …, 2*(nely+1)-2
        left_edge_x = dofs[0 : 2 * (nely + 1) : 2]
        # Bottom-right corner y-DOF: last DOF in the global vector
        bottom_right_y = np.array([self.ndof - 1])
        return np.union1d(left_edge_x, bottom_right_y)

    def get_load_vector(self) -> np.ndarray:
        f = np.zeros((self.ndof, 1))
        # Downward unit load at node 0 (top-left of the right edge in the
        # original indexing), DOF index 1 → f[1] = -1
        f[1, 0] = -1.0
        return f
