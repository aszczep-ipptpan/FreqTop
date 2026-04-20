from abc import ABC, abstractmethod
import numpy as np


def checkerboard_init(nelx: int, nely: int, block_size: int = 10) -> np.ndarray:
    """Create a 0-1 checkerboard initial density field.

    Alternating solid (1.0) and void (0.0) rectangular blocks of
    ``block_size × block_size`` elements.  The top-left block is solid.

    Element indexing follows the column-major convention used throughout
    FreqTop: element index ``el = elx * nely + ely``.

    Parameters
    ----------
    nelx, nely : int
        Mesh dimensions (number of elements in x and y directions).
    block_size : int
        Side length of each square block in elements (default 10).

    Returns
    -------
    x : np.ndarray, shape (nelx * nely,)
        Density field with values 0.0 or 1.0 in a checkerboard pattern.
    """
    elx = np.arange(nelx)
    ely = np.arange(nely)
    # block coordinates for every element position
    block_x = elx // block_size          # shape (nelx,)
    block_y = ely // block_size          # shape (nely,)
    # (nelx, nely) grid — [i, j] = density for element (elx=i, ely=j)
    # C-order ravel gives index i*nely + j, matching el = elx*nely + ely
    grid = ((block_x[:, np.newaxis] + block_y[np.newaxis, :]) % 2 == 0).astype(float)
    return grid.ravel()


class TopOptSolver(ABC):
    """Abstract base class for topology optimisation problems.

    Subclasses define the domain size, boundary conditions, and applied loads
    for a specific structural problem.
    """

    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely

    @abstractmethod
    def get_fixed_dofs(self) -> np.ndarray:
        """Return indices of constrained (fixed) degrees of freedom."""
        pass

    @abstractmethod
    def get_load_vector(self) -> np.ndarray:
        """Return the global load vector f of shape (ndof, 1)."""
        pass

    @property
    def ndof(self) -> int:
        return 2 * (self.nelx + 1) * (self.nely + 1)

    # ------------------------------------------------------------------
    # Passive-region interface — default: no passive elements.
    # BeamDomain overrides all three with real implementations.
    # ------------------------------------------------------------------

    def has_passive_elements(self) -> bool:
        """Return True if this problem defines passive (fixed-density) elements."""
        return False

    def get_passive_elements(self) -> np.ndarray:
        """Return sorted array of passive element indices (empty by default)."""
        return np.empty(0, dtype=int)

    def get_active_elements(self) -> np.ndarray:
        """Return sorted array of active element indices (all elements by default)."""
        return np.arange(self.nelx * self.nely, dtype=int)
