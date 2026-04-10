from abc import ABC, abstractmethod
import numpy as np


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
