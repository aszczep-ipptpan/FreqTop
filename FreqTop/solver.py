import numpy as np

from .utils.base import TopOptSolver as Problem
from .fe.fe_solver import FESolver
from .filters.base import Filter
from .optimizers.base import Optimizer


class TopOptSolver:
    """Orchestrates the topology optimisation loop.

    Parameters
    ----------
    problem : TopOptSolver (Problem ABC)
        Defines domain, BCs, and loads.
    fe_solver : FESolver
        Assembles and solves the FE system; computes sensitivities.
    filter : Filter
        Filters sensitivities and maps design vars to physical densities.
    optimizer : Optimizer
        Updates design variables given sensitivities.
    volfrac : float
        Target volume fraction.
    callbacks : sequence of callable
        Zero or more callbacks called each iteration with
        signature ``(loop, obj, xPhys, change)``.
    max_iter : int
        Hard upper bound on number of iterations.
    tol : float
        Convergence tolerance on the inf-norm design change.
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
    ):
        self.problem = problem
        self.fe_solver = fe_solver
        self.filter = filter
        self.optimizer = optimizer
        self.volfrac = volfrac
        self.callbacks = callbacks
        self.max_iter = max_iter
        self.tol = tol

    def run(self) -> np.ndarray:
        """Execute the optimisation loop.

        Returns
        -------
        xPhys : np.ndarray, shape (nelx*nely,)
            Final physical density field.
        """
        nelx, nely = self.problem.nelx, self.problem.nely
        n = nelx * nely

        x = np.full(n, self.volfrac)
        xPhys = self.filter.filter_design(x)

        change = 1.0
        loop = 0

        time_array = []

        while change > self.tol and loop < self.max_iter:
            loop += 1

            u = self.fe_solver.solve(xPhys)
            obj, dc, dv = self.fe_solver.sensitivities(xPhys, u)
            dc, dv = self.filter.filter_sensitivities(x, dc, dv)
            x_new, time = self.optimizer.update(x, dc, dv, self.volfrac)
            xPhys = self.filter.filter_design(x_new)
            change = float(np.linalg.norm(x_new - x, np.inf))
            x = x_new

            time_array.append(time)

            for cb in self.callbacks:
                cb(loop, obj, xPhys, change, time)

        return xPhys
