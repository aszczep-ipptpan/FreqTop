import numpy as np
from .base import Callback


class ConsoleLogger(Callback):
    """Prints the iteration history to stdout.

    Parameters
    ----------
    nelx, nely : int
        Domain dimensions (needed to compute the current volume fraction).
    volfrac : float
        Target volume fraction (used in the vol. display formula).
    problem_type : str
        ``"min_compliance"`` (default) or ``"max_frequency"``.
        Controls the label printed next to the objective value.
    """

    def __init__(
        self,
        nelx: int,
        nely: int,
        volfrac: float,
        problem_type: str = "min_compliance",
    ):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.problem_type = problem_type
        self.time_array = []
        self.cum_time = []

    def __call__(
        self,
        loop: int,
        obj: float,
        xPhys: np.ndarray,
        change: float,
        time: float,
    ) -> None:
        vol = xPhys.sum() / (self.nelx * self.nely)

        self.update(loop, obj, vol, change, time)

        if self.problem_type == "max_frequency":
            obj_label = f"w1 [rad/s]: {obj:.6f}"
        else:
            obj_label = f"obj.: {obj:.3f}"

        cum = sum(self.time_array)
        print(
            f"it.: {loop:4d} , {obj_label}  "
            f"Vol.: {vol:.3f}, ch.: {change:.3f}, "
            f"time: {time:.3f}, cum time: {cum:.3f}"
        )

    def update(self, loop, obj, vol, change, time):
        self.time_array.append(time)
        self.cum_time.append(sum(self.time_array))
