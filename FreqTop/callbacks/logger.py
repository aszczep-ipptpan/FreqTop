import numpy as np
from .base import Callback


class ConsoleLogger(Callback):
    """Prints the iteration history to stdout — identical to the original
    165-line code's print statement inside the loop.

    Parameters
    ----------
    nelx, nely : int
        Domain dimensions (needed to compute the current volume fraction).
    volfrac : float
        Target volume fraction (used in the vol. display formula).
    """

    def __init__(self, nelx: int, nely: int, volfrac: float):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
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

        print(
            f"it.: {loop:4d} , obj.: {obj:.3f}  "
            f"Vol.: {vol:.3f}, ch.: {change:.3f}, time: {time:.3f}, cum time: {sum(self.time_array):.3f}"
        )

    def update(self, loop, obj, vol, change, time):
        self.time_array.append(time)
        self.cum_time.append(sum(self.time_array))

