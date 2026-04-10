import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from base import Callback


class LivePlotter(Callback):
    """Displays a live grey-scale image of the density field updated every
    iteration — identical behaviour to the original 165-line code.

    Parameters
    ----------
    nelx, nely : int
        Domain dimensions (needed to reshape the flat density vector).
    """

    def __init__(self, nelx: int, nely: int):
        self.nelx = nelx
        self.nely = nely
        plt.ion()
        self.fig, ax = plt.subplots()
        dummy = np.zeros((nelx, nely))
        self.im = ax.imshow(
            -dummy.T,
            cmap="gray",
            interpolation="none",
            norm=colors.Normalize(vmin=-1, vmax=0),
        )
        self.fig.show()

    def __call__(
        self,
        loop: int,
        obj: float,
        xPhys: np.ndarray,
        change: float,
    ) -> None:
        self.im.set_array(-xPhys.reshape((self.nelx, self.nely)).T)
        self.fig.canvas.draw()

    def keep_open(self) -> None:
        """Block until the plot window is closed (call after solver.run())."""
        plt.show()
