import numpy as np


def lk(E: float = 1.0, nu: float = 0.3) -> np.ndarray:
    """Return the 8×8 element stiffness matrix for a unit square
    plane-stress Q4 element with Young's modulus *E* and Poisson's
    ratio *nu*.

    This is the direct extraction of the ``lk()`` function from the
    original 165-line code, kept parameter-free at call sites by
    using the same defaults (E=1, nu=0.3).
    """
    k = np.array([
        1 / 2 - nu / 6,
        1 / 8 + nu / 8,
        -1 / 4 - nu / 12,
        -1 / 8 + 3 * nu / 8,
        -1 / 4 + nu / 12,
        -1 / 8 - nu / 8,
        nu / 6,
        1 / 8 - 3 * nu / 8,
    ])
    KE = E / (1 - nu ** 2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE
