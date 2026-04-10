import numpy as np
from scipy.sparse import coo_matrix


def build_filter_matrix(nelx: int, nely: int, rmin: float):
    """Build the weighted neighbourhood matrix H and its row sums Hs.

    Used by both the sensitivity filter and the density filter.

    Parameters
    ----------
    nelx, nely : int
        Number of elements in x and y directions.
    rmin : float
        Filter radius (in element units).

    Returns
    -------
    H : scipy.sparse.csc_matrix, shape (nelx*nely, nelx*nely)
    Hs : np.ndarray, shape (nelx*nely, 1)
        Row sums of H.
    """
    nfilter = int(nelx * nely * ((2 * (np.ceil(rmin) - 1) + 1) ** 2))
    iH = np.zeros(nfilter)
    jH = np.zeros(nfilter)
    sH = np.zeros(nfilter)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            kk1 = int(np.maximum(i - (np.ceil(rmin) - 1), 0))
            kk2 = int(np.minimum(i + np.ceil(rmin), nelx))
            ll1 = int(np.maximum(j - (np.ceil(rmin) - 1), 0))
            ll2 = int(np.minimum(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k * nely + l
                    fac = rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = np.maximum(0.0, fac)
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx * nely, nelx * nely)).tocsc()
    Hs = H.sum(1)
    return H, Hs
