"""FreqTop/config/loader.py — JSON parameter loading and domain factory."""

import json


def load_parameters(path: str = "parameters.json") -> dict:
    """Load and return the raw JSON parameter dict from *path*."""
    with open(path, "r") as f:
        return json.load(f)


def make_beam_domain(path: str = "parameters.json"):
    """
    Factory: create a fully configured BeamDomain from a JSON parameters file.

    This is the preferred entry point for JSON-driven problems.  It reads
    the ``domain``, ``bc``, and ``materials`` sections and wires everything
    up — passive flanges, supports, and self-weight included.

    Parameters
    ----------
    path : str
        Path to a parameters JSON file (default: ``"parameters.json"``).

    Returns
    -------
    BeamDomain

    Example
    -------
    ::

        from FreqTop.config.loader import make_beam_domain
        from FreqTop.fe.fe_solver  import FESolver

        domain = make_beam_domain("parameters.json")
        fe     = FESolver(domain, penal=3.0)
    """
    # Import here to avoid circular imports at module load time
    from ..fe.domain import BeamDomain
    return BeamDomain.from_json(path)


def map_to_run_args(params: dict) -> dict:
    """
    Map a raw parameters dict to keyword arguments for TopOptSolver.run().

    Useful when driving the solver from a JSON file without BeamDomain.
    """
    return {
        "nelx":     params["domain"]["mesh"]["nelx"],
        "nely":     params["domain"]["mesh"]["nely"],
        "volfrac":  params["optimisation"]["volume_fraction"],
        "rmin":     (params["optimisation"]["filter_radius"]
                     * params["domain"]["size"]["length"]),
        "penal":    params["optimisation"]["penalization"],
        "ft":       1 if params["optimisation"]["filter_type"] == "heaviside" else 0,
        "max_iter": params["optimisation"]["max_iters"],
        "tol":      params["optimisation"]["convergence_tol"],
    }
