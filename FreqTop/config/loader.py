"""FreqTop/config/loader.py — JSON parameter loading and domain factory."""

import copy
import json


def load_parameters(path: str = "parameters.json") -> dict:
    """Load and return the raw JSON parameter dict from *path*."""
    with open(path, "r", encoding="utf-8") as f:
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


def apply_overrides(params: dict, overrides: dict) -> dict:
    """Return a deep copy of *params* with dot-notation key overrides applied.

    Each key in *overrides* is a dot-separated path into the nested params
    dict (e.g. ``"domain.mesh.nelx"``).  Any JSON-serialisable value is
    accepted.

    Example
    -------
    >>> p = apply_overrides(params, {"domain.mesh.nelx": 500,
    ...                              "domain.mesh.nely": 50})
    """
    result = copy.deepcopy(params)
    for dotted_key, value in overrides.items():
        keys   = dotted_key.split(".")
        target = result
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    return result


def make_beam_domain_from_dict(params: dict):
    """Factory: create a BeamDomain from an already-loaded parameters dict.

    Avoids re-reading the JSON file when the caller has already applied
    overrides (e.g. during a parameter sweep).
    """
    from ..fe.domain import BeamDomain
    return BeamDomain.from_dict(params)
