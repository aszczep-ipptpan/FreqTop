import json


def load_parameters(path: str = "parameters.json") -> dict:
    with open(path, "r") as f:
        params = json.load(f)
    return params


def map_to_run_args(params: dict) -> dict:
    """
    Mapuje parameters.json → parametry używane w run.py
    """

    return {
        "nelx": params["domain"]["mesh"]["nelx"],
        "nely": params["domain"]["mesh"]["nely"],
        "volfrac": params["optimisation"]["volume_fraction"],
        "rmin": params["optimisation"]["filter_radius"] * params["domain"]["size"]["length"],
        "penal": params["optimisation"]["penalization"],
        "ft": 1 if params["optimisation"]["filter_type"] == "heaviside" else 0,
        "max_iter": params["optimisation"]["max_iters"],
        "tol": params["optimisation"]["convergence_tol"]
    }