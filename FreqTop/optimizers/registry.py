"""
FreqTop/optimizers/registry.py
================================
Optimizer registry — the single place that maps method names to constructors.

Open/Closed Principle
---------------------
To add a new optimizer (e.g. MMA):
  1. Add one entry to OPTIMIZER_REGISTRY.
  2. Optionally add its name to HESSIAN_METHODS if it uses second-order information.
  3. Nothing else changes — run.py, plotter.py, and evaluator.py are unaware.

Usage
-----
    from FreqTop.optimizers.registry import make_optimizer, resolve_methods

    opt = make_optimizer("SQP", move=0.8, penal=3.0)
    methods = resolve_methods("BOTH")   # -> ["OC", "SQP"]
    methods = resolve_methods("OC")     # -> ["OC"]
"""

from __future__ import annotations

from typing import Callable

from FreqTop.optimizers.oc  import OCOptimizer
from FreqTop.optimizers.sqp import SQPOptimizer


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps method name -> factory(move, penal) -> optimizer instance.
#: Insertion order defines the execution order when mode == "BOTH".
OPTIMIZER_REGISTRY: dict[str, Callable] = {
    "OC":  lambda move, penal: OCOptimizer(move=move),
    "SQP": lambda move, penal: SQPOptimizer(move=move, penal=penal),
}

#: Methods that use a second-order (Hessian-based) update step.
#: Used to populate RunResult.hessian_used without string comparisons.
HESSIAN_METHODS: frozenset[str] = frozenset({"SQP"})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def make_optimizer(method: str, move: float, penal: float):
    """Instantiate the optimizer registered under *method*.

    Parameters
    ----------
    method : str
        Key in OPTIMIZER_REGISTRY (e.g. "OC", "SQP").
    move : float
        Move limit passed to the optimizer constructor.
    penal : float
        SIMP penalization exponent (used by second-order methods).

    Raises
    ------
    ValueError
        When *method* is not present in OPTIMIZER_REGISTRY.
    """
    if method not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{method}'. "
            f"Available: {list(OPTIMIZER_REGISTRY)}"
        )
    return OPTIMIZER_REGISTRY[method](move, penal)


def resolve_methods(mode: str) -> list[str]:
    """Expand a mode string to an ordered list of method names.

    Parameters
    ----------
    mode : str
        "BOTH"  — all keys in OPTIMIZER_REGISTRY (insertion order).
        "<KEY>" — any single registered method name.

    Returns
    -------
    list[str]
        Ordered method names to run.

    Raises
    ------
    ValueError
        When *mode* is neither "BOTH" nor a registered method key.
    """
    if mode == "BOTH":
        return list(OPTIMIZER_REGISTRY)
    if mode in OPTIMIZER_REGISTRY:
        return [mode]
    raise ValueError(
        f"Unknown mode '{mode}'. "
        f"Use 'BOTH' or one of: {list(OPTIMIZER_REGISTRY)}"
    )
