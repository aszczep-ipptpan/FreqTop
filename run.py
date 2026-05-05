"""
run.py — command-line entry point for FreqTop with profiling and plotting.

Modes
-----
JSON mode (default when parameters.json exists):
    python run.py [method] [max_iter]

    Reads all domain / BC / material settings from parameters.json.
    Passive flanges, self-weight, and support conditions are activated
    automatically via BeamDomain.

Legacy CLI mode:
    python run.py nelx nely volfrac rmin penal ft [method]

    Positional floats / ints trigger the old CantileverProblem path
    (no passive regions, no JSON).

Arguments
---------
JSON mode
    method   str   BOTH   "OC", "SQP", or "BOTH"
    max_iter int   70     maximum iterations (overrides JSON value if given)

Legacy CLI mode
    nelx     int   180
    nely     int    60
    volfrac  float  0.4
    rmin     float  5.4
    penal    float  3.0
    ft       int     1    filter type: 0=sensitivity, 1=density
    method   str   BOTH

Outputs (written to ./outputs/)
--------------------------------
    density_OC_iter*.png / density_SQP_iter*.png
    convergence_*.png, cumtime_*.png, cummem_*.png,
    stage_breakdown_*.png, comparison_oc_vs_sqp.png
    density_animation_*.gif, cumtime_animation_both.gif
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs

from FreqTop.solver            import TopOptSolver
from FreqTop.fe.fe_solver      import FESolver
from FreqTop.filters.sensitivity import SensitivityFilter
from FreqTop.filters.density     import DensityFilter
from FreqTop.optimizers.oc       import OCOptimizer
from FreqTop.optimizers.sqp      import SQPOptimizer
from FreqTop.callbacks.logger    import ConsoleLogger
from FreqTop.utils.base          import checkerboard_init

from FreqTop.profiling import AlgorithmProfiler
from FreqTop.profiling.instrumented import ProfiledFESolver, ProfiledOptimizer
from FreqTop.callbacks.profiling_callback import ProfilingCallback
from FreqTop.viz import TopOptPlotter, PlotterConfig, TOPOPT_SIMP_LATEX

# ---------------------------------------------------------------------------
# Detect run mode: JSON vs. legacy CLI
# ---------------------------------------------------------------------------

_JSON_FILE = "parameters.json"

# Sweep mode is triggered by passing "SWEEP" as the first argument.
# It bypasses JSON/legacy routing entirely.
#_sweep_mode = len(sys.argv) > 1 and sys.argv[1].upper() == "SWEEP"
_sweep_mode = False

_use_json  = (not _sweep_mode) and os.path.exists(_JSON_FILE) and (
    len(sys.argv) < 2 or not sys.argv[1].lstrip("-").isdigit()
)

if not _sweep_mode:
    if _use_json:
        # ── JSON mode ──────────────────────────────────────────────────────────
        from FreqTop.config.loader import load_parameters, make_beam_domain

        params   = load_parameters(_JSON_FILE)
        opt_cfg  = params["optimisation"]
        domain   = make_beam_domain(_JSON_FILE)

        nelx     = domain.nelx
        nely     = domain.nely
        volfrac  = float(opt_cfg["volume_fraction"])
        rmin     = float(opt_cfg["filter_radius"]) * float(params["domain"]["size"]["length"])
        penal    = float(opt_cfg["penalization"])
        ft       = 1 if opt_cfg.get("filter_type", "density") == "heaviside" else 0
        max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else int(opt_cfg["max_iters"])
        tol      = float(opt_cfg["convergence_tol"])
        mode     = sys.argv[1].upper() if len(sys.argv) > 1 else "SQP"

        problem_label = (
            f"{params['meta']['name']}  {nelx}x{nely}"
            f"  vf={volfrac}  p={penal}"
        )

        def _make_problem():
            return domain   # shared; BeamDomain is stateless after __init__

    else:
        # ── Legacy CLI mode ────────────────────────────────────────────────────
        from FreqTop.utils.cantilever import CantileverProblem

        def _arg(idx, default, cast=float):
            return cast(sys.argv[idx]) if len(sys.argv) > idx else default

        nelx     = _arg(1, 180, int)
        nely     = _arg(2,  60, int)
        volfrac  = _arg(3, 0.4, float)
        rmin     = _arg(4, 5.4, float)
        penal    = _arg(5, 3.0, float)
        ft       = _arg(6, 1,   int)
        mode     = sys.argv[7].upper() if len(sys.argv) > 7 else "BOTH"
        max_iter = 20
        tol      = 0.01

        problem_label = f"Cantilever {nelx}x{nely}  vf={volfrac}  p={penal}"

        def _make_problem():
            return CantileverProblem(nelx=nelx, nely=nely)


# ---------------------------------------------------------------------------
# Common setup (skipped in sweep mode)
# ---------------------------------------------------------------------------

if not _sweep_mode:
    print("=" * 60)
    print("  FreqTop — Topology Optimisation with Profiling & Plotting")
    print("=" * 60)
    print(f"  Mode    : {'JSON (' + _JSON_FILE + ')' if _use_json else 'Legacy CLI'}")
    print(f"  Domain  : {nelx} x {nely}")
    print(f"  volfrac : {volfrac}  rmin : {rmin:.3g}  penal : {penal}")
    print(f"  Filter  : {'Density/Heaviside' if ft else 'Sensitivity'}")
    print(f"  Method  : {mode}  max_iter : {max_iter}")
    if _use_json:
        prob = _make_problem()
        if prob.has_passive_elements():
            n_pass = len(prob.get_passive_elements())
            print(f"  Passive : {n_pass} elements ({n_pass/prob.nelxy:.1%} of domain)")
    print("=" * 60)


def _make_filter(nelx, nely, rmin, ft):
    return (SensitivityFilter if ft == 0 else DensityFilter)(nelx, nely, rmin)


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

# Four meshes, all 10:1 (length:height) aspect ratio, progressively refined.
_SWEEP_MESHES = [
    (200, 20, "200x20"),
]

_SWEEP_MOVES        = [0.4, 0.6, 0.8, 1.0]
_SWEEP_VOLFRAC      = 0.4
_SWEEP_PENAL        = 3.0
_SWEEP_FT           = 1       # density filter
_SWEEP_MAX_ITER     = 100
_SWEEP_TOL          = [1e-0, 1e-1]


def _run_sweep_case(
    *,
    case_name: str,
    nelx: int,
    nely: int,
    rmin: float,
    volfrac: float,
    penal: float,
    ft: int,
    max_iter: int,
    tol: float,
    move: float,
    init_type: str,   # "uniform" | "checkerboard"
) -> tuple[str, str]:
    """Run OC then SQP for one sweep case, write shared outputs, return summaries.

    Both methods share the same output directory and produce the same file set
    as the normal ``BOTH`` mode — density snapshots, convergence, volume fraction,
    cumulative time/memory, stage breakdown, comparison plot, and animations.
    """
    from FreqTop.utils.cantilever import CantileverProblem

    x_init = (
        checkerboard_init(nelx, nely, block_size=10)
        if init_type == "checkerboard"
        else None
    )

    oc_profiler = sqp_profiler = None
    oc_cb       = sqp_cb       = None

    for method in ("OC", "SQP"):
        print(f"\n{'-'*60}")
        print(f"  Running {method}  [{case_name}]")
        print(f"{'-'*60}")

        problem = CantileverProblem(nelx=nelx, nely=nely)
        fe      = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
        filt    = _make_filter(nelx, nely, rmin, ft)

        raw_opt = (
            OCOptimizer(move=move)
            if method == "OC"
            else SQPOptimizer(move=move, penal=penal)
        )

        profiler     = AlgorithmProfiler(method=method)
        profiled_fe  = ProfiledFESolver(fe, profiler)
        profiled_opt = ProfiledOptimizer(raw_opt, profiler)

        prof_cb = ProfilingCallback(profiler, collect_density=True, collect_objective=True)
        logger  = ConsoleLogger(nelx=nelx, nely=nely, volfrac=volfrac)

        solver = TopOptSolver(
            problem   = problem,
            fe_solver = profiled_fe,
            filter    = filt,
            optimizer = profiled_opt,
            volfrac   = volfrac,
            callbacks = (logger, prof_cb),
            max_iter  = max_iter,
            tol       = tol,
            x_init    = x_init,
        )
        solver.run()
        print(f"\n{profiler.summary()}")

        if method == "OC":
            oc_profiler, oc_cb = profiler, prof_cb
        else:
            sqp_profiler, sqp_cb = profiler, prof_cb

    # Shared output directory — no method suffix, mirrors the BOTH mode layout.
    output_dir = os.path.join("outputs_sweep", case_name)
    sweep_cfg = PlotterConfig(
        output_dir       = output_dir,
        dpi              = 100,
        animation_fps    = 5,
        animation_format = "gif",
        title_prefix     = TOPOPT_SIMP_LATEX,
        show_grid        = True,
        font_size        = 10,
    )

    problem_label = f"{case_name}  vf={volfrac}  p={penal}  move={move}"

    density_hist = {}
    oc_obj = sqp_obj = None
    if oc_cb:
        density_hist["OC"] = oc_cb.density_history
        oc_obj = oc_cb.obj_history
    if sqp_cb:
        density_hist["SQP"] = sqp_cb.density_history
        sqp_obj = sqp_cb.obj_history

    plotter = TopOptPlotter(
        config          = sweep_cfg,
        oc_profiler     = oc_profiler,
        sqp_profiler    = sqp_profiler,
        density_history = density_hist,
        problem_label   = problem_label,
    )

    plotter.render_all(nelx=nelx, nely=nely, oc_obj=oc_obj, sqp_obj=sqp_obj)

    for m, hist in density_hist.items():
        if hist:
            plotter.animate_density(method=m, nelx=nelx, nely=nely, save=True)

    plotter.animate_cumulative_time(method=None, save=True)

    print(f"  Results saved to: {output_dir}/")
    return oc_profiler.summary(), sqp_profiler.summary()


def run_sweep() -> list[tuple[str, tuple | None]]:
    """Parameter sweep over move limits, mesh sizes, and initialisation types.

    Dimensions tested
    -----------------
    * meshes : 100x10, 300x30, 500x50, 1000x100  (all 10:1 aspect ratio)
    * tol    : 1e-0, 1e-1, 1e-2, 1e-3
    * move   : 0.2, 0.4, 0.6, 0.8, 1.0
    * init   : uniform (plain volfrac), checkerboard (10x10 blocks, 0/1)

    Both OC and SQP run inside each case and share one output directory,
    producing the same file set as the normal ``BOTH`` mode, e.g.::

        outputs_sweep/100x10_tol1e-02_move0.5_checkerboard/

    MemoryError handling
    --------------------
    A ``MemoryError`` inside any case is caught, a ``[SKIP]`` warning is
    printed, and the sweep continues with the next case.  Other unexpected
    exceptions are handled the same way.

    Returns
    -------
    results : list of (case_name, (oc_summary, sqp_summary) | None)
        ``None`` indicates the case was skipped due to an error.
    """
    results: list[tuple[str, tuple | None]] = []
    total = (
        len(_SWEEP_MESHES) * len(_SWEEP_TOL) * len(_SWEEP_MOVES) * 2
    )  # mesh × tol × move × init
    done  = 0

    for nelx_s, nely_s, mesh_label in _SWEEP_MESHES:
        # Scale filter radius with mesh density (≈ 4 % of nelx keeps the
        # relative stencil size constant across refinement levels).
        rmin_s = max(2.0, 0.04 * nelx_s)

        for tol_val in _SWEEP_TOL:
            for move_val in _SWEEP_MOVES:
                for init_type in ("uniform", "checkerboard"):
                    done += 1
                    case_name = (
                        f"{mesh_label}_tol{tol_val:.0e}"
                        f"_move{move_val:.1f}_{init_type}"
                    )
                    print(f"\n{'='*60}")
                    print(f"  SWEEP [{done}/{total}]: {case_name}")
                    print(f"  mesh={nelx_s}x{nely_s}  rmin={rmin_s:.1f}"
                          f"  vf={_SWEEP_VOLFRAC}  move={move_val}"
                          f"  tol={tol_val:.0e}")
                    print(f"{'='*60}")

                    try:
                        summaries = _run_sweep_case(
                            case_name = case_name,
                            nelx      = nelx_s,
                            nely      = nely_s,
                            rmin      = rmin_s,
                            volfrac   = _SWEEP_VOLFRAC,
                            penal     = _SWEEP_PENAL,
                            ft        = _SWEEP_FT,
                            max_iter  = _SWEEP_MAX_ITER,
                            tol       = tol_val,
                            move      = move_val,
                            init_type = init_type,
                        )
                        results.append((case_name, summaries))

                    except MemoryError as exc:
                        print(f"\n  [SKIP] MemoryError in {case_name}: {exc}")
                        results.append((case_name, None))

                    except Exception as exc:  # noqa: BLE001
                        print(f"\n  [SKIP] {type(exc).__name__} in {case_name}: {exc}")
                        results.append((case_name, None))

    succeeded = sum(s is not None for _, s in results)
    print(f"\nSweep complete — {succeeded}/{len(results)} cases succeeded.")
    print(f"All sweep outputs written to: outputs_sweep/")
    return results


if _sweep_mode:
    # ── Sweep mode ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("  FreqTop — Parameter Sweep")
    print(f"  Meshes : {[lbl for *_, lbl in _SWEEP_MESHES]}")
    print(f"  Moves  : {_SWEEP_MOVES}")
    print(f"  Init   : uniform + checkerboard (10x10 blocks)")
    print(f"  Method : OC + SQP  (both per case, BOTH-mode output)")
    print(f"  Output : outputs_sweep/<mesh>_move<val>_<init>/")
    print("=" * 60)
    run_sweep()

else:
    # ── Normal (JSON / Legacy CLI) mode ────────────────────────────────────

    cfg = PlotterConfig(
        output_dir       = "outputs",
        dpi              = 120,
        animation_fps    = 6,
        animation_format = "gif",
        title_prefix     = TOPOPT_SIMP_LATEX,
        show_grid        = True,
        font_size        = 11,
    )

    # "uniform" — plain volfrac field (original behaviour)
    # "checkerboard" — 0/1 checkerboard with 10x10-element blocks
    INIT_TYPE = "uniform"

    def run_method(method: str, init_type: str = INIT_TYPE) -> tuple:
        """Construct solver, wrap with profiler, run, return (profiler, cb)."""
        print(f"\n{'-'*60}")
        print(f"  Running {method} optimiser  (max_iter={max_iter}, init={init_type})")
        print(f"{'-'*60}")

        x_init = (
            checkerboard_init(nelx, nely, block_size=10)
            if init_type == "checkerboard"
            else None
        )

        problem = _make_problem()
        fe      = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
        filt    = _make_filter(nelx, nely, rmin, ft)

        raw_opt = OCOptimizer(move=0.8) if method == "OC" else SQPOptimizer(move=0.8, penal=penal)

        profiler     = AlgorithmProfiler(method=method)
        profiled_fe  = ProfiledFESolver(fe, profiler)
        profiled_opt = ProfiledOptimizer(raw_opt, profiler)

        prof_cb = ProfilingCallback(profiler, collect_density=True, collect_objective=True)
        logger  = ConsoleLogger(nelx=nelx, nely=nely, volfrac=volfrac)

        solver = TopOptSolver(
            problem   = problem,
            fe_solver = profiled_fe,
            filter    = filt,
            optimizer = profiled_opt,
            volfrac   = volfrac,
            callbacks = (logger, prof_cb),
            max_iter  = max_iter,
            tol       = tol,
            x_init    = x_init,
        )
        solver.run()

        print(f"\n{profiler.summary()}")
        return profiler, prof_cb

    # ── Run optimiser(s) ───────────────────────────────────────────────────

    oc_profiler = sqp_profiler = None
    oc_cb       = sqp_cb       = None

    if mode in ("OC", "BOTH"):
        oc_profiler, oc_cb = run_method("OC")

    if mode in ("SQP", "BOTH"):
        sqp_profiler, sqp_cb = run_method("SQP")

    # ── Collect results and render plots ───────────────────────────────────

    density_history = {}
    oc_obj = sqp_obj = None

    if oc_cb:
        density_history["OC"] = oc_cb.density_history
        oc_obj = oc_cb.obj_history

    if sqp_cb:
        density_history["SQP"] = sqp_cb.density_history
        sqp_obj = sqp_cb.obj_history

    plotter = TopOptPlotter(
        config          = cfg,
        oc_profiler     = oc_profiler,
        sqp_profiler    = sqp_profiler,
        density_history = density_history,
        problem_label   = problem_label,
    )

    print("\nRendering static plots ...")
    plotter.render_all(nelx=nelx, nely=nely, oc_obj=oc_obj, sqp_obj=sqp_obj, oc_change=oc_cb.change_history if oc_cb else None, sqp_change=sqp_cb.change_history if sqp_cb else None)

    print("Rendering animations ...")
    for m in [m for m in ("OC", "SQP") if density_history.get(m)]:
        print(f"  density animation [{m}] ...")
        plotter.animate_density(method=m, nelx=nelx, nely=nely, save=True)

    print("  cumulative-time animation [both] ...")
    plotter.animate_cumulative_time(method=None, save=True)

    print(f"\nAll outputs written to:  {cfg.output_dir}/")
    print("Done.")
