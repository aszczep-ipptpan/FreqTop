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

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs

from FreqTop.solver            import TopOptSolver
from FreqTop.fe.fe_solver      import FESolver
from FreqTop.filters.sensitivity import SensitivityFilter
from FreqTop.filters.density     import DensityFilter
from FreqTop.optimizers.registry import make_optimizer, resolve_methods, HESSIAN_METHODS
from FreqTop.runner_types        import RunResult, SimulationConfig
from FreqTop.callbacks.logger    import ConsoleLogger

from FreqTop.profiling import AlgorithmProfiler
from FreqTop.profiling.instrumented import ProfiledFESolver, ProfiledOptimizer
from FreqTop.callbacks.profiling_callback import ProfilingCallback
from FreqTop.viz import TopOptPlotter, PlotterConfig, TOPOPT_SIMP_LATEX, TOPOPT_FREQ_LATEX

from FreqTop.config.loader import load_parameters, make_beam_domain

# ---------------------------------------------------------------------------
# Detect run mode: JSON vs. legacy CLI
# ---------------------------------------------------------------------------

_JSON_FILE   = "parameters_maxfrequency.json"
_OUTPUT_CFG  = "output_config.json"

_out_cfg        = json.load(open(_OUTPUT_CFG, encoding="utf-8")) if os.path.exists(_OUTPUT_CFG) else {}
_output_folder  = _out_cfg.get("output", {}).get("folder", "outputs")
_sweep_enabled  = _out_cfg.get("sweep",  {}).get("enabled", False)


def _make_filter(nelx, nely, rmin, ft):
    return (SensitivityFilter if ft == 0 else DensityFilter)(nelx, nely, rmin)


def run_single(
    *,
    problem,
    nelx: int,
    nely: int,
    volfrac: float,
    rmin: float,
    penal: float,
    ft: int,
    max_iter: int,
    tol: float,
    move: float,
    output_dir: str,
    problem_label: str,
    case_label: str = "",
    objective: str = "min_compliance",
    length_x: float = 1.0,
    length_y: float = 1.0,
    style_cfg: dict | None = None,
) -> tuple[str, str]:
    """Run OC and SQP for one case, write outputs to output_dir, return (oc_summary, sqp_summary)."""
    results: dict[str, RunResult] = {}

    for method in ["MMA", "SQP"]:#["OC", "SQP"]:#resolve_methods("BOTH"):
        print(f"\n{'-'*60}")
        suffix = f"  [{case_label}]" if case_label else ""
        print(f"  Running {method}{suffix}")
        print(f"{'-'*60}")

        fe           = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
        filt         = _make_filter(nelx, nely, rmin, ft)
        raw_opt      = make_optimizer(method, move=move, penal=penal)
        profiler     = AlgorithmProfiler(method=method)
        profiled_fe  = ProfiledFESolver(fe, profiler)
        profiled_opt = ProfiledOptimizer(raw_opt, profiler)
        prof_cb      = ProfilingCallback(profiler, collect_density=True, collect_objective=True)
        logger       = ConsoleLogger(nelx=nelx, nely=nely, volfrac=volfrac, problem_type=objective)

        solver = TopOptSolver(
            problem   = problem,
            fe_solver = profiled_fe,
            filter    = filt,
            optimizer = profiled_opt,
            volfrac   = volfrac,
            callbacks = (logger, prof_cb),
            max_iter  = max_iter,
            tol       = tol,
        )

        if objective == "max_frequency":
            solver.max_frequency()
        if objective == "min_compliance":
            solver.min_compliance()

        print(f"\n{profiler.summary()}")

        results[method] = RunResult(
            method       = method,
            sim_config   = SimulationConfig(nelx=nelx, nely=nely),
            profiler     = profiler,
            cb           = prof_cb,
            hessian_used = method in HESSIAN_METHODS,
        )

    _title_prefix = TOPOPT_FREQ_LATEX if objective == "max_frequency" else TOPOPT_SIMP_LATEX
    _s = style_cfg or {}
    cfg = PlotterConfig(
        output_dir       = output_dir,
        dpi              = _s.get("dpi",              100),
        fig_size         = tuple(_s.get("fig_size",   [9.0, 5.5])),
        animation_fps    = _s.get("animation_fps",    5),
        animation_format = _s.get("animation_format", "gif"),
        density_cmap     = _s.get("density_cmap",     "gray"),
        title_prefix     = _title_prefix,
        show_grid        = _s.get("show_grid",         True),
        font_size        = _s.get("font_size",         11),
        title_font_size  = _s.get("title_font_size",   11),
        label_font_size  = _s.get("label_font_size",   11),
        tick_font_size   = _s.get("tick_font_size",    10),
        cbar_font_size   = _s.get("cbar_font_size",    9),
        font_family      = _s.get("font_family",       "Times New Roman"),
        problem_type     = objective,
    )
    plotter = TopOptPlotter(config=cfg, results=results, problem_label=problem_label)
    plotter.render_all(nelx=nelx, nely=nely, length_x=length_x, length_y=length_y)

    for m in results:
        if plotter.density_history.get(m):
            plotter.animate_density(
                method=m, nelx=nelx, nely=nely, save=True,
                length_x=length_x, length_y=length_y,
            )
    plotter.animate_cumulative_time(method=None, save=True)

    print(f"  Results saved to: {output_dir}/")

    return (
        results["OC"].profiler.summary()  if "OC"  in results else "",
        results["SQP"].profiler.summary() if "SQP" in results else "",
    )


def run_sweep() -> list[tuple[str, tuple | None]]:
    """Parameter sweep over move limits, mesh sizes, and initialisation types.

    Dimensions tested
    -----------------
    * meshes : 100x10, 300x30, 500x50, 1000x100  (all 10:1 aspect ratio)

    * init   : uniform (plain volfrac)

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
    _p      = json.load(open(_JSON_FILE, encoding="utf-8"))
    _opt    = _p["optimisation"]
    _swcfg  = _out_cfg.get("sweep", {})

    _volfrac    = float(_opt["volume_fraction"])
    _penal      = float(_opt["penalization"])
    _ft         = 1 if _opt.get("filter_type", "density") == "heaviside" else 0
    _max_iter   = int(_opt["max_iters"])
    _moves      = _swcfg.get("moves", [float(_opt.get("move_limit", _opt.get("move", 0.2)))])
    _tol_list   = _swcfg.get("tol",   [float(_opt["convergence_tol"])])
    _objective  = _opt.get("objective", "min_compliance")
    _SWEEP_MESHES = _swcfg.get("values")

    problem = make_beam_domain(_JSON_FILE)

    results: list[tuple[str, tuple | None]] = []
    total = (
        len(_SWEEP_MESHES) * len(_tol_list) * len(_moves)
    )  # mesh × tol × move × init
    done  = 0

    for nelx_s, nely_s in [ x.values() for x in _SWEEP_MESHES]:
        mesh_label = f"{nelx_s}x{nely_s}"
        # Scale filter radius with mesh density (≈ 4 % of nelx keeps the
        # relative stencil size constant across refinement levels).
        rmin_s = max(2.0, 0.04 * nelx_s)

        for tol_val in _tol_list:
            for move_val in _moves:
                done += 1
                case_name = f"{mesh_label}"
                print(f"\n{'='*60}")
                print(f"  SWEEP [{done}/{total}]: {case_name}")
                print(f"  mesh={nelx_s}x{nely_s}  rmin={rmin_s:.1f}"
                        f"  vf={_volfrac}  move={move_val}"
                        f"  tol={tol_val:.0e}")
                print(f"{'='*60}")

                try:
                    summaries = run_single(
                        problem       = problem,
                        nelx          = problem.nelx,
                        nely          = problem.nely,
                        volfrac       = _volfrac,
                        rmin          = rmin_s,
                        penal         = _penal,
                        ft            = _ft,
                        max_iter      = _max_iter,
                        tol           = tol_val,
                        move          = move_val,
                        output_dir    = os.path.join(_output_folder, case_name),
                        problem_label = f"{case_name}  vf={_volfrac}  p={_penal}  move={move_val}",
                        case_label    = case_name,
                        objective     = _objective,
                        length_x      = getattr(problem, "length_x", 1.0),
                        length_y      = getattr(problem, "length_y", 1.0),
                        style_cfg     = _out_cfg.get("style", {}),
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
    print(f"All sweep outputs written to: {_output_folder}/")
    return results


if _sweep_enabled:
    run_sweep()
else:
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
    tol       = float(opt_cfg["convergence_tol"])
    mode      = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTH"
    move      = float(opt_cfg.get("move_limit", opt_cfg.get("move", 0.2)))
    objective = opt_cfg.get("objective", "min_compliance")

    problem_label = (
        f"{params['meta']['name']}  {nelx}x{nely}"
        f"  vf={volfrac}  p={penal}"
    )

    print("=" * 60)
    print("  FreqTop — Topology Optimisation with Profiling & Plotting")
    print("=" * 60)
    print(f"  Mode    : JSON ({_JSON_FILE})")
    print(f"  Domain  : {nelx} x {nely}")
    print(f"  volfrac : {volfrac}  rmin : {rmin:.3g}  penal : {penal}")
    print(f"  Filter  : {'Density/Heaviside' if ft else 'Sensitivity'}")
    print(f"  Method  : {mode}  max_iter : {max_iter}")
    if domain.has_passive_elements():
        n_pass = len(domain.get_passive_elements())
        print(f"  Passive : {n_pass} elements ({n_pass/domain.nelxy:.1%} of domain)")
    print("=" * 60)

    run_single(
        problem       = domain,
        nelx          = nelx,
        nely          = nely,
        volfrac       = volfrac,
        rmin          = rmin,
        penal         = penal,
        ft            = ft,
        max_iter      = max_iter,
        tol           = tol,
        move          = move,
        output_dir    = _output_folder,
        problem_label = problem_label,
        objective     = objective,
        length_x      = domain.length_x,
        length_y      = domain.length_y,
        style_cfg     = _out_cfg.get("style", {}),
    )
