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
from FreqTop.optimizers.registry import make_optimizer, resolve_methods, HESSIAN_METHODS
from FreqTop.runner_types        import RunResult, SimulationConfig
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
_sweep_mode = True

_SWEEP_MESHES = []

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

    sweep_results: dict[str, RunResult] = {}

    for method in resolve_methods("BOTH"):
        print(f"\n{'-'*60}")
        print(f"  Running {method}  [{case_name}]")
        print(f"{'-'*60}")

        problem = CantileverProblem(nelx=nelx, nely=nely)
        fe      = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
        filt    = _make_filter(nelx, nely, rmin, ft)

        raw_opt = make_optimizer(method, move=move, penal=penal)

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

        sweep_results[method] = RunResult(
            method       = method,
            sim_config   = SimulationConfig(nelx=nelx, nely=nely),
            profiler     = profiler,
            cb           = prof_cb,
            hessian_used = method in HESSIAN_METHODS,
        )

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

    plotter = TopOptPlotter(
        config        = sweep_cfg,
        results       = sweep_results,
        problem_label = problem_label,
    )

    plotter.render_all(nelx=nelx, nely=nely)

    for m in sweep_results:
        if plotter.density_history.get(m):
            plotter.animate_density(method=m, nelx=nelx, nely=nely, save=True)

    plotter.animate_cumulative_time(method=None, save=True)

    print(f"  Results saved to: {output_dir}/")
    return (
        sweep_results["OC"].profiler.summary()  if "OC"  in sweep_results else "",
        sweep_results["SQP"].profiler.summary() if "SQP" in sweep_results else "",
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
    import json as _json
    _p      = _json.load(open(_JSON_FILE, encoding="utf-8"))
    _opt    = _p["optimisation"]
    _swcfg  = (_json.load(open("output_config.json", encoding="utf-8"))
               if os.path.exists("output_config.json") else {}).get("sweep", {})

    _volfrac  = float(_opt["volume_fraction"])
    _penal    = float(_opt["penalization"])
    _ft       = 1 if _opt.get("filter_type", "density") == "heaviside" else 0
    _max_iter = int(_opt["max_iters"])
    _moves    = _swcfg.get("moves", [float(_opt.get("move", 0.8))])
    _tol_list = _swcfg.get("tol",   [float(_opt["convergence_tol"])])
    _SWEEP_MESHES = _swcfg.get("values")

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
                case_name = (
                    f"{mesh_label}_tol{tol_val:.0e}"
                    f"_move{move_val:.1f}"
                )
                print(f"\n{'='*60}")
                print(f"  SWEEP [{done}/{total}]: {case_name}")
                print(f"  mesh={nelx_s}x{nely_s}  rmin={rmin_s:.1f}"
                        f"  vf={_volfrac}  move={move_val}"
                        f"  tol={tol_val:.0e}")
                print(f"{'='*60}")

                try:
                    summaries = _run_sweep_case(
                        case_name = case_name,
                        nelx      = nelx_s,
                        nely      = nely_s,
                        rmin      = rmin_s,
                        volfrac   = _volfrac,
                        penal     = _penal,
                        ft        = _ft,
                        max_iter  = _max_iter,
                        tol       = tol_val,
                        move      = move_val,
                        init_type = "uniform"
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


def _run_one_point(
    params: dict,
    mode: str,
    output_dir: str,
    max_iter_override: "int | None" = None,
) -> "dict[str, RunResult]":
    """Run all methods for one JSON-parameter configuration; render to output_dir.

    This is the single entry point for both the single-run and sweep paths.
    All values are derived from *params* so no module-level globals are needed.
    """
    from FreqTop.config.loader import make_beam_domain_from_dict

    opt_cfg   = params["optimisation"]
    _domain   = make_beam_domain_from_dict(params)
    _nelx     = _domain.nelx
    _nely     = _domain.nely
    _volfrac  = float(opt_cfg["volume_fraction"])
    _rmin     = float(opt_cfg["filter_radius"]) * float(params["domain"]["size"]["length"])
    _penal    = float(opt_cfg["penalization"])
    _ft       = 1 if opt_cfg.get("filter_type", "density") == "heaviside" else 0
    _max_iter = max_iter_override if max_iter_override is not None else int(opt_cfg["max_iters"])
    _tol      = float(opt_cfg["convergence_tol"])
    _label    = (f"{params['meta']['name']}  {_nelx}x{_nely}"
                 f"  vf={_volfrac}  p={_penal}")

    print(f"\n  Domain  : {_nelx} x {_nely}  |  vf={_volfrac}  penal={_penal}")
    if _domain.has_passive_elements():
        _np = len(_domain.get_passive_elements())
        print(f"  Passive : {_np} elements ({_np / _domain.nelxy:.1%})")

    _cfg = PlotterConfig(
        output_dir       = output_dir,
        dpi              = 120,
        animation_fps    = 6,
        animation_format = "gif",
        title_prefix     = TOPOPT_SIMP_LATEX,
        show_grid        = True,
        font_size        = 11,
    )

    def _single_method(method: str) -> tuple:
        print(f"\n{'-'*60}")
        print(f"  Running {method} optimiser  (max_iter={_max_iter})")
        print(f"{'-'*60}")
        _fe   = FESolver(_domain, penal=_penal, Emin=1e-9, Emax=1.0)
        _filt = _make_filter(_nelx, _nely, _rmin, _ft)
        _opt  = make_optimizer(method, move=0.8, penal=_penal)
        _prof = AlgorithmProfiler(method=method)
        _pfe  = ProfiledFESolver(_fe, _prof)
        _popt = ProfiledOptimizer(_opt, _prof)
        _cb   = ProfilingCallback(_prof, collect_density=True, collect_objective=True)
        _log  = ConsoleLogger(nelx=_nelx, nely=_nely, volfrac=_volfrac)
        TopOptSolver(
            problem   = _domain,
            fe_solver = _pfe,
            filter    = _filt,
            optimizer = _popt,
            volfrac   = _volfrac,
            callbacks = (_log, _cb),
            max_iter  = _max_iter,
            tol       = _tol,
        ).run()
        print(f"\n{_prof.summary()}")
        return _prof, _cb

    _results: dict[str, RunResult] = {}
    for _m in resolve_methods(mode):
        _p, _c = _single_method(_m)
        _results[_m] = RunResult(
            method       = _m,
            sim_config   = SimulationConfig(nelx=_nelx, nely=_nely),
            profiler     = _p,
            cb           = _c,
            hessian_used = _m in HESSIAN_METHODS,
        )

    _plotter = TopOptPlotter(config=_cfg, results=_results, problem_label=_label)
    print("\nRendering static plots ...")
    _plotter.render_all(nelx=_nelx, nely=_nely)
    print("Rendering animations ...")
    for _m in _results:
        if _plotter.density_history.get(_m):
            print(f"  density animation [{_m}] ...")
            _plotter.animate_density(method=_m, nelx=_nelx, nely=_nely, save=True)
    print("  cumulative-time animation [both] ...")
    _plotter.animate_cumulative_time(method=None, save=True)
    print(f"\nAll outputs written to: {output_dir}/")
    return _results


if _sweep_mode:
    # ── Old hardcoded sweep mode (kept for backward compat) ────────────────
    print("=" * 60)
    print("  FreqTop — Parameter Sweep")
    print(f"  Meshes : {[lbl for *_, lbl in _SWEEP_MESHES]}")
    print(f"  Moves  : (from output_config.json sweep.moves)")
    print(f"  Init   : uniform + checkerboard (10x10 blocks)")
    print(f"  Method : OC + SQP  (both per case, BOTH-mode output)")
    print(f"  Output : outputs_sweep/<mesh>_move<val>_<init>/")
    print("=" * 60)
    run_sweep()

elif _use_json:
    # ── JSON mode: single run or config-driven sweep ───────────────────────
    import json as _json

    _out_cfg  = _json.load(open("output_config.json", encoding="utf-8")) \
                if os.path.exists("output_config.json") else {}
    _base_out = _out_cfg.get("output", {}).get("folder", "outputs")
    _max_cli  = int(sys.argv[2]) if len(sys.argv) > 2 else None
    _sweep_sec = _out_cfg.get("sweep", {"enabled": False})

    if _sweep_sec.get("enabled", False):
        from FreqTop.config.loader import apply_overrides
        from FreqTop.viz.comparison_table import ComparisonRow, ComparisonTable

        _vals    = _sweep_sec.get("values", [])
        _ct_cfg  = _sweep_sec.get("comparison_table", {})
        _ct_on   = _ct_cfg.get("enabled", True)
        _ct_file = _ct_cfg.get("filename", "comparison_table.tsv")

        print(f"  Sweep   : {len(_vals)} points")
        print(f"  Variable: {list(_vals[0]) if _vals else '?'}")
        print("=" * 60)

        _table = ComparisonTable() if _ct_on else None

        for _i, _ov in enumerate(_vals, 1):
            print(f"\n{'='*60}")
            print(f"  SWEEP [{_i}/{len(_vals)}]: {_ov}")
            print(f"{'='*60}")
            _lbl        = "_".join(f"{k.split('.')[-1]}{v}" for k, v in _ov.items())
            _out_i      = os.path.join(_base_out, _lbl)
            _pt_results = _run_one_point(apply_overrides(params, _ov), mode, _out_i, _max_cli)
            if _table is not None:
                _table.add(ComparisonRow.from_results(_pt_results))

        if _table is not None:
            _table.to_csv(os.path.join(_base_out, _ct_file))
        print("\nSweep complete.")
    else:
        _run_one_point(params, mode, _base_out, _max_cli)

else:
    # ── Legacy CLI mode — closure-based run (no sweep support) ────────────
    cfg = PlotterConfig(
        output_dir       = "outputs",
        dpi              = 120,
        animation_fps    = 6,
        animation_format = "gif",
        title_prefix     = TOPOPT_SIMP_LATEX,
        show_grid        = True,
        font_size        = 11,
    )

    INIT_TYPE = "uniform"

    def run_method(method: str, init_type: str = INIT_TYPE) -> tuple:
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

        raw_opt = make_optimizer(method, move=0.8, penal=penal)

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

    results: dict[str, RunResult] = {}

    for _method in resolve_methods(mode):
        _profiler, _cb = run_method(_method)
        results[_method] = RunResult(
            method       = _method,
            sim_config   = SimulationConfig(nelx=nelx, nely=nely),
            profiler     = _profiler,
            cb           = _cb,
            hessian_used = _method in HESSIAN_METHODS,
        )

    plotter = TopOptPlotter(
        config        = cfg,
        results       = results,
        problem_label = problem_label,
    )

    print("\nRendering static plots ...")
    plotter.render_all(nelx=nelx, nely=nely)

    print("Rendering animations ...")
    for m in results:
        if plotter.density_history.get(m):
            print(f"  density animation [{m}] ...")
            plotter.animate_density(method=m, nelx=nelx, nely=nely, save=True)

    print("  cumulative-time animation [both] ...")
    plotter.animate_cumulative_time(method=None, save=True)

    print(f"\nAll outputs written to:  {cfg.output_dir}/")
    print("Done.")
