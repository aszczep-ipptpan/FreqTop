"""
run.py — command-line entry point for FreqTop with profiling and plotting.

Usage
-----
    python run.py [nelx] [nely] [volfrac] [rmin] [penal] [ft] [method]

Arguments (all optional, positional)
-------------------------------------
nelx    int    180   number of elements in x
nely    int     60   number of elements in y
volfrac float  0.4   target volume fraction
rmin    float  5.4   filter radius
penal   float  3.0   SIMP penalisation exponent
ft      int      1   filter type: 0=sensitivity, 1=density
method  str   both   which optimiser to run: "OC", "SQP", or "both"

Examples
--------
    python run.py                          # defaults, run both OC and SQP
    python run.py 60 20 0.4 3.0 3.0 1 OC  # run only OC on a small mesh
    python run.py 60 20 0.4 3.0 3.0 1 SQP # run only SQP

Outputs (written to ./outputs/)
--------------------------------
    density_OC_iter*.png / density_SQP_iter*.png
    convergence_OC.png / convergence_SQP.png
    cumtime_OC.png / cumtime_SQP.png / cumtime_both.png
    cummem_OC.png  / cummem_SQP.png  / cummem_both.png
    stage_breakdown_OC.png / stage_breakdown_SQP.png
    comparison_oc_vs_sqp.png
    density_animation_OC.gif / density_animation_SQP.gif
    cumtime_animation_both.gif
"""

import sys
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs

from FreqTop.utils.cantilever import CantileverProblem
from FreqTop.solver            import TopOptSolver
from FreqTop.fe.fe_solver      import FESolver
from FreqTop.filters.sensitivity import SensitivityFilter
from FreqTop.filters.density     import DensityFilter
from FreqTop.optimizers.oc       import OCOptimizer
from FreqTop.optimizers.sqp      import SQPOptimizer
from FreqTop.callbacks.logger    import ConsoleLogger

from FreqTop.profiling import AlgorithmProfiler
from FreqTop.profiling.instrumented import ProfiledFESolver, ProfiledOptimizer
from FreqTop.callbacks.profiling_callback import ProfilingCallback
from FreqTop.viz import TopOptPlotter, PlotterConfig


def _arg(idx, default, cast=float):
    return cast(sys.argv[idx]) if len(sys.argv) > idx else default

nelx    = _arg(1, 180, int)
nely    = _arg(2,  60, int)
volfrac = _arg(3, 0.4, float)
rmin    = _arg(4, 5.4, float)
penal   = _arg(5, 3.0, float)
ft      = _arg(6, 1,   int)
mode    = sys.argv[7].upper() if len(sys.argv) > 7 else "BOTH"

print("=" * 60)
print("  FreqTop — Topology Optimisation with Profiling & Plotting")
print("=" * 60)
print(f"  Domain  : {nelx} x {nely}")
print(f"  volfrac : {volfrac}  rmin : {rmin}  penal : {penal}")
print(f"  Filter  : {'Density' if ft else 'Sensitivity'}")
print(f"  Method  : {mode}")
print("=" * 60)

problem_label = f"Cantilever {nelx}x{nely}  vf={volfrac}  p={penal}"

cfg = PlotterConfig(
    output_dir       = "outputs",
    dpi              = 120,
    animation_fps    = 6,
    animation_format = "gif",
    title_prefix     = "FreqTop",
    show_grid        = True,
    font_size        = 11,
)


def _make_filter(nelx, nely, rmin, ft):
    return (SensitivityFilter if ft == 0 else DensityFilter)(nelx, nely, rmin)


def run_method(method: str, max_iter: int = 20) -> tuple:
    """Construct solver, wrap with profiler, run, return (profiler, profiling_cb)."""
    print(f"\n{'─'*60}")
    print(f"  Running {method} optimiser  (max_iter={max_iter})")
    print(f"{'─'*60}")

    problem = CantileverProblem(nelx=nelx, nely=nely)
    fe      = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
    filt    = _make_filter(nelx, nely, rmin, ft)

    raw_optimizer = OCOptimizer(move=0.2) if method == "OC" else SQPOptimizer(move=0.2, penal=penal)

    profiler     = AlgorithmProfiler(method=method)
    profiled_fe  = ProfiledFESolver(fe, profiler)
    profiled_opt = ProfiledOptimizer(raw_optimizer, profiler)

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
        tol       = 0.01,
    )
    solver.run()

    print(f"\n{profiler.summary()}")
    return profiler, prof_cb


oc_profiler = sqp_profiler = None
oc_cb       = sqp_cb       = None

if mode in ("OC", "BOTH"):
    oc_profiler, oc_cb = run_method("OC")

if mode in ("SQP", "BOTH"):
    sqp_profiler, sqp_cb = run_method("SQP")


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
plotter.render_all(nelx=nelx, nely=nely, oc_obj=oc_obj, sqp_obj=sqp_obj)

print("Rendering animations ...")
for method in [m for m in ("OC", "SQP") if density_history.get(m)]:
    print(f"  density animation [{method}] ...")
    plotter.animate_density(method=method, nelx=nelx, nely=nely, save=True)

print("  cumulative-time animation [both] ...")
plotter.animate_cumulative_time(method=None, save=True)

print(f"\nAll outputs written to:  {cfg.output_dir}/")
print("Done.")
