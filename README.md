# FreqTop

> Gradient-based structural topology optimisation — SIMP compliance minimisation with OC and SQP solvers.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

FreqTop solves minimum-compliance **topology optimisation** on a 2-D structured regular mesh using the **SIMP** (Solid Isotropic Material with Penalisation) parametrisation. It benchmarks two update strategies — **Optimality Criteria (OC)** and **Sequential Quadratic Programming (SQP)** — and wraps both with an instrumented profiling and visualisation pipeline. The solver targets computational mechanics research where algorithm convergence behaviour and per-stage wall-clock timing are first-class outputs, not afterthoughts.

---

## Functionality

**FreqTop** minimises structural compliance $c = \mathbf{u}^T \mathbf{K} \mathbf{u}$ subject to a volume fraction constraint over a structured cantilever domain. Each iteration assembles the global stiffness system from element densities, solves it, computes adjoint sensitivities, applies a spatial **filter** to smooth the gradient field, then delegates the density update to an `Optimizer`. A **callback chain** fires after every iteration, separating logging, profiling, and checkpointing concerns from the solver core. Results — convergence histories, density animations, per-stage timing breakdowns — are rendered by a dedicated visualisation layer.

## Numerical Formulation

The **OC** method applies a heuristic fixed-point update derived from the KKT conditions, finding the volume-constraint Lagrange multiplier by bisection at O(n) cost per iteration. The **SQP** method (Rojas-Labanda & Stolpe 2016, Algorithm 1) builds a diagonal Hessian approximation from the SIMP sensitivity expression, solves an inequality QP analytically, corrects with an equality QP on the active set, and enforces descent via a merit-function line search — second-order convergence at the cost of persistent multiplier state across iterations. Both methods support a **density filter** (convolution applied to design variables before FE evaluation) and a **sensitivity filter** (convolution applied to gradients only).

## Architecture

```
FreqTop/
├── solver.py       — main optimisation loop            (Controller)
├── utils/          — problem ABCs, CantileverProblem   (Model: domain & BCs)
├── fe/             — FE assembly, Ku=f, sensitivities  (Model: physics)
├── filters/        — density and sensitivity filters   (Model: regularisation)
├── optimizers/     — OC and SQP update strategies      (Model: optimisation)
├── callbacks/      — per-iteration observer hooks      (Observer)
├── profiling/      — time/memory instrumentation       (Decorator)
├── viz/            — plots, animations, comparisons    (View)
└── config/         — JSON parameter loading
```

---

## Quick-start

```python
from FreqTop.utils.cantilever import CantileverProblem
from FreqTop.fe.fe_solver import FESolver
from FreqTop.filters.density import DensityFilter
from FreqTop.optimizers.oc import OCOptimizer
from FreqTop.callbacks.logger import ConsoleLogger
from FreqTop.solver import TopOptSolver

problem   = CantileverProblem(nelx=180, nely=60)      # mesh: width × height in elements
fe        = FESolver(problem, penal=3.0)               # penal: SIMP exponent; Emin/Emax default to [1e-9, 1.0]
filt      = DensityFilter(nelx=180, nely=60, rmin=5.4) # rmin: filter radius in element lengths
optimizer = OCOptimizer()
solver    = TopOptSolver(
    problem, fe, filt, optimizer,
    volfrac=0.4,                    # target solid volume fraction ∈ [0, 1]
    callbacks=(ConsoleLogger(),),
    max_iter=200,
    tol=0.01,                       # convergence: inf-norm change in design variables
)
xPhys = solver.run()                # returns physical density field, shape (nelx*nely,)
```

Or run the bundled entry point from the command line:

```bash
python run.py [nelx] [nely] [volfrac] [rmin] [penal] [ft] [method]
# ft:     0 = sensitivity filter, 1 = density filter
# method: OC | SQP | BOTH
```

Results are written to `./outputs/`.

!!! note
    Installation dependencies are not yet documented in this repo. The library requires at minimum `numpy`, `scipy`, and `matplotlib`. Confirm the full dependency surface from `run.py` imports before setting up an environment.

---

## Public API

### info "MODEL — Problem & FE"
    | Class | Module | Description |
    |---|---|---|
    | `CantileverProblem` | `utils/cantilever.py` | Fixed-left cantilever BCs and point-load vector |
    | `FESolver` | `fe/fe_solver.py` | Assemble K, solve Ku=f, adjoint compliance sensitivities |

### info "MODEL — Filters"
    | Class | Module | Description |
    |---|---|---|
    | `DensityFilter` | `filters/density.py` | Convolve x → xPhys; filter dc and dv |
    | `SensitivityFilter` | `filters/sensitivity.py` | Pass x as-is; smooth sensitivities only |

### info "MODEL — Optimizers"
    | Class | Module | Description |
    |---|---|---|
    | `OCOptimizer` | `optimizers/oc.py` | Heuristic fixed-point update; bisection Lagrange multiplier |
    | `SQPOptimizer` | `optimizers/sqp.py` | Diagonal-Hessian SQP; IQP + EQP correction + line search |

### example "CONTROLLER — Orchestration"
    | Class | Module | Description |
    |---|---|---|
    | `TopOptSolver` | `solver.py` | Main loop: FE solve → sensitivities → filter → update → callbacks |

### success "VIEW — Visualisation"
    | Class | Module | Description |
    |---|---|---|
    | `TopOptPlotter` | `viz/plotter.py` | Render density fields, convergence curves, time profiles, GIF animations |
    | `PlotterConfig` | `viz/plotter.py` | Dataclass: output dir, DPI, colours, colourmap, font sizes |

### warning "OBSERVER — Callbacks & Profiling"
    Communication layer between MODEL, CONTROLLER, and VIEW — fires after every iteration without coupling to any single layer.
    | Class | Module | Description |
    |---|---|---|
    | `ConsoleLogger` | `callbacks/logger.py` | Per-iteration compliance / volume / change table |
    | `ProfilingCallback` | `callbacks/profiling_callback.py` | Commits profiler state; collects objective and density history |
    | `AlgorithmProfiler` | `profiling/profiler.py` | Context-manager stage timing and memory measurement |
    | `ProfiledFESolver` | `profiling/instrumented.py` | Decorator: wraps `FESolver` with stage timing |
    | `ProfiledOptimizer` | `profiling/instrumented.py` | Decorator: wraps any `Optimizer` with stage timing |
