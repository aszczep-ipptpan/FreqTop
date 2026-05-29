# FreqTop

> Gradient-based structural topology optimisation — SIMP with OC, MMA, and SQP solvers.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

FreqTop solves two classes of structural **topology optimisation** on a 2-D structured quad mesh using the **SIMP** (Solid Isotropic Material with Penalisation) parametrisation:

| Objective | Problem formulation |
|-----------|-------------------|
| `min_compliance` | Minimise $\mathbf{u}^T\mathbf{K}(\boldsymbol{\rho})\,\mathbf{u}$ subject to volume constraint |
| `max_frequency` | Maximise fundamental eigenfrequency $\omega_1(\boldsymbol{\rho})$ subject to volume constraint |

Three update strategies are supported — **OC**, **MMA**, and **SQP** — benchmarked with an instrumented profiling and visualisation pipeline.

---

## Table of Contents

1. [Functionality](#functionality)
2. [Architecture](#architecture)
3. [Quick-start](#quick-start)
4. [Optimisers](#optimisers)
   - [OC — Optimality Criteria](#oc--optimality-criteria)
   - [MMA — Method of Moving Asymptotes](#mma--method-of-moving-asymptotes)
   - [SQP — Sequential Quadratic Programming](#sqp--sequential-quadratic-programming)
5. [JSON Input: parameters file](#json-input-parameters-file)
6. [JSON Output: output_config.json](#json-output-output_configjson)
7. [Filters](#filters)
8. [Public API](#public-api)

---

## Functionality

Each iteration of the optimisation loop:

1. **FE solve** — assembles global $\mathbf{K}(\boldsymbol{\rho})$ (and $\mathbf{M}(\boldsymbol{\rho})$ for frequency), solves the system, computes adjoint sensitivities.
2. **Filter** — spatially smooths the gradient field (density or sensitivity filter).
3. **Optimiser update** — delegates the density update to the chosen `Optimizer`.
4. **Callbacks** — logging, profiling, and density/objective history collection fire after every iteration.
5. **Convergence check** — stops when $\|\Delta\boldsymbol{\rho}\|_2 < \varepsilon$.

**SIMP material model** — for stiffness the standard penalised interpolation:

$$E_e(\rho_e) = E_{\min} + \rho_e^p \,(E_0 - E_{\min})$$

For mass (frequency problem) a linear interpolation is used with the **Tcherniak (2002) modification** to eliminate spurious localised eigenmodes in near-void regions:

$$M_e(\rho_e) = \begin{cases} \left[\rho_{\min} + \rho_e\,(\rho_{\max}-\rho_{\min})\right] M_e^* & \rho_e > 0.1 \\ c_0\,\rho_e^6\, \rho_{\max}\,M_e^* & \rho_e \le 0.1 \end{cases}$$

---

## Architecture

```text
FreqTop/
├── solver.py          MaterialProperties, Element, FEM, TopOptSolver
├── runner_types.py    RunResult, SimulationConfig
├── problems.py        Analytic benchmark problems (WeightedQuadratic, SIMPAnalogue)
├── fe/
│   ├── domain.py      MeshDomain, BeamDomain (JSON-driven BCs, passive flanges)
│   ├── fe_solver.py   FESolver  — K assembly, Ku=f, compliance sensitivities
│   ├── elements.py    QUAD4 element stiffness matrix lk()
│   └── modal_load.py  Modal / frequency analysis helpers
├── filters/
│   ├── density.py     DensityFilter    — convolve x → xPhys, filter dc and dv
│   ├── sensitivity.py SensitivityFilter — pass x as-is, smooth sensitivities only
│   └── _matrix.py     Weighted neighbourhood matrix H and row sums Hs
├── optimizers/
│   ├── oc.py          OCOptimizer
│   ├── mma.py         MMAOptimizer  (mmasub + subsolv, Svanberg 2007)
│   ├── sqp.py         SQPOptimizer  (L-BFGS SQP, Rojas-Labanda & Stolpe 2016)
│   └── registry.py    make_optimizer(), resolve_methods()
├── callbacks/
│   ├── logger.py              ConsoleLogger
│   └── profiling_callback.py  ProfilingCallback
├── profiling/
│   ├── profiler.py     AlgorithmProfiler — stage timing + memory
│   └── instrumented.py ProfiledFESolver, ProfiledOptimizer decorators
├── config/
│   └── loader.py   load_parameters(), make_beam_domain(), apply_overrides()
├── utils/
│   ├── base.py         TopOptSolver ABC
│   └── cantilever.py   CantileverProblem (legacy simple cantilever)
└── viz/
    ├── plotter.py          TopOptPlotter, PlotterConfig
    ├── plotter_sweep.py    Sweep result comparison
    ├── comparison_table.py TSV export
    └── hessian.py          Hessian approximation visualisation
```

---

## Quick-start

### JSON mode (recommended)

```bash
# Run all three optimisers, read settings from parameters_maxfrequency.json
python run.py

# Run only MMA, override max iterations
python run.py MMA 100
```

Outputs are written to the folder defined in `output_config.json → output.folder`.

### Programmatic API

```python
from FreqTop.config.loader import make_beam_domain
from FreqTop.fe.fe_solver   import FESolver
from FreqTop.filters.density import DensityFilter
from FreqTop.optimizers.mma  import MMAOptimizer
from FreqTop.callbacks.logger import ConsoleLogger
from FreqTop.solver import TopOptSolver

domain    = make_beam_domain("parameters_maxfrequency.json")
fe        = FESolver(domain, penal=3.0, Emin=1e-9, Emax=1.0)
filt      = DensityFilter(domain.nelx, domain.nely, rmin=3.6)
optimizer = MMAOptimizer()
solver    = TopOptSolver(
    domain, fe, filt, optimizer,
    volfrac=0.40,
    callbacks=(ConsoleLogger(domain.nelx, domain.nely, 0.40,
                             problem_type="max_frequency"),),
    max_iter=100,
    tol=1e-3,
)
xPhys = solver.max_frequency()
```

---

## Optimisers

All three implement the same interface:

```python
x_new, elapsed_s, mem_mb = optimizer.update(x, dc, dv, volfrac)
```

| Argument | Shape | Description |
|----------|-------|-------------|
| `x` | `(n_el,)` | Current design variables $\rho \in [0, 1]$ |
| `dc` | `(n_el,)` | Objective sensitivity $\partial f / \partial \rho_e$ |
| `dv` | `(n_el,)` | Volume sensitivity $\partial g / \partial \rho_e$ |
| `volfrac` | `float` | Target volume fraction |

---

### OC — Optimality Criteria

**File:** `optimizers/oc.py`  
**Reference:** Classical SIMP OC heuristic (Bendsøe & Sigmund 2003, Chapter 1)

#### Method

Applies the fixed-point update derived from KKT conditions:

$$\rho_e^{\,\text{new}} = \text{clip}\!\left(\rho_e \sqrt{\frac{-\,\partial f/\partial\rho_e}{\lambda\,\partial g/\partial\rho_e}},\; [\rho_e - m,\; \rho_e + m]\right)$$

The Lagrange multiplier $\lambda$ is found by bisection until the volume constraint is satisfied exactly. Cost: $O(n)$ per iteration.

#### Constructor

```python
OCOptimizer(move: float = 0.2, bisect_tol: float = 1e-3)
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `move` | `0.2` | **Move limit** — maximum allowable density change per element per iteration. Controls stability; too large causes oscillation, too small slows convergence. Typical range `[0.1, 0.3]`. |
| `bisect_tol` | `1e-3` | Convergence tolerance for the Lagrange-multiplier bisection. Bisection runs from $\lambda_1 = 0$ to $\lambda_2 = 10^9$. Tighter values give more accurate constraint satisfaction at marginal cost. |

#### Internal constants

| Constant | Value | Description |
|----------|-------|-------------|
| `EPS` | `1e-30` | Floor applied to `dv * lam` to prevent division by zero |
| `l1 / l2` | `0 / 1e9` | Bisection bracket for Lagrange multiplier |

---

### MMA — Method of Moving Asymptotes

**File:** `optimizers/mma.py`  
**Reference:** Svanberg K. (2007). *MMA and GCMMA — two methods for nonlinear optimization.* KTH Stockholm.

#### Method

Constructs a separable convex approximation of the objective and constraints around the current design. The approximation uses asymptotes $L_j$ (lower) and $U_j$ (upper) that adapt based on the sign of successive design changes:

- Oscillating element → asymptotes contract (factor `asydecr = 0.7`)
- Monotone element → asymptotes expand (factor `asyincr = 1.2`)

The resulting subproblem is solved by a primal–dual interior-point Newton method (`subsolv`).

#### Constructor

```python
MMAOptimizer(m: int = 1)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `m` | `1` | Number of inequality constraints. Default is 1 (volume constraint only). |

#### Internal state (persistent across iterations)

| Attribute | Description |
|-----------|-------------|
| `_iter` | Iteration counter — controls asymptote initialisation vs. adaptation |
| `_xold1` | Design variables from previous iteration |
| `_xold2` | Design variables from two iterations ago |
| `_low` | Lower asymptotes from previous iteration |
| `_upp` | Upper asymptotes from previous iteration |

#### Key constants inside `mmasub`

| Constant | Value | Role | Effect if changed |
|----------|-------|------|-------------------|
| `asyinit` | `0.5` | Initial asymptote distance: $L_j = x_j - 0.5(x_j^{\max} - x_j^{\min})$ | Larger → bigger first step (risk of void topology and spurious modes); smaller → slower early convergence |
| `asyincr` | `1.2` | Asymptote expansion factor when direction is monotone | Higher → faster convergence on smooth landscapes; may cause oscillation |
| `asydecr` | `0.7` | Asymptote contraction factor when direction oscillates | Lower → stronger damping of oscillations; slows recovery |
| `albefa` | `0.1` | Inner offset: $\alpha_j = L_j + 0.1(x_j - L_j)$ | Controls proximity to singularity of the approximation function |
| `move` | `1.0` | Explicit move limit on $[\alpha_j, \beta_j]$ — with asymptote clamping, effective move is `≤ 0.18` | At `1.0` has no additional effect; actual limit set by asymptotes |
| `raa0` | `1e-5` | Regularisation: adds $r_{aa0}/x_j^{\text{range}}$ to both $p_0, q_0$ | Ensures strict convexity; too large biases toward midpoint |
| `epsimin` | `1e-7` | KKT residual tolerance for inner Newton (`subsolv`) | Tighter → more accurate subproblem; more Newton steps |

#### Asymptote clamping (applied from iteration 3 onward)

```
low ∈ [xval − 0.2·(xmax−xmin),  xval − 0.01·(xmax−xmin)]
upp ∈ [xval + 0.01·(xmax−xmin),  xval + 0.2·(xmax−xmin)]
```

This limits the effective move to at most 18 % of the design range per iteration after the first two steps.

#### Volume constraint formulation

```python
fval = dot(dv, x) - volfrac * sum(dv)    # < 0 means feasible
c    = 1000.0                             # penalty for constraint slack variable y
d    = 0.0                               # quadratic slack penalty (unused)
a0   = 1.0                               # objective scaling
a    = 0.0                               # z-variable coefficient (unused)
```

> **Note:** `c = 1000` defines the upper bound on the constraint Lagrange multiplier in the subproblem. When objective sensitivities carry physical units ($|\partial\omega_1/\partial\rho| \gg 1$), normalise `dc` before calling `update` to ensure the volume constraint remains active.

---

### SQP — Sequential Quadratic Programming

**File:** `optimizers/sqp.py`  
**Reference:** Rojas-Labanda S. & Stolpe M. (2016). *An efficient second-order SQP method for structural topology optimization.* Struct Multidisc Optim **53**, 1315–1333.

#### Method

Implements Algorithm 1 (SQP+) with an L-BFGS Hessian:

1. **Hessian** $B_k$ — initialised from the SIMP diagonal $B_k^{(0)} = \tfrac{p-1}{x_e}\,|\partial f/\partial x_e|$; updated with L-BFGS curvature pairs from the Lagrangian gradient.
2. **IQP** — solves the inequality-constrained QP via bisection on the volume multiplier.
3. **EQP** — equality-constrained Newton correction on the active constraint set.
4. **Line search** — Armijo backtracking on an $l_1$ merit function, contraction factor `kappa`.
5. **Multiplier update** — augmented-Lagrangian update with scale protection to prevent saw-tooth oscillation.

#### Constructor

```python
SQPOptimizer(
    move:       float = 0.2,
    penal:      float = 3.0,
    stat_tol:   float = 1e-6,
    feas_tol:   float = 1e-8,
    comp_tol:   float = 1e-6,
    bisect_tol: float = 1e-4,
    sigma:      float = 1e-4,
    kappa:      float = 0.5,
    eps5:       float = 1e-6,
    eps4:       float = 1e-4,
    l2_init:    float = 1e9,
)
```

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `move` | `0.2` | **Move limit** — bounds per-element search direction: $d_e \in [\max(0, x_e - m) - x_e,\; \min(1, x_e + m) - x_e]$. Same role as in OC. |
| `penal` | `3.0` | SIMP exponent — used to construct the diagonal Hessian: $B_e = \tfrac{p-1}{x_e}\,|\partial f/\partial x_e|$. Must match the penalisation used in the FE loop. |
| `stat_tol` | `1e-6` | Stationarity KKT residual $\varepsilon_1$: $\|\nabla f + \lambda \nabla g + \xi - \eta\|_\infty$. |
| `feas_tol` | `1e-8` | Feasibility KKT residual $\varepsilon_2$: $\max(g^+, \text{bound violations})$. |
| `comp_tol` | `1e-6` | Complementarity KKT residual $\varepsilon_3$: $\max(|g\lambda|, \|(x-1)\xi\|_\infty, \|x\eta\|_\infty)$. |
| `bisect_tol` | `1e-4` | IQP bisection convergence on volume multiplier $\lambda$. |
| `sigma` | `1e-4` | Sufficient-decrease factor (Armijo condition): $\phi(\alpha d) \le \phi(0) - \sigma \alpha q_{\text{red}}$. |
| `kappa` | `0.5` | Line-search backtracking factor: $\alpha \leftarrow \kappa\alpha$ at each failed step. |
| `eps4` | `1e-4` | Active-set detection tolerance: element $e$ in active set if $|d_e - d_e^{\text{bound}}| < \varepsilon_4$. |
| `eps5` | `1e-6` | Merit function relaxation for active constraint detection. |
| `l2_init` | `1e9` | Upper bracket for bisection on $\lambda$. Must be large enough that the initial upper bound is non-binding. |

#### Persistent state

| Attribute | Description |
|-----------|-------------|
| `_lam` | Volume-constraint Lagrange multiplier (scalar, ≥ 0) |
| `_xi` | Upper-bound multipliers $(n,)$, ≥ 0 |
| `_eta` | Lower-bound multipliers $(n,)$, ≥ 0 |
| `_s_list` / `_y_list` | L-BFGS curvature pairs (max 15, from Lagrangian gradient) |
| `_lbfgs_xprev` / `_lbfgs_gprev` | Previous iterate and Lagrangian gradient |
| `_B0_diag` | SIMP diagonal Hessian (refreshed each call) |

Call `optimizer.reset()` before starting a new problem to clear all persistent state.

---

## JSON Input: parameters file

The solver reads problem settings from a single JSON file. By default this is `parameters_maxfrequency.json`. Pass a different path to `make_beam_domain(path)`.

### Full schema with defaults

```jsonc
{
  "meta": {
    "name": "string",                         // human-readable label (used in plot titles)
    "notes": "string",                        // free text
    "units": {
      "length":          "m",                 // informational only
      "force":           "N",
      "density":         "kg/m3",
      "elastic_modulus": "GPa"                // "GPa" | "MPa" | "Pa"
                                              // E value is auto-converted to Pa
    }
  },

  "domain": {
    "size": {
      "length": 9.0,                          // beam length [m]
      "height": 1.0                           // beam height [m]
    },
    "thickness": 1.0,                         // out-of-plane thickness [m]
    "mesh": {
      "nelx": 180,                            // elements in x direction
      "nely": 20,                             // elements in y direction
      "FEM_massmatrix_gamma": 1.0             // consistent mass matrix scaling γ
    },
    "passive_regions": {
      "flange": {
        "is_bottom_passive_flag": false,      // lock bottom flange at ρ=1
        "bottom_relative_height": 0.1,        // flange height as fraction of nely
        "is_top_passive_flag":    false,      // lock top flange at ρ=1
        "top_relative_height":    0.1
      }
    }
  },

  "materials": {
    "base": {
      "E":   205.0,                           // Young's modulus [GPa if units.elastic_modulus="GPa"]
      "nu":  0.3,                             // Poisson's ratio
      "rho": 7850.0                           // mass density [kg/m³]
    }
  },

  "bc": {
    "supports": [
      // each entry is one support definition
      {
        "type":     "clamped",                // "clamped" | "middle" | "node"
        "location": "left",                   // "left" | "right" | "bottom" | "top"
        "constraint": "xy"                    // optional: "x" | "y" | "xy" (default "xy")
      },
      {
        "type": "node",
        "rx":   0.0,                          // relative x position ∈ [0,1]
        "ry":   0.0                           // relative y position ∈ [0,1]
      }
    ],
    "loads": {
      "is_selfweight":  true,                 // add gravity body force
      "gravity":        10.0,                 // gravitational acceleration [m/s²]
      "concentrated_forces": [
        {
          "edge":                       "top",  // edge to place force on
          "relative_horizontal_position": 0.5,  // position along edge ∈ [0,1]
          "direction":                  "y",    // "x" | "y"
          "value":                      0.0     // force magnitude [N]
        }
      ]
    }
  },

  "optimisation": {
    "objective":        "max_frequency",      // "max_frequency" | "min_compliance"
    "volume_fraction":  0.40,                 // target V* / V₀ ∈ (0, 1)
    "method":           "SIMP",               // material model (always SIMP)
    "penalization":     3.0,                  // SIMP exponent p (stiffness)
    "filter_radius":    0.2,                  // r_min as fraction of domain length
    "move_limit":       0.2,                  // per-element move limit (OC and SQP)
    "max_iters":        50,                   // hard iteration ceiling
    "convergence_tol":  1e-9,                 // stop when ‖Δρ‖₂ < tol
    "filter_type":      "heaviside",          // "sensitivity" | "density" | "heaviside"
    "heaviside_beta":   20,                   // β projection sharpness (heaviside only)
    "heaviside_eta":    0.5,                  // η threshold (heaviside only)
    "E_min":            0.1,                  // relative stiffness floor (void elements)
    "rho_min":          1e-4,                 // relative mass floor (void elements)
    "l2":               1e9                   // upper bisection bracket for SQP / OC
  }
}
```

### Support type reference

| `type` | `location` / extra | Constraint applied |
|--------|--------------------|--------------------|
| `"clamped"` | `"left"` / `"right"` / `"bottom"` / `"top"` | All nodes on that edge, both DOFs |
| `"middle"` | `"left"` / `"right"` / `"bottom"` / `"top"` | Single midpoint node of that edge |
| `"node"` | `"rx"`, `"ry"` (relative coords) | Nearest grid node |

### Filter type reference

| `filter_type` | Python class | Description |
|---------------|-------------|-------------|
| `"sensitivity"` | `SensitivityFilter` | Smooths sensitivities only; physical density = design variable |
| `"density"` | `DensityFilter` | Smooths design variables before physics (removes checkerboard) |
| `"heaviside"` | `DensityFilter` | Same as density but `heaviside_beta` / `heaviside_eta` control projection sharpness towards 0–1 |

---

## JSON Output: output_config.json

Controls all output paths, plots, CSV export, and figure styling. Read at startup by `run.py`.

```jsonc
{
  "run": {
    "method": "BOTH",          // default method if not given on CLI: "OC"|"MMA"|"SQP"|"BOTH"
    "solver": "topopt"         // reserved; always "topopt"
  },

  "output": {
    "folder": "results_beam_clamped_test"   // output directory (created if absent)
  },

  "sweep": {
    "enabled": false,          // true = parameter sweep mode
    "values": [                // list of dot-notation overrides applied to parameters.json
      {"domain.mesh.nelx": 100, "domain.mesh.nely": 10},
      {"domain.mesh.nelx": 200, "domain.mesh.nely": 20}
    ],
    "comparison_table": {
      "enabled": true,
      "filename": "comparison_table.tsv"   // TSV summary across all sweep cases
    }
  },

  "plots": {
    // Each entry: {"enabled": true|false}
    "density":                    {"enabled": false},  // density field snapshot (final iter)
    "convergence":                {"enabled": true},   // objective vs iteration
    "change_norm":                {"enabled": true},   // ‖Δρ‖ vs iteration (log scale)
    "volume_fraction":            {"enabled": false},  // V/V₀ vs iteration
    "stage_breakdown":            {"enabled": false},  // stacked-bar: time per stage
    "change_vs_cumtime":          {"enabled": false},  // |Δf| vs cumulative time
    "cumulative_time":            {"enabled": false},  // wall-clock time vs iteration
    "cumulative_memory":          {"enabled": false},  // memory (MB) vs iteration
    "obj_vs_cumtime":             {"enabled": false},  // objective vs cumulative time
    "comparison":                 {"enabled": false},  // side-by-side method comparison
    "density_animation":          {"enabled": true},   // animated GIF: density evolution
    "density_animation_singleframes": {"enabled": true}, // PNG per iteration (for video editing)
    "cumtime_animation":          {"enabled": true}    // animated GIF: cumulative time
  },

  "csv": {
    "enabled": true,
    // Tab-separated output, one file per method: oc_data.csv, mma_data.csv, sqp_data.csv
    "columns": {
      "method":               true,   // optimizer name
      "iteration":            true,   // iteration index
      "time_s":               true,   // wall-clock time this iteration [s]
      "cumulative_time_s":    true,   // cumulative wall-clock time [s]
      "memory_mb":            true,   // memory delta this iteration [MB]
      "cumulative_memory_mb": true,   // cumulative memory [MB]
      "objective_f":          true,   // objective value (compliance or ω₁)
      "change":               true    // ‖Δρ‖ convergence norm
    }
  },

  "style": {
    "dpi":              150,           // raster output resolution
    "fig_size":         [9.0, 5.5],   // figure size [inches]
    "font_family":      "Times New Roman",
    "font_size":        18,
    "title_font_size":  18,
    "show_grid":        true,
    "density_cmap":     "gray",        // matplotlib colormap for density plots
    "animation_fps":    5,             // frames per second for GIF animations
    "animation_format": "gif"          // "gif" | "svg"
  }
}
```

---

## Filters

| Class | `ft` | Applied to | Effect |
|-------|------|-----------|--------|
| `SensitivityFilter` | `0` | Sensitivities `dc`, not `x` | Smooths gradient; physical density = design variable |
| `DensityFilter` | `1` | Design variables `x` → `xPhys` | Removes checkerboard; smooths topology |

Both use the same weighted neighbourhood matrix built by `_matrix.build_filter_matrix(nelx, nely, rmin)`:

$$\tilde{x}_e = \frac{\sum_{f \in N_e} H_{ef}\, x_f}{\sum_{f \in N_e} H_{ef}}, \qquad H_{ef} = \max\!\left(0,\; r_{\min} - \|e - f\|\right)$$

Filter radius `rmin` is specified in `parameters.json` as a fraction of the domain length and converted to element units by `rmin = filter_radius * domain_length`.

---

## Public API

### MODEL — Problem & FE

| Class | Module | Description |
|-------|--------|-------------|
| `MeshDomain` | `fe/domain.py` | Structured 2-D quad mesh: node/element indexing, DOF map |
| `BeamDomain` | `fe/domain.py` | Extends MeshDomain with JSON-driven BCs, loads, passive flanges |
| `CantileverProblem` | `utils/cantilever.py` | Legacy fixed-left cantilever with tip load |
| `FESolver` | `fe/fe_solver.py` | Assemble K, solve Ku=f, compliance sensitivities |
| `MaterialProperties` | `solver.py` | SIMP parameters + physical E, ρ (reads from JSON via `read_json_params`) |
| `Element` | `solver.py` | QUAD4 element stiffness and consistent mass matrices |
| `FEM` | `solver.py` | K/M assembly and generalised EVP for frequency optimisation |

### MODEL — Filters

| Class | Module | Description |
|-------|--------|-------------|
| `DensityFilter` | `filters/density.py` | Convolve x → xPhys; filter dc and dv |
| `SensitivityFilter` | `filters/sensitivity.py` | Pass x as-is; smooth sensitivities only |

### MODEL — Optimisers

| Class | Module | Description |
|-------|--------|-------------|
| `OCOptimizer` | `optimizers/oc.py` | Heuristic fixed-point update; O(n) bisection multiplier |
| `MMAOptimizer` | `optimizers/mma.py` | Moving asymptotes; separable convex subproblem (Svanberg 2007) |
| `SQPOptimizer` | `optimizers/sqp.py` | L-BFGS SQP; IQP + EQP correction + merit line search |

### CONTROLLER — Orchestration

| Class | Module | Description |
|-------|--------|-------------|
| `TopOptSolver` | `solver.py` | Main loop: `min_compliance()` and `max_frequency()` |

### VIEW — Visualisation

| Class | Module | Description |
|-------|--------|-------------|
| `TopOptPlotter` | `viz/plotter.py` | Density fields, convergence curves, time profiles, GIF animations |
| `PlotterConfig` | `viz/plotter.py` | Dataclass: output dir, DPI, colours, colourmap, font sizes, problem type |

### OBSERVER — Callbacks & Profiling

| Class | Module | Description |
|-------|--------|-------------|
| `ConsoleLogger` | `callbacks/logger.py` | Per-iteration objective / volume / change table; `problem_type` selects label |
| `ProfilingCallback` | `callbacks/profiling_callback.py` | Collects objective and density history per iteration |
| `AlgorithmProfiler` | `profiling/profiler.py` | Context-manager stage timing and memory measurement |
| `ProfiledFESolver` | `profiling/instrumented.py` | Wraps `FESolver` with per-stage timing |
| `ProfiledOptimizer` | `profiling/instrumented.py` | Wraps any `Optimizer` with per-stage timing |

---

## Dependencies

```
numpy >= 1.24
scipy >= 1.10
matplotlib >= 3.7
Pillow            # GIF animation export
```

---

## References

- Bendsøe M.P. & Sigmund O. (2003). *Topology Optimization: Theory, Methods and Applications.* Springer.
- Du J. & Olhoff N. (2007). Topological design of freely vibrating continuum structures for maximum values of simple and multiple eigenfrequencies and frequency gaps. *Struct Multidisc Optim* **34**, 91–110.
- Rojas-Labanda S. & Stolpe M. (2016). An efficient second-order SQP method for structural topology optimization. *Struct Multidisc Optim* **53**, 1315–1333.
- Svanberg K. (2007). *MMA and GCMMA — two methods for nonlinear optimization.* KTH Stockholm.
- Tcherniak D. (2002). Topology optimization of resonating structures using SIMP method. *Int J Numer Methods Eng* **54**, 1605–1622.
