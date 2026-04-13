"""
run_topopt.py — command-line entry point for pytopopt.

Usage (matches the original 165-line script):
    python run_topopt.py [nelx] [nely] [volfrac] [rmin] [penal] [ft]

Defaults:
    nelx=180  nely=60  volfrac=0.4  rmin=5.4  penal=3.0  ft=1
    ft=0 → SensitivityFilter,  ft=1 → DensityFilter
"""

import sys
import matplotlib.pyplot as plt

from FreqTop.utils.cantilever import CantileverProblem
from FreqTop.solver import TopOptSolver
from FreqTop.fe.fe_solver import FESolver
from FreqTop.filters.sensitivity import SensitivityFilter
from FreqTop.filters.density import DensityFilter
from FreqTop.optimizers.oc import OCOptimizer
from FreqTop.optimizers.sqp import SQPOptimizer
from FreqTop.callbacks.plotter import LivePlotter
from FreqTop.callbacks.logger import ConsoleLogger



# ── Default parameters (identical to the original script) ────────────────────
nelx    = 180
nely    = 60
volfrac = 0.4
rmin    = 5.4
penal   = 3.0
ft      = 1      # 0 → sensitivity filter,  1 → density filter

if len(sys.argv) > 1: nelx    = int(sys.argv[1])
if len(sys.argv) > 2: nely    = int(sys.argv[2])
if len(sys.argv) > 3: volfrac = float(sys.argv[3])
if len(sys.argv) > 4: rmin    = float(sys.argv[4])
if len(sys.argv) > 5: penal   = float(sys.argv[5])
if len(sys.argv) > 6: ft      = int(sys.argv[6])

# ── Banner ────────────────────────────────────────────────────────────────────
print("Minimum compliance problem with OC")
print(f"ndes: {nelx} x {nely}")
print(f"volfrac: {volfrac}, rmin: {rmin}, penal: {penal}")
print(f"Filter method: {['Sensitivity based', 'Density based'][ft]}")

# ── Assemble components ───────────────────────────────────────────────────────
problem   = CantileverProblem(nelx=nelx, nely=nely)
fe        = FESolver(problem, penal=penal, Emin=1e-9, Emax=1.0)
filt      = (SensitivityFilter if ft == 0 else DensityFilter)(nelx, nely, rmin)
optimizer = SQPOptimizer(move=0.2)

plotter = LivePlotter(nelx=nelx, nely=nely)
logger  = ConsoleLogger(nelx=nelx, nely=nely, volfrac=volfrac)

solver = TopOptSolver(
    problem   = problem,
    fe_solver = fe,
    filter    = filt,
    optimizer = optimizer,
    volfrac   = volfrac,
    callbacks = (plotter, logger),
    max_iter  = 20,
    tol       = 0.01,
)

# ── Run ───────────────────────────────────────────────────────────────────────
solver.run()

# Keep the plot window open after the loop finishes
plotter.keep_open()
