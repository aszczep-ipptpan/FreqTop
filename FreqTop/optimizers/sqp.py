"""
FreqTop/optimizers/sqp.py
=========================
Sequential Quadratic Programming optimizer for structural topology
optimisation — TopSQP algorithm.

Drop-in replacement for OCOptimizer:

    from FreqTop.optimizers.sqp import SQPOptimizer
    optimizer = SQPOptimizer(move=0.2, penal=3.0)

    solver = TopOptSolver(
        ...
        optimizer = optimizer,
        ...
    )

Reference
---------
Rojas-Labanda, S. & Stolpe, M. (2016).
"An efficient second-order SQP method for structural topology optimization."
Structural and Multidisciplinary Optimization 53:1315-1333.
DOI 10.1007/s00158-015-1381-2

Problem (Pc) -- Section 3
-------------------------
    minimize   f(t) = u(t)^T K(t) u(t)          compliance
    subject to g(t) = a^T t - V  <=  0           volume constraint
               0   <=  t_i        <=  1           density bounds

Lagrangian (eq. 1):
    L(t, lam, xi, eta) = f(t) + lam*g(t) + xi^T(t-1) + eta^T(-t)

Algorithm 1 -- SQP+ (Morales et al. 2010, specialised for Pc)
-------------------------------------------------------------
Given t_k, lam_k, xi_k, eta_k:
  STEP 1  Check KKT conditions (Section 2.1)
  STEP 2  Build Bk: PSD diagonal Hessian approx (Section 4)
  STEP 3  Solve IQP via bisection -> d_iq, lam_iq (Section 2.2)
  STEP 4  Update penalty pi = ||lam_iq||_inf
  STEP 5  Estimate active set W^g, W^u, W^l (Section 7)
  STEP 6  Solve EQP -> d_eq, lam_eq (Section 2.3, eq. 11)
  STEP 7  Contraction parameter beta (Section 2.4)
  STEP 8  d_cand = d_iq + beta * d_eq
  STEP 9  Line search on phi_pi (eq. 14-15)
  STEP 10 Accept: t_{k+1} = t_k + alpha * d
  STEP 11 Update multipliers lam, xi, eta (Section 2.5)
"""

import numpy as np
import time


class SQPOptimizer:
    """
    TopSQP: second-order SQP method for minimum-compliance topology optimisation.

    Implements Algorithm 1 (SQP+) from Rojas-Labanda & Stolpe (2016),
    specialised for the single-volume-constraint minimum-compliance
    problem (Pc) with SIMP material interpolation.

    This class is a drop-in replacement for OCOptimizer in the FreqTop
    repository.  The public interface is identical:

        optimizer = SQPOptimizer(move=0.2, penal=3.0)
        xnew, g   = optimizer.update(x, dc, dv, g, volfrac)

    Parameters
    ----------
    move : float
        Move limit for the search direction per element.
        Defines:  lo_i = max(0, t_i - move) - t_i  (lower bound on d_i)
                  hi_i = min(1, t_i + move) - t_i  (upper bound on d_i)
        Default: 0.2  (same as OCOptimizer).
    penal : float
        SIMP penalisation exponent p.
        Used in the diagonal Hessian: Bk_i = (p-1)/t_i * (-dc_i).
        Default: 3.0.
    stat_tol : float
        Stationarity KKT tolerance eps1 (Table 2).  Default: 1e-6.
    feas_tol : float
        Feasibility KKT tolerance eps2 (Table 2).  Default: 1e-8.
    comp_tol : float
        Complementarity KKT tolerance eps3 (Table 2).  Default: 1e-6.
    bisect_tol : float
        Bisection convergence for the IQP volume multiplier lambda.
        Default: 1e-4.
    sigma : float
        Sufficient decrease factor for line search (Section 2.4).
        Default: 1e-4.
    kappa : float
        Line search back-tracking factor (Section 2.4).
        Default: 0.5.
    eps5 : float
        Merit function relaxation (Section 7, Table implementation note).
        Default: 1e-6.
    eps4 : float
        Active set detection tolerance (Section 7).
        Default: 1e-4.
    l2_init : float
        Upper bracket for bisection on the volume Lagrange multiplier.
        Default: 1e9.
    """

    def __init__(
        self,
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
    ):
        self.move       = move
        self.penal      = penal
        self.stat_tol   = stat_tol
        self.feas_tol   = feas_tol
        self.comp_tol   = comp_tol
        self.bisect_tol = bisect_tol
        self.sigma      = sigma
        self.kappa      = kappa
        self.eps5       = eps5
        self.eps4       = eps4
        self.l2_init    = l2_init

        # Persistent Lagrange multipliers updated each call (Section 2.5)
        #   _lam : volume constraint multiplier  (scalar, >= 0)
        #   _xi  : upper-bound multipliers       (n,), >= 0
        #   _eta : lower-bound multipliers       (n,), >= 0
        self._lam = 0.0
        self._xi  = None   # initialised on first call to update()
        self._eta = None

    # =========================================================================
    # PUBLIC INTERFACE  (identical to OCOptimizer.update)
    # =========================================================================

    def update(self, x, dc, dv, volfrac):
        """Compute the SQP update for the design variables.

        Parameters
        ----------
        x       : current design variable vector
        dc      : objective sensitivities
        dv      : volume sensitivities (passive elements must be zeroed by caller)
        volfrac : effective volume fraction target for the active elements
                  (same value as passed to OCOptimizer.update)

        The constraint residual g = sum(dv*x) - volfrac*sum(dv) is computed
        internally, matching the convention used by the OC bisection.
        """
        # Constraint residual: g < 0 means feasible, g > 0 means violated
        g = float(np.dot(dv, x)) - volfrac * float(dv.sum())

        start = time.perf_counter()
        self._ensure_dual_variables(x)
        #self._compute_kkt_residuals(x, dc, dv, g) #unused zbedny balast obliczeniowy
        Bk, lo, hi = self._build_local_model(x, dc)
        # === OC-like step ===
        d_iq, lam_iq = self._solve_iqp(dc, dv, Bk, lo, hi, g)
        xi_iq, eta_iq = self._recover_bound_multipliers(dc, dv, Bk, d_iq, lam_iq)
        # === active set + correction ===
        active = self._estimate_active_set(d_iq, lo, hi, dv, g)
        d_eq, lam_eq = self._solve_eqp(dc, dv, Bk, d_iq, active, lam_iq)
        beta = self._compute_contraction(d_iq, d_eq, lo, hi, dv, g, active)
        d_cand = d_iq + beta * d_eq
        # === merit & line search ===
        d_final, alpha, lam_final = self._line_search(
            x, d_iq, d_cand, dc, dv, Bk, g, lam_iq, lam_eq, xi_iq, eta_iq
        )
        xnew = np.clip(x + d_final, 0.0, 1.0)
        self._update_multipliers(alpha, lam_iq, xi_iq, eta_iq)
        end = time.perf_counter()
        total_time = end - start
        return xnew, total_time
    

    def _build_local_model(self, x, dc):
        EPS = 1e-14
        p = self.penal
        Bk = (p - 1.0) / np.maximum(x, EPS) * (-dc)
        lo = np.maximum(0.0, x - self.move) - x
        hi = np.minimum(1.0, x + self.move) - x
        return Bk, lo, hi

    def _solve_iqp(self, dc, dv, Bk, lo, hi, g):
        EPS = 1e-14
        l1, l2 = 0.0, self.l2_init
        while (l2 - l1) / (l1 + l2 + EPS) > self.bisect_tol:
            lam = 0.5 * (l1 + l2)
            d = np.clip(
                -(dc + lam * dv) / np.maximum(Bk, EPS),
                lo, hi
            )
            if g + float(np.dot(dv, d)) > 0:
                l1 = lam
            else:
                l2 = lam
        return d, lam
    

    def _recover_bound_multipliers(self, dc, dv, Bk, d, lam):
        z = -(dc + Bk * d + lam * dv)
        xi = np.maximum(0.0, z)
        eta = np.maximum(0.0, -z)
        return xi, eta


    def _estimate_active_set(self, d, lo, hi, dv, g):
        eps = self.eps4

        lin_g = g + float(np.dot(dv, d))

        return {
            "Wg": abs(lin_g) < eps,
            "Wu": np.abs(d - hi) < eps,
            "Wl": np.abs(d - lo) < eps,
        }


    
    def _solve_eqp(self, dc, dv, Bk, d_iq, active, lam_iq):
        free = ~(active["Wu"] | active["Wl"])
        d_eq = np.zeros_like(d_iq)
        lam_eq = lam_iq
        if active["Wg"] and free.sum() > 1:
            B = np.maximum(Bk[free], 1e-14)
            a = dv[free]
            d0 = d_iq[free]
            rhs = -(dc[free] + B * d0)
            num = float(np.dot(a, rhs / B))
            den = float(np.dot(a, a / B))
            if abs(den) > 1e-14:
                lam_eq = num / den
                d_eq[free] = (rhs - a * lam_eq) / B
        return d_eq, lam_eq

    def _compute_contraction(self, d_iq, d_eq, lo, hi, dv, g, active):
        beta = 1.0
        tiny = 1e-14

        if not active["Wg"]:
            vol_step = float(np.dot(dv, d_eq))
            if vol_step > tiny:
                slack = -g - float(np.dot(dv, d_iq))
                beta = min(beta, slack / vol_step)

        # bounds analogicznie (skrócone)
        return float(np.clip(beta, 0.0, 1.0))



    def _line_search(self, x, d_iq, d_cand, dc, dv, Bk, g, lam_iq, lam_eq,  xi_iq, eta_iq):
        def merit_step(d):
            dg = float(np.dot(dv, d))
            return (
                float(np.dot(dc, d))
                + 0.5 * float(np.dot(d, Bk * d))
                + self._penalty_from_trial(lam_iq, xi_iq, eta_iq) * (max(0.0, g + dg) - max(0.0, g))
            )
        qred = -(np.dot(dc, d_iq) + 0.5 * np.dot(d_iq, Bk * d_iq))
        if merit_step(d_cand) <= -self.sigma * qred:
            return d_cand, 1.0, lam_eq
        alpha = 1.0
        for _ in range(20):
            if merit_step(alpha * d_iq) <= -self.sigma * alpha * qred:
                return alpha * d_iq, alpha, lam_iq
            alpha *= self.kappa
        return alpha * d_iq, alpha, lam_iq

    def _penalty_from_trial(self, lam, xi, eta):
        return max(
            float(lam),
            float(xi.max()),
            float(eta.max()),
            1e-3
        )


    def _update_multipliers(self, alpha, lam_iq, xi_iq, eta_iq):
        self._lam = alpha * lam_iq + (1 - alpha) * self._lam
        self._xi  = alpha * xi_iq  + (1 - alpha) * self._xi
        self._eta = alpha * eta_iq + (1 - alpha) * self._eta
        self._lam = max(0.0, self._lam)
        self._xi  = np.maximum(0.0, self._xi)
        self._eta = np.maximum(0.0, self._eta)


    
    def __call__(self, x, dc, dv, volfrac):
        """Delegate to update() -- lets TopOptSolver call optimizer(...)."""
        return self.update(x, dc, dv, volfrac)

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def _compute_kkt_residuals(self, x, dc, dv, g):
        """
        Compute the three KKT residual norms (Section 2.1).

        Returns dict with keys:
            stationarity    : ||df + J^T lam + xi - eta||_inf
            feasibility     : max(g+, bound violations)
            complementarity : max(|g lam|, |(t-1) xi|_inf, |(-t) eta|_inf)
            satisfied       : bool -- all three within tolerances
        """
        xi  = self._xi  if self._xi  is not None else np.zeros_like(dc)
        eta = self._eta if self._eta is not None else np.zeros_like(dc)
        lam = self._lam

        stat = float(np.abs(dc + lam * dv + xi - eta).max())
        feas = float(max(
            max(0.0, g),
            float(np.maximum(0.0, x - 1.0).max()),
            float(np.maximum(0.0, -x).max()),
        ))
        comp = float(max(
            abs(g * lam),
            float(np.abs((x - 1.0) * xi).max()),
            float(np.abs((-x)      * eta).max()),
        ))
        return {
            'stationarity':    stat,
            'feasibility':     feas,
            'complementarity': comp,
            'satisfied': (stat <= self.stat_tol
                          and feas <= self.feas_tol
                          and comp <= self.comp_tol),
        }

    def _ensure_dual_variables(self, x: np.ndarray):
        """Ensure dual variables (λ, ξ, η) are initialized and consistent.

        λ   – multiplier for volume constraint (scalar)
        ξ   – multipliers for upper bounds (x <= 1)
        η   – multipliers for lower bounds (x >= 0)
        """

        n = x.size

        # --- initialize if first call ---
        if self._xi is None or self._eta is None:
            self._xi = np.zeros(n)
            self._eta = np.zeros(n)

        if not hasattr(self, "_lam"):
            self._lam = 0.0

        # --- handle size mismatch (important in adaptive meshes etc.) ---
        if self._xi.size != n:
            self._xi = np.zeros(n)

        if self._eta.size != n:
            self._eta = np.zeros(n)

        # --- enforce dual feasibility ---
        self._lam = max(0.0, float(self._lam))
        self._xi = np.maximum(0.0, self._xi)
        self._eta = np.maximum(0.0, self._eta)

    def reset(self):
        """Reset all persistent multiplier state (call before a new problem)."""
        self._lam = 0.0
        self._xi  = None
        self._eta = None