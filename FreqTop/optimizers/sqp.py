"""
FreqTop/optimizers/sqp.py
=========================
Sequential Quadratic Programming optimizer for structural topology
optimisation — TopSQP algorithm.

Drop-in replacement for OCOptimizer:

    from FreqTop.optimizers.sqp import SQPOptimizer
    optimizer = SQPOptimizer(move=0.05, penal=3.0)

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
from scipy.optimize import minimize


class SQPOptimizer:
    """
    TopSQP: second-order SQP method for minimum-compliance topology optimisation.

    Implements Algorithm 1 (SQP+) from Rojas-Labanda & Stolpe (2016),
    specialised for the single-volume-constraint minimum-compliance
    problem (Pc) with SIMP material interpolation.

    This class is a drop-in replacement for OCOptimizer in the FreqTop
    repository.  The public interface is identical:

        optimizer = SQPOptimizer(move=0.05, penal=3.0)
        xnew, g   = optimizer.update(x, dc, dv, g, volfrac)

    Parameters
    ----------
    move : float
        Move limit for the search direction per element.
        Defines:  lo_i = max(0, t_i - move) - t_i  (lower bound on d_i)
                  hi_i = min(1, t_i + move) - t_i  (upper bound on d_i)
        Default: 0.2  (same as OCOptimizer).

        **Key stability parameter for max_frequency problems.**
        With move=0.2 (old default) the large per-iteration density swing can cause
        eigenfrequency oscillations (non-monotonic growth).
        Use move=0.05 to obtain smooth, monotonically increasing ω₁.
        This is the single most effective knob for eliminating jumps.
        The parameters_maxfrequency.json file already sets move_limit=0.05.
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
        move:       float = 0.05,
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

        # L-BFGS state (Byrd et al. 1994 compact representation)
        self._lbfgs_memory = 15
        self._s_list: list = []   # displacement vectors s_k = x_k+1 - x_k
        self._y_list: list = []   # gradient-difference vectors y_k = g_k+1 - g_k
        self._lbfgs_xprev = None  # previous iterate
        self._lbfgs_gprev = None  # previous Lagrangian gradient
        self._B0_diag = None      # Fix P1: SIMP diagonal initial Hessian

    # =========================================================================
    # PUBLIC INTERFACE  (identical to OCOptimizer.update)
    # =========================================================================

    def update(self, x, dc, dv, volfrac):
        """Compute one L-BFGS SQP step for the design variables.

        Parameters
        ----------
        x       : (n,) current design variables
        dc      : (n,) objective sensitivities  (df0/dx)
        dv      : (n,) volume sensitivities
        volfrac : volume fraction target
        """
        start = time.thread_time()

        # P1: SIMP-calibrated diagonal Hessian — gives Bk_i ~ (p-1)/x_i * |dc_i|
        # so unconstrained step d_i = -dc_i/Bk_i ~ ±move for all elements.
        # Use abs(dc) — not max(-dc,0) — so elements with dc>0 also get a
        # finite Hessian; max(-dc,0) would leave them with Bk=EPS giving
        # effectively linear objectives and causing all elements to hit move
        # limits in the same direction (produces oscillation).
        EPS = 1e-14
        Bk_simp = (self.penal - 1.0) / np.maximum(x, EPS) * np.abs(dc)
        Bk_simp = np.maximum(Bk_simp, EPS)
        self._B0_diag = Bk_simp

        # P3: normalize constraint to prevent n_el * rho_pen dominating Bk_simp.
        # With dv = ones(n_el), dv_norm = sqrt(n_el), so the effective per-element
        # constraint stiffness becomes rho_pen/n_el instead of rho_pen.
        g      = float(np.dot(dv, x)) - volfrac * float(dv.sum())
        dv_norm = float(np.linalg.norm(dv)) + 1e-300
        fv      = np.array([g / dv_norm], dtype=np.float64)
        A       = (dv / dv_norm).reshape(1, -1)

        # Lagrangian gradient  dL/dx = dc + lam * dv
        if not hasattr(self, "_lam") or self._lam is None:
            self._lam = 0.0
        g_lag = dc + float(self._lam) * dv

        # L-BFGS curvature update (curvature computed from Lagrangian gradient)
        if self._lbfgs_xprev is not None and self._lbfgs_gprev is not None:
            s_new = x - self._lbfgs_xprev
            y_new = g_lag - self._lbfgs_gprev
            sy    = float(s_new @ y_new)
            ss    = np.linalg.norm(s_new) * np.linalg.norm(y_new)
            if sy > 1e-14 * (ss + 1e-300):
                self._s_list.append(s_new.copy())
                self._y_list.append(y_new.copy())
                if len(self._s_list) > self._lbfgs_memory:
                    self._s_list.pop(0)
                    self._y_list.pop(0)

        # Step bounds: move limit clipped to [0, 1]
        xl = np.maximum(0.0, x - self.move)
        xu = np.minimum(1.0, x + self.move)
        dl = xl - x
        du = xu - x

        # P3: rho_pen calibrated to Hessian scale so constraint term ~ Bk_simp.
        Bk_mean = float(np.mean(Bk_simp))
        rho_pen = max(Bk_mean, 10.0 * abs(float(self._lam)) * Bk_mean + Bk_mean)

        res = minimize(
            lambda d: self._qp_obj(d, dc, fv, A, rho_pen),
            np.zeros_like(x),
            method="L-BFGS-B",
            jac=True,
            bounds=list(zip(dl, du)),
            options={"maxiter": 3000, "ftol": 1e-45, "gtol": 1e-49},
        )
        d = res.x

        xnew = np.clip(x + d, xl, xu)

        # P2: standard augmented-Lagrangian multiplier update with exponential
        # damping (α=0.4) to suppress sign-oscillation in cv_unnorm.
        # In max_frequency runs the volume constraint is alternately slightly
        # over/under, causing dlam to flip sign every iteration.  The damping
        # keeps _lam from jumping, smoothing the L-BFGS y-vectors and
        # reducing the "sawtooth" pattern in the change_vs_iteration plot.
        # Normalized form: dlam = rho_pen * cv_unnorm / dv_norm^2
        #   (prevents n_el * rho_pen from dominating the update scale)
        _LAM_DAMP  = 0.4   # fraction of dlam applied each iteration
        cv_unnorm  = g + float(np.dot(dv, d))
        dlam       = rho_pen * cv_unnorm / (dv_norm ** 2)
        lam_scale  = float(np.abs(dc).max()) / (float(np.abs(dv).mean()) + 1e-300)
        self._lam  = float(np.clip(
            self._lam + _LAM_DAMP * dlam, 0.0, 5.0 * lam_scale
        ))

        # Store state for next L-BFGS curvature pair
        self._lbfgs_xprev = x.copy()
        self._lbfgs_gprev = g_lag.copy()

        end = time.thread_time()
        total_time = end - start
        mem_mb = (dc.nbytes + dv.nbytes + x.nbytes + xnew.nbytes) / 1024**2
        return xnew, total_time, mem_mb
    

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

    # =========================================================================
    # L-BFGS DIRECT-HESSIAN  (Byrd et al. 1994, compact representation)
    # =========================================================================

    def _lbfgs_sigma(self) -> float:
        """Scaling factor for B_0 = sigma*I (safeguarded Barzilai-Borwein)."""
        if not self._s_list:
            return 1.0
        sy = float(self._s_list[-1] @ self._y_list[-1])
        yy = float(self._y_list[-1] @ self._y_list[-1])
        return max(sy / yy, 1e-10) if yy > 1e-14 else 1.0

    def _B_matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Apply L-BFGS direct Hessian B_k to vector v.

        Compact representation (Nocedal & Wright §7.2 / Byrd et al. 1994):

            B_k = sigma*I - [sigma*S, Y] * M^{-1} * [sigma*S, Y]^T

        where  S = [s_0,...,s_{k-1}],  Y = [y_0,...,y_{k-1}],
               M = [[sigma * S^T S,  L ],
                    [L^T,           -D ]]
        with D = diag(s_i^T y_i) and L = strictly-lower part of S^T Y.

        Cost: O(k*n + k^3), k = len(_s_list) <= _lbfgs_memory.
        """
        k = len(self._s_list)
        if k == 0:
            # P1: use SIMP diagonal when available, fall back to identity
            if self._B0_diag is not None:
                return self._B0_diag * v
            return v.copy()

        # P1: base scale from SIMP diagonal mean — keeps B_k in same magnitude
        # as the element-wise Hessian rather than using the BB scalar sigma.
        if self._B0_diag is not None:
            sig = float(np.mean(self._B0_diag))
        else:
            sig = self._lbfgs_sigma()
        sig = max(sig, 1e-30)

        S   = np.column_stack(self._s_list)   # (n, k)
        Y   = np.column_stack(self._y_list)   # (n, k)

        StY = S.T @ Y
        D   = np.diag(np.diag(StY))           # (k, k) diagonal
        L   = np.tril(StY, -1)                # (k, k) strictly lower triangular

        M = np.block([
            [sig * (S.T @ S),  L  ],
            [L.T,             -D  ],
        ])   # (2k, 2k)

        Wv = np.concatenate([sig * (S.T @ v), Y.T @ v])   # (2k,)

        # P4: adaptive regularization scaled to M magnitude to prevent cancellation
        reg = max(1e-10 * (float(np.abs(M).max()) + 1e-300), 1e-14)
        try:
            sol = np.linalg.solve(M + reg * np.eye(2 * k), Wv)
        except np.linalg.LinAlgError:
            # P4: fall back to SIMP diagonal on numerical failure
            if self._B0_diag is not None:
                return self._B0_diag * v
            return sig * v

        return sig * v - sig * (S @ sol[:k]) - Y @ sol[k:]

    def _qp_obj(self, d: np.ndarray, g0: np.ndarray, fv: np.ndarray,
                A: np.ndarray, rho_pen: float):
        """
        QP objective and gradient for the L-BFGS subproblem.

            min  g0^T d + 1/2 d^T B d  +  rho_pen/2 * ||max(0, fv + A d)||^2
            s.t. dl <= d <= du   (handled externally via L-BFGS-B bounds)

        Returns (value, gradient) compatible with scipy.optimize.minimize jac=True.
        """
        Bd   = self._B_matvec(d)
        val  = float(g0 @ d) + 0.5 * float(d @ Bd)
        grad = g0 + Bd
        cv   = fv + A @ d
        viol = np.maximum(0.0, cv)
        val  += 0.5 * rho_pen * float(viol @ viol)
        grad  = grad + rho_pen * (A.T @ viol)
        return val, grad

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
        """Reset all persistent multiplier and L-BFGS state (call before a new problem)."""
        self._lam = 0.0
        self._xi  = None
        self._eta = None
        self._s_list = []
        self._y_list = []
        self._lbfgs_xprev = None
        self._lbfgs_gprev = None
        self._B0_diag = None