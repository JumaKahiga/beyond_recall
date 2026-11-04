"""Frozen fixed-sample recovery audit — the ONLY path to ACCEPT.

Design is chosen from PERMITTED info only (|D|, threshold from T,b, alpha):
  * threshold: unseen misses must satisfy M_D <= floor(T/9) - b for R_L>=0.90.
  * cutoff c(n): max observed hits x with the exact tail upper bound still
    certifying M_D <= threshold.
  * affordable design : take the cutoff available at
    the largest affordable sample, then use the SMALLEST sample that permits
    that same cutoff — fewer labels, >= acceptance probability.

Execution actually draws the D-sample (charged) and accepts iff observed hits
<= cutoff. Accept-iff-x<=c is a valid alpha-level test: x<=c => U(|D|,n,x,alpha)
<= U(|D|,n,c,alpha) <= threshold, so R_L>=rho is certified at alpha.
"""
from __future__ import annotations

import math
from functools import lru_cache


@lru_cache(maxsize=None)
def _logC(a, b):
    if b < 0 or b > a:
        return float("-inf")
    return math.lgamma(a + 1) - math.lgamma(b + 1) - math.lgamma(a - b + 1)


def _cdf(N, K, n, x):
    lo = max(0, n - (N - K)); hi = min(n, K, x)
    if hi < lo:
        return 0.0
    d = _logC(N, n)
    t = [_logC(K, i) + _logC(N - K, n - i) - d for i in range(lo, hi + 1)]
    m = max(t)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in t))


def _upper_bound(N, n, x, delta):
    """Largest m with P(X<=x | Hypergeom(N,m,n)) > delta (decreasing in m)."""
    lo, hi = x, N - (n - x)
    if _cdf(N, hi, n, x) > delta:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _cdf(N, mid, n, x) > delta:
            lo = mid
        else:
            hi = mid - 1
    return lo


def cutoff(D_size, n, thresh, alpha=0.05):
    c = -1
    for x in range(0, n + 1):
        if _upper_bound(D_size, n, x, alpha) <= thresh:
            c = x
        else:
            break
    return c


def affordable_design(D_size, T, b, A_union, cap, alpha=0.05):
    """Permitted design: (n, cutoff) or None if no affordable valid audit."""
    thresh = T // 9 - b
    if thresh < 0 or D_size <= 0:
        return None
    n_max = min(D_size, cap - A_union)
    if n_max <= 0:
        return None
    c = cutoff(D_size, n_max, thresh, alpha)
    if c < 0:
        return None                       # cannot certify even at the max sample
    # smallest n permitting the same cutoff c: smallest n with U(D,n,c,alpha)<=thresh
    lo, hi = max(c, 1), n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if _upper_bound(D_size, mid, c, alpha) <= thresh:
            hi = mid
        else:
            lo = mid + 1
    return {"n": lo, "cutoff": c, "threshold": thresh, "D_size": D_size,
            "T": T, "b": b}


def execute(design, D_window_ids, oracle, rng, early_stop=True):
    """Draw the frozen D-sample and return the real verdict.

    EARLY-FAILURE STOPPING: reveal the planned sample in
    random order and stop as soon as cutoff+1 positives are seen — acceptance
    is then impossible under the frozen design, so the decision is IDENTICAL to
    inspecting the whole sample; only the labels actually inspected are charged.
    Acceptance still requires inspecting the full sample and finding <= cutoff.
    The frozen plan is still n (affordability is checked against the full n).
    """
    n = min(design["n"], len(D_window_ids))
    order = rng.sample(D_window_ids, n)     # the frozen sample, revealed in order
    c = design["cutoff"]
    hits = 0
    inspected = 0
    for w in order:
        y = oracle.purchase(w, "audit")
        inspected += 1
        if y == 1:
            hits += 1
            if early_stop and hits > c:     # rejection certain — stop revealing
                return {"accept": False, "observed_hits": hits, "cutoff": c,
                        "n_planned": n, "labels_inspected": inspected,
                        "stopped_early": True}
    return {"accept": hits <= c, "observed_hits": hits, "cutoff": c,
            "n_planned": n, "labels_inspected": inspected, "stopped_early": False}


def true_power(design, M_D_true):
    """Retrospective power label (evaluator-only) for a frozen design."""
    return _cdf(design["D_size"], M_D_true, design["n"], design["cutoff"])
