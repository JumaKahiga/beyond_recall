"""Exact hypergeometric tail inversions .

Pure stdlib, deterministic. X ~ Hypergeom(N, m, n) counts positives in a
simple random sample of size n drawn without replacement from a population
of N windows of which m are positive:

    Pr(X = x) = C(m, x) * C(N - m, n - x) / C(N, n).

All probability comparisons are done in exact integer arithmetic: a tail
probability with numerator T over denominator C(N, n) is compared with a
rational allowance delta = p/q via  T * q  vs  p * C(N, n).

The inverted one-sided bounds are

    L(N, n, x; delta) = min{ m : Pr(X_m >= x) > delta },
    U(N, n, x; delta) = max{ m : Pr(X_m <= x) > delta }.

Pr(X_m >= x) is nondecreasing in m and Pr(X_m <= x) is nonincreasing in m
(coupling: converting one negative window into a positive can only raise
the sampled positive count), so integer bisection over m is valid.  This
monotonicity in the *population total m* is a theorem; it is unrelated to
the power-versus-sample-size monotonicity that must not be assumed.

delta = 0 gives the logical endpoints used for census and unsampled
strata: L = x and U = N - (n - x); for a census (n = N) both equal x.
"""

from fractions import Fraction
from functools import lru_cache
from math import factorial

_fact = lru_cache(maxsize=8192)(factorial)


@lru_cache(maxsize=65536)
def comb(a, b):
    """Exact C(a, b) via cached factorials.

    Identical values to math.comb; on this interpreter (3.9) the
    divide-and-conquer factorial plus one big division is several times
    faster for the large arguments used here, and repeated tail sums
    reuse the cached factorials.
    """
    if a < 0 or b < 0:
        raise ValueError("comb arguments must be nonnegative")
    if b > a:
        return 0
    return _fact(a) // (_fact(b) * _fact(a - b))


def as_fraction(delta):
    """Normalize an error allowance to an exact Fraction.

    Accepts Fraction, int, or str directly.  A float is interpreted via
    its shortest decimal representation (str), so 0.025 means exactly
    1/40; pass a Fraction to control this precisely.
    """
    if isinstance(delta, Fraction):
        f = delta
    elif isinstance(delta, int):
        f = Fraction(delta)
    elif isinstance(delta, float):
        f = Fraction(str(delta))
    elif isinstance(delta, str):
        f = Fraction(delta)
    else:
        raise TypeError("delta must be Fraction, int, float, or str")
    if not (0 <= f < 1):
        raise ValueError("delta must satisfy 0 <= delta < 1, got %s" % f)
    return f


def _check_args(N, n, x):
    if N < 0 or n < 0 or x < 0 or n > N or x > n:
        raise ValueError(
            "invalid hypergeometric arguments N=%s n=%s x=%s" % (N, n, x))


def pmf_numerator(N, m, n, x):
    """Numerator C(m, x) * C(N - m, n - x) of Pr(X_m = x); denominator C(N, n)."""
    _check_args(N, n, x)
    if m < 0 or m > N:
        raise ValueError("m must lie in [0, N]")
    if x > m or n - x > N - m:
        return 0
    return comb(m, x) * comb(N - m, n - x)


def cdf_numerator(N, m, n, x):
    """Numerator of Pr(X_m <= x); denominator C(N, n).  x may be -1 (gives 0)."""
    if m < 0 or m > N:
        raise ValueError("m must lie in [0, N]")
    lo = max(0, n - (N - m))
    hi = min(x, n, m)
    total = 0
    for k in range(lo, hi + 1):
        total += comb(m, k) * comb(N - m, n - k)
    return total


def cdf(N, m, n, x):
    """Exact Pr(X_m <= x) as a Fraction."""
    _check_args(N, n, max(x, 0) if x >= 0 else 0)
    return Fraction(cdf_numerator(N, m, n, x), comb(N, n))


def lower_bound_m(N, n, x, delta):
    """L(N, n, x; delta) = min{ m : Pr(X_m >= x) > delta }.

    One-sided lower confidence bound on the stratum positive total:
    Pr(L <= m_true) >= 1 - delta for every true total (see DERIVATION.md).
    """
    _check_args(N, n, x)
    delta = as_fraction(delta)
    if x == 0:
        return 0  # Pr(X_m >= 0) = 1 > delta for every m
    total = comb(N, n)
    p, q = delta.numerator, delta.denominator

    def ok(m):
        # Pr(X_m >= x) > delta, via 1 - Pr(X_m <= x-1), exact integers
        num_ge = total - cdf_numerator(N, m, n, x - 1)
        return num_ge * q > p * total

    # ok is monotone nondecreasing in m; ok(m) is False for m < x
    # (Pr = 0) and ok(N) is True (Pr = 1 since x <= n and delta < 1).
    if ok(x):
        return x
    lo, hi = x, N  # ok(lo) False, ok(hi) True
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def upper_bound_m(N, n, x, delta):
    """U(N, n, x; delta) = max{ m : Pr(X_m <= x) > delta }.

    One-sided upper confidence bound on the stratum positive total:
    Pr(U >= m_true) >= 1 - delta for every true total.
    """
    _check_args(N, n, x)
    delta = as_fraction(delta)
    if x == n:
        return N  # Pr(X_m <= n) = 1 > delta for every m
    total = comb(N, n)
    p, q = delta.numerator, delta.denominator

    def ok(m):
        return cdf_numerator(N, m, n, x) * q > p * total

    # ok is monotone nonincreasing in m; ok(m) is True for m <= x
    # (X_m <= m <= x surely, so Pr = 1 > delta).
    if ok(N):
        return N
    lo, hi = x, N  # ok(lo) True, ok(hi) False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo
