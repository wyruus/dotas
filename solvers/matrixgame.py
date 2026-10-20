"""
Zero-sum matrix game solver.

The Bellman operator for a simultaneous-move game requires the value of the
stage game

    Val(A) = max_{p in Δm} min_{q in Δn} pᵀ A q

at *every* state, so this function is the inner loop of the whole solver and is
optimised accordingly.

Two-tier strategy
-----------------
1. **Saddle point test** - O(mn).  If

       max_i min_j A[i][j]  ==  min_j max_i A[i][j]

   the game has a pure-strategy equilibrium.  Most states in a duel have one
   (one side has a dominant action), so this short-circuits the majority of
   calls and is exact.

2. **Fictitious play with bound tracking**.  Row's best response against q
   gives an *upper* bound on the value, column's best response against p gives
   a *lower* bound:

       max_i (Aq)_i  >=  v*  >=  min_j (pᵀA)_j

   Tracking the tightest pair seen gives an interval that brackets v*, and the
   midpoint is a far better estimate than the empirical payoff, which converges
   more slowly.  `solve` returns the interval half-width as `err`.

Failure mode that this implementation avoids: plain fictitious play with
deterministic tie-breaking can lock onto a pure action pair that is *not* an
equilibrium (e.g. the identity game 2x2 gets stuck at (0,0) with value 1
instead of 0.5).  Bound tracking detects this immediately, because the bounds
do not close.

`MatGameCache` memoises solutions keyed by the *quantised* payoff matrix.
After HP/mana quantisation many distinct states share an identical matrix, so
the hit rate is high and the expensive path is taken rarely.
"""

from __future__ import annotations

import struct

QUANT = 1e-4   # matrix quantisation for the cache key


def saddle_point(A, m: int, n: int):
    """Return (value, row, col) if a pure saddle point exists, else None."""
    if m == 0 or n == 0:
        return None

    row_min = [0.0] * m
    row_arg = [0] * m
    for i in range(m):
        base = i * n
        best = A[base]
        bi = 0
        for j in range(1, n):
            v = A[base + j]
            if v < best:
                best = v
                bi = j
        row_min[i] = best
        row_arg[i] = bi

    col_max = [A[j] for j in range(n)]
    for i in range(1, m):
        base = i * n
        for j in range(n):
            v = A[base + j]
            if v > col_max[j]:
                col_max[j] = v

    lower = max(row_min)          # maximin
    upper = min(col_max)          # minimax
    if upper - lower <= 1e-12:
        i = row_min.index(lower)
        return lower, i, row_arg[i]
    return None


def fictplay(A, m: int, n: int, iters: int = 400, tol: float = 2e-3):
    """Fictitious play with tightest-bound tracking.

    Returns (value, p, q, err) where value = midpoint of the tightest bracket
    and err = its half width.  err <= tol on success; otherwise the iteration
    budget ran out and err is an honest error bar.
    """
    cp = [0.0] * m
    cq = [0.0] * n
    # Seed with the pure maximin / minimax choices so the first bounds are
    # meaningful instead of vacuous.
    row_min = [min(A[i * n:(i + 1) * n]) for i in range(m)]
    cp[row_min.index(max(row_min))] = 1.0
    col_max = [max(A[i * n + j] for i in range(m)) for j in range(n)]
    cq[col_max.index(min(col_max))] = 1.0
    tot_p = tot_q = 1.0

    lo, hi = -1e300, 1e300        # tightest bracket seen so far
    for _ in range(iters):
        inv_tot_q = 1.0 / tot_q
        inv_tot_p = 1.0 / tot_p

        # row player's best response to q  -> upper bound on v*
        best_i, best_val = 0, -1e300
        for i in range(m):
            base = i * n
            s = 0.0
            for j in range(n):
                c = cq[j]
                if c:
                    s += A[base + j] * c
            s *= inv_tot_q
            if s > best_val:
                best_val = s
                best_i = i

        # column player's best response to p -> lower bound on v*
        best_j, worst_val = 0, 1e300
        for j in range(n):
            s = 0.0
            for i in range(m):
                c = cp[i]
                if c:
                    s += A[i * n + j] * c
            s *= inv_tot_p
            if s < worst_val:
                worst_val = s
                best_j = j

        if best_val < hi:
            hi = best_val
        if worst_val > lo:
            lo = worst_val

        cp[best_i] += 1.0
        cq[best_j] += 1.0
        tot_p += 1.0
        tot_q += 1.0

        if hi - lo <= 2.0 * tol:
            break

    if lo > hi:            # numerically possible if the game is degenerate
        lo = hi = 0.5 * (lo + hi)
    return 0.5 * (lo + hi), [c / tot_p for c in cp], [c / tot_q for c in cq], \
        0.5 * (hi - lo)


def solve(A, m: int, n: int, iters: int = 400, tol: float = 2e-3):
    """Solve an m x n zero-sum game.

    Returns (value, p, q, info); info['exact'] is True iff a saddle point was
    found, info['err'] is the bracket half-width otherwise.
    """
    if m == 1 and n == 1:
        return A[0], [1.0], [1.0], {"exact": True, "err": 0.0}

    sp = saddle_point(A, m, n)
    if sp is not None:
        val, i, j = sp
        p = [0.0] * m
        q = [0.0] * n
        p[i] = 1.0
        q[j] = 1.0
        return val, p, q, {"exact": True, "err": 0.0}

    val, p, q, err = fictplay(A, m, n, iters=iters, tol=tol)
    return val, p, q, {"exact": False, "err": err}


class MatGameCache:
    """Memoise stage-game solutions by quantised payoff matrix."""

    def __init__(self, maxsize: int = 400_000):
        self.d = {}
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def key(self, A, m, n):
        inv = 1.0 / QUANT
        return (m, n, struct.pack(
            f"<{m * n}i", *[int(round(a * inv)) for a in A]))

    def value(self, A, m, n, iters: int = 400, tol: float = 2e-3):
        k = self.key(A, m, n)
        v = self.d.get(k)
        if v is not None:
            self.hits += 1
            return v
        self.misses += 1
        val, p, q, info = solve(A, m, n, iters=iters, tol=tol)
        if len(self.d) < self.maxsize:
            # store the value, and the strategies only when they are pure
            self.d[k] = (val, p if info["exact"] else None,
                         q if info["exact"] else None, info)
        return self.d[k]
