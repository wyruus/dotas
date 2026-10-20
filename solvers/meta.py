"""
Outer level: the build meta-game.

With unlimited gold the question "what is the optimal build" has no answer in
isolation.  Dota is built on hard counters - BKB beats magic burst, Ghost
Scepter beats physical, Linken's beats single-target initiation - so if one
six-slot build dominated, the meta would collapse.  The mathematically correct
question is the equilibrium of the meta-game

    M[b1][b2] = V(duel | Pudge builds b1, Invoker builds b2)

where V is the tactical value from the inner DP.  This module computes:

* the **Nash equilibrium** of M (mixed strategies - the equilibrium support is
  the set of builds that survive);
* the **Stackelberg** solutions, i.e. the value when one side commits to a
  build first and the other best-responds knowing it (this models the real
  drafting situation: the later pick wins the matchup);
* the **best response** to any fixed opposing build.

Payoff matrices here are small (tens of builds per side) but each entry costs a
full tactical solve, so the matrix is evaluated lazily and cached.
"""

from __future__ import annotations

from .matrixgame import solve, fictplay


class BuildMetaGame:
    def __init__(self, builds1, builds2, oracle, tol=1e-3):
        """
        builds1, builds2 : sequences of build descriptors (must be hashable)
        oracle           : f(b1, b2) -> float in [0,1], P(player 1 wins)
        """
        self.b1 = list(builds1)
        self.b2 = list(builds2)
        self.oracle = oracle
        self._cache = {}
        self.tol = tol
        self.calls = 0

    def payoff(self, b1, b2):
        k = (b1, b2)
        v = self._cache.get(k)
        if v is None:
            v = self.oracle(b1, b2)
            self._cache[k] = v
            self.calls += 1
        return v

    def matrix(self, rows=None, cols=None):
        rows = self.b1 if rows is None else rows
        cols = self.b2 if cols is None else cols
        m = len(rows)
        n = len(cols)
        A = [0.0] * (m * n)
        for i, r in enumerate(rows):
            base = i * n
            for j, c in enumerate(cols):
                A[base + j] = self.payoff(r, c)
        return A, m, n

    # ------------------------------------------------------------------
    def nash(self, iters=600):
        A, m, n = self.matrix()
        val, p, q, info = solve(A, m, n, iters=iters, tol=self.tol)
        return {
            "value": val,
            "p1_mix": dict(zip(self.b1, p)),
            "p2_mix": dict(zip(self.b2, q)),
            "info": info,
            "matrix": (A, m, n),
        }

    def stackelberg(self):
        """Both commitment orders.

        p1_leads : Pudge commits first, Invoker answers -> Pudge maximises the
                   value Invoker leaves him.
        p2_leads : Invoker commits first, Pudge answers.
        """
        A, m, n = self.matrix()
        # Pudge commits, Invoker best-responds by minimising
        p1_lead = []
        for i in range(m):
            best = min(A[i * n + j] for j in range(n))
            p1_lead.append(best)
        # Invoker commits, Pudge best-responds by maximising
        p2_lead = []
        for j in range(n):
            best = max(A[i * n + j] for i in range(m))
            p2_lead.append(best)

        i_best = max(range(m), key=lambda i: p1_lead[i])
        j_best = min(range(n), key=lambda j: p2_lead[j])
        return {
            "p1_leads_value": p1_lead[i_best],
            "p1_leads_build": self.b1[i_best],
            "p1_leads_vs": min(range(n), key=lambda j: A[i_best * n + j]),
            "p2_leads_value": p2_lead[j_best],
            "p2_leads_build": self.b2[j_best],
            "p2_leads_vs": max(range(m), key=lambda i: A[i * n + j_best]),
            "all_p1_lead": dict(zip(self.b1, p1_lead)),
            "all_p2_lead": dict(zip(self.b2, p2_lead)),
        }

    def best_response_p1(self, b2):
        vals = [(self.payoff(b1, b2), b1) for b1 in self.b1]
        return max(vals)

    def best_response_p2(self, b1):
        vals = [(self.payoff(b1, b2), b2) for b2 in self.b2]
        return min(vals)

    def support(self, mix, eps=1e-3):
        """Builds played with non-negligible probability in a mixed strategy."""
        return {b: w for b, w in mix.items() if w > eps}


# ----------------------------------------------------------------------
# Pareto filtering on capability vectors
# ----------------------------------------------------------------------
def pareto_front(builds, caps, maximise=None):
    """Drop dominated builds.

    builds : iterable of build descriptors
            caps   : f(build) -> tuple of capability numbers
            maximise : set of indices to maximise (others are minimised).
                       Default: maximise all.
    Returns the list of non-dominated builds.
    """
    builds = list(builds)
    vecs = {b: caps(b) for b in builds}
    ndim = len(next(iter(vecs.values())))
    if maximise is None:
        maximise = set(range(ndim))

    def better_or_equal(a, b):
        all_ge = True
        any_g = False
        for k in range(ndim):
            av, bv = a[k], b[k]
            if k not in maximise:
                av, bv = -av, -bv
            if av < bv - 1e-12:
                all_ge = False
                break
            if av > bv + 1e-12:
                any_g = True
        return all_ge and any_g

    out = []
    for b in builds:
        dominated = False
        for c in builds:
            if c is b:
                continue
            if better_or_equal(vecs[c], vecs[b]):
                dominated = True
                break
        if not dominated:
            out.append(b)
    return out
