"""
Value iteration for a deterministic two-player zero-sum game with simultaneous
moves (Shapley's stochastic game, undiscounted with a regularising discount).

    V(s) = Val_{u,v}[ gamma * V(f(s,u,v)) ],      V|_T = terminal payoff

`Val` is the stage-game value from `matrixgame.solve`.  Terminal states are
absorbing.

Because the state space is fixed a priori (not grown by reachability - the
reachable closure of a duel is essentially the whole product space, so a BFS
closure buys nothing), we use Gauss-Seidel sweeps with a residual-based
stopping rule.

Convergence diagnostics
-----------------------
For a reach-avoid game the interesting quantity is the *undecided mass*:

    U = #{ s : eps < V(s) < 1-eps }

states whose value is still genuinely mixed.  If U stops changing while the
residual keeps shrinking, the remaining iteration budget is being spent on
numerical noise, and we stop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .matrixgame import MatGameCache


@dataclass
class DPResult:
    V: dict
    policy1: dict
    policy2: dict
    sweeps: int
    residual: float
    max_err: float          # largest stage-game bracket half-width encountered
    seconds: float
    cache_stats: tuple = (0, 0)
    history: list = field(default_factory=list)

    def value(self, s, default=0.5):
        return self.V.get(s, default)


def solve_game(states, transitions, gamma=1.0, tol=1e-6, max_sweeps=1000,
               max_err=5e-3, verbose=False, log_every=10,
               min_sweeps=30, patience=2):
    """Run value iteration.

    Parameters
    ----------
    states : iterable of hashable states (the full a-priori state set)
    transitions : callable s -> Transitions
        Transitions has:
            terminal : float | None   (absorbing value, or None)
            n_u, n_v : int
            succ     : list[list[state]]   n_u x n_v successor states
    gamma : discount factor (slightly < 1 to guarantee contraction)
    tol : stop when the sup-norm residual of a sweep drops below this
    max_err : stop if the worst stage-game bracket error exceeds this
              (means the matrix solver is not resolving stages well enough
              for the requested tolerance)
    """
    t0 = time.time()
    V = {s: 0.5 for s in states}
    cache = MatGameCache()
    policy1, policy2 = {}, {}
    worst_err = 0.0
    prev_undecided = None
    quiet = 0
    history = []
    A = []          # reused payoff buffer

    terminal_flags = {}
    n = len(V)

    for sweep in range(max_sweeps):
        delta = 0.0
        solved = 0
        for s in V:
            tr = transitions(s)
            if tr.terminal is not None:
                if V[s] != tr.terminal:
                    V[s] = tr.terminal
                    terminal_flags[s] = True
                continue
            nu, nv = tr.n_u, tr.n_v
            need = nu * nv
            if len(A) < need:
                A = [0.0] * need
            succ = tr.succ
            k = 0
            for i in range(nu):
                row = succ[i]
                for j in range(nv):
                    A[k] = V[row[j]]
                    k += 1
            if gamma != 1.0:
                for idx in range(need):
                    A[idx] *= gamma

            val, p, q, info = cache.value(A, nu, nv)
            if info["err"] > worst_err:
                worst_err = info["err"]
            d = abs(val - V[s])
            if d > delta:
                delta = d
            V[s] = val
            policy1[s] = p
            policy2[s] = q
            solved += 1

        undecided = sum(1 for v in V.values() if 1e-3 < v < 1 - 1e-3)
        history.append((sweep, delta, undecided))
        if verbose and sweep % log_every == 0:
            print(f"  sweep {sweep:4d}  residual={delta:.3e}  "
                  f"undecided={undecided:7d}  cache={cache.hits}/{cache.misses}")

        # A uniform initialisation is a fixed point of the interior operator
        # (all successors are 0.5 too), so `delta` can be ~0 on early sweeps
        # long before terminal information has propagated across the state
        # space.  Never stop before `min_sweeps`, and require the residual to
        # stay small for `patience` consecutive sweeps.
        if sweep + 1 >= min_sweeps and delta < tol:
            quiet += 1
            if quiet >= patience:
                break
        else:
            quiet = 0
        # Stall detection: the residual is no longer shrinking and the set of
        # genuinely mixed states has stopped moving -> further sweeps spend
        # the budget on numerical noise.
        if prev_undecided is not None and undecided == prev_undecided \
                and delta < 10 * tol and sweep + 1 >= min_sweeps:
            break
        prev_undecided = undecided

    return DPResult(V, policy1, policy2, sweep + 1, delta, worst_err,
                    time.time() - t0, (cache.hits, cache.misses), history)


def rollout(transitions, policy1, policy2, start, max_steps=4000):
    """Follow the extracted policies from `start`; returns the trajectory and
    the realised outcome (the game is deterministic given the policies)."""
    s = start
    traj = [s]
    for _ in range(max_steps):
        tr = transitions(s)
        if tr.terminal is not None:
            return traj, tr.terminal
        p = policy1.get(s)
        q = policy2.get(s)
        i = max(range(len(p)), key=p.__getitem__)
        j = max(range(len(q)), key=q.__getitem__)
        s = tr.succ[i][j]
        traj.append(s)
    return traj, None
