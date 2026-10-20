"""
Numpy value iteration for the simultaneous-move reach-avoid game.

Same Bellman operator as `solvers.dp`, but vectorised over the whole state
space.  The point is that the *stage game* is identical in structure at every
state, so both tiers of the matrix-game solver vectorise:

    tier 1  saddle-point test   ->  row-min/max/col-max reductions
    tier 2  fictitious play     ->  einsum + argmax, run for all unresolved
                                    states simultaneously

The state space is chunked along the flat index so that the (n_u, n_v, chunk)
payoff tensor stays inside memory.

A successor function is supplied by the caller:

    succ_fn(coords, i, j) -> flat successor indices, int64, shape (chunk,)

where `coords` is the tuple of 1-D coordinate arrays for the chunk.  Keeping
successors as a *function* rather than a precomputed table is what makes large
spaces fit: a table for 10 M states x 200 action pairs would need 8 GB, while
recomputing from coordinates is a few arithmetic ops.

Cooldown handling
-----------------
Cooldowns are stored as *remaining DP steps*, which is the only representation
that is correct without an extra timer per ability.  Because every cooldown
ticks at the same rate, a single global phase counter (`phase = step mod k`)
lets coarse buckets decrement on a schedule: with bucket width `k`, an ability
counts down by one bucket each time the phase wraps.
"""

from __future__ import annotations

import os as _os

# MUST happen before numpy is imported.  numpy's BLAS backend starts a thread
# pool at import time, and fork()ing a multithreaded process can deadlock the
# child inside the allocator or the BLAS lock.  Single-threaded BLAS also
# avoids oversubscription: we parallelise across states, not inside BLAS.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import numpy as np


def _fictplay_vec(G, nu, nv, iters=150, tol=1e-3):
    """Vectorised fictitious play with tightest-bound tracking.

    G : (nu, nv, m) payoff tensor for m independent games.
    Returns the midpoint of the tightest bracket for each game.
    """
    m = G.shape[2]
    cp = np.zeros((m, nu))
    cq = np.zeros((m, nv))
    ar = np.arange(m)
    cp[ar, G.min(axis=1).argmax(axis=0)] = 1.0
    cq[ar, G.max(axis=0).argmin(axis=0)] = 1.0
    tot_p = np.ones(m)
    tot_q = np.ones(m)
    lo = np.full(m, -1e30)
    hi = np.full(m, 1e30)

    # Active-set shrinking: a game whose bracket has closed is finished and is
    # dropped from all further work.  Without this, a chunk in which almost no
    # state has a saddle point costs the full iteration budget for every state
    # even though most converge in a handful of iterations.
    active = ar
    for _ in range(iters):
        if active.size == 0:
            break
        Gs = G[:, :, active]                       # (nu, nv, a)
        q = cq[active] / tot_q[active, None]
        p = cp[active] / tot_p[active, None]

        # The game axis MUST carry the same label in both operands.  Writing
        # it as 'ijn,mj->mi' makes einsum treat n and m as independent and sum
        # over the game axis, mixing unrelated games together - which produced
        # values of order m/2 and O(m^2) cost.
        rowpay = np.einsum('ijn,nj->ni', Gs, q)     # (a, nu)
        bi = rowpay.argmax(axis=1)
        bv = rowpay[ar[:active.size], bi]
        colpay = np.einsum('ijn,ni->nj', Gs, p)     # (a, nv)
        bj = colpay.argmin(axis=1)
        wv = colpay[ar[:active.size], bj]

        np.minimum(hi[active], bv, out=bv)         # bv now holds the new hi
        hi[active] = bv
        np.maximum(lo[active], wv, out=wv)
        lo[active] = wv

        cp[active, bi] += 1.0
        cq[active, bj] += 1.0
        tot_p[active] += 1.0
        tot_q[active] += 1.0

        done = (hi[active] - lo[active]) <= 2.0 * tol
        if done.any():
            active = active[~done]

    return 0.5 * (lo + hi)


def stage_values(G, nu, nv, fp_iters=150, fp_tol=1e-3):
    """Value of the stage game for every state in the chunk.

    G : (nu, nv, m).  Returns (values (m,), n_nonsaddle, worst_gap).
    """
    rowmin = G.min(axis=1)          # (nu, m)
    maximin = rowmin.max(axis=0)    # (m,)
    colmax = G.max(axis=0)          # (nv, m)
    minimax = colmax.min(axis=0)    # (m,)

    # maximin <= minimax always; equality is exactly the saddle-point test
    nonsaddle = maximin < minimax - 1e-9
    val = maximin.copy()
    n_bad = int(nonsaddle.sum())
    if n_bad:
        idx = np.nonzero(nonsaddle)[0]
        sub = np.ascontiguousarray(G[:, :, idx])
        val[idx] = _fictplay_vec(sub, nu, nv, fp_iters, fp_tol)
    return val, n_bad, float(np.max(minimax - maximin))


def solve_game_np(shape, n_actions, succ_fn, term_val,
                  chunk=400_000, max_sweeps=400, min_sweeps=25,
                  tol=1e-5, patience=2, fp_iters=150, fp_tol=1e-3,
                  verbose=True, log_every=5, V0=None):
    """Value iteration on a fully vectorised state space.

    shape      : tuple of dimension sizes
    n_actions  : (n_u, n_v)
    succ_fn    : f(coords, i, j) -> flat int64 successors, len == chunk
    term_val   : np.ndarray of `shape`; np.nan marks a non-terminal state
    """
    nu, nv = n_actions
    size = int(np.prod(shape))
    dims = len(shape)

    V = np.full(size, 0.5, dtype=np.float64) if V0 is None else V0.ravel().copy()
    term_flat = term_val.ravel()
    is_term = ~np.isnan(term_flat)
    V[is_term] = term_flat[is_term]

    chunk = max(1, min(chunk, size))
    starts = list(range(0, size, chunk))

    # ONE reusable payoff buffer.  Allocating per chunk (let alone per whole
    # space) does not fit: nu*nv*size floats is 8 GB at 10 M states with 100
    # action pairs.
    G = np.empty((nu, nv, chunk))
    C = np.empty(chunk, dtype=np.int64)

    quiet = 0
    delta = 0.0
    worst_gap = 0.0
    sweep = 0
    for sweep in range(max_sweeps):
        delta = 0.0
        n_bad_tot = 0
        for s0 in starts:
            n = min(chunk, size - s0)
            c = C[:n]
            c[:] = np.arange(s0, s0 + n)
            coords = tuple(x.astype(np.int16)
                           for x in np.unravel_index(c, shape))
            g = G[:, :, :n]
            # terminal states never need a stage game
            live = ~is_term[c]
            if not live.any():
                continue
            for i in range(nu):
                for j in range(nv):
                    idxs, ws = succ_fn(coords, i, j)
                    acc = ws[0] * V[idxs[0]]
                    for k in (1, 2, 3):
                        acc += ws[k] * V[idxs[k]]
                    g[i, j] = acc
            li = np.nonzero(live)[0]
            sub = np.ascontiguousarray(g[:, :, li])
            vals, n_bad, gap = stage_values(sub, nu, nv, fp_iters, fp_tol)
            n_bad_tot += n_bad
            if gap > worst_gap:
                worst_gap = gap
            new = V[c[li]]
            if li.size:
                d_chunk = float(np.max(np.abs(vals - new)))
                if d_chunk > delta:
                    delta = d_chunk
            V[c[li]] = vals

        # V may have been written for terminal states by succ lookups only;
        # re-pin them (they are absorbing)
        V[is_term] = term_flat[is_term]

        if verbose and (sweep % log_every == 0 or sweep < 3):
            und = int(np.count_nonzero((V > 1e-3) & (V < 1 - 1e-3)))
            print(f"  sweep {sweep:4d}  delta={delta:.3e}  nonsaddle={n_bad_tot:7d}"
                  f"  undecided={und:9d}")

        if sweep + 1 >= min_sweeps and delta < tol:
            quiet += 1
            if quiet >= patience:
                break
        else:
            quiet = 0

    return V.reshape(shape), {
        "sweeps": sweep + 1, "delta": delta, "worst_gap": worst_gap,
        "size": size, "chunks": len(starts),
    }


# ==========================================================================
# Parallel value iteration
# ==========================================================================
# Chunks are independent within a sweep (pure Jacobi: every worker reads the
# frozen previous iterate and writes its own slice of the next one), so the
# sweep parallelises exactly across processes with no synchronisation beyond
# the barrier at the end of the sweep.  Double buffering removes the race that
# in-place Gauss-Seidel would create across process boundaries.
#
# The successor function and the terminal mask are inherited by the workers
# through fork() rather than pickled, so nothing large crosses a pipe.

import multiprocessing as mp
from multiprocessing import shared_memory

_P = {}          # worker-local state
_M = {}          # model, set in the parent before the pool is forked


def _worker_init(shm_read, shm_write, size, shape, nu, nv, chunk,
                 fp_iters, fp_tol):
    # keep the SharedMemory objects alive: if they are collected the mmap is
    # closed underneath the numpy view, which reads freed memory
    sr = shared_memory.SharedMemory(name=shm_read)
    sw = shared_memory.SharedMemory(name=shm_write)
    _P["_shm"] = (sr, sw)
    _P["r"] = np.ndarray((size,), dtype=np.float64, buffer=sr.buf)
    _P["w"] = np.ndarray((size,), dtype=np.float64, buffer=sw.buf)
    _P.update(size=size, shape=shape, nu=nu, nv=nv, chunk=chunk,
              fp_iters=fp_iters, fp_tol=fp_tol,
              G=np.empty((nu, nv, chunk)), C=np.empty(chunk, dtype=np.int64))


def _worker_chunk(s0):
    P = _P
    size, shape = P["size"], P["shape"]
    nu, nv, chunk = P["nu"], P["nv"], P["chunk"]
    Vr, Vw = P["r"], P["w"]
    term_flat = _M["term_flat"]
    is_term = _M["is_term"]
    succ_fn = _M["succ"]

    n = min(chunk, size - s0)
    c = P["C"][:n]
    c[:] = np.arange(s0, s0 + n)
    coords = tuple(x.astype(np.int16) for x in np.unravel_index(c, shape))
    g = P["G"][:, :, :n]

    live = ~is_term[c]
    tm = is_term[c]
    if tm.any() and Vw is not Vr:
        Vw[c[tm]] = term_flat[c[tm]]
    if not live.any():
        return 0.0, 0, 0.0

    for i in range(nu):
        for j in range(nv):
            idxs, ws = succ_fn(coords, i, j)
            acc = ws[0] * Vr[idxs[0]]
            for k in (1, 2, 3):
                acc += ws[k] * Vr[idxs[k]]
            g[i, j] = acc

    li = np.nonzero(live)[0]
    sub = np.ascontiguousarray(g[:, :, li])
    vals, n_bad, gap = stage_values(sub, nu, nv, P["fp_iters"], P["fp_tol"])
    new = Vr[c[li]]
    delta = float(np.max(np.abs(vals - new))) if li.size else 0.0
    Vw[c[li]] = vals
    return delta, n_bad, gap


def solve_game_np_mp(shape, n_actions, succ_fn, term_val, chunk=400_000,
                     nproc=2, max_sweeps=400, min_sweeps=25, tol=1e-5,
                     patience=2, fp_iters=150, fp_tol=1e-3,
                     verbose=True, log_every=5, inplace=False, V0=None):
    """Multiprocess driver around the same operator as `solve_game_np`.

    With `inplace=True` the workers read and write ONE shared buffer, which
    turns the sweep into asynchronous Gauss-Seidel.  It converges in far fewer
    sweeps than the double-buffered Jacobi version: information propagates
    within a sweep instead of only between sweeps.  Aligned 8-byte stores are
    atomic on x86-64, so a concurrent reader sees either the old or the new
    value, never a torn one; the schedule is non-deterministic but the fixed
    point is the same."""
    nu, nv = n_actions
    size = int(np.prod(shape))
    chunk = max(1, min(chunk, size))
    starts = list(range(0, size, chunk))

    term_flat = np.ascontiguousarray(term_val.ravel())
    is_term = ~np.isnan(term_flat)
    _M["term_flat"] = term_flat
    _M["is_term"] = is_term
    _M["succ"] = succ_fn            # inherited by fork, never pickled

    shm_r = shared_memory.SharedMemory(create=True, size=size * 8)
    if inplace:
        shm_w = shm_r                      # single buffer, read+write
    else:
        shm_w = shared_memory.SharedMemory(create=True, size=size * 8)
    Vr = np.ndarray((size,), dtype=np.float64, buffer=shm_r.buf)
    Vw = np.ndarray((size,), dtype=np.float64, buffer=shm_w.buf)
    if V0 is None:
        Vr[:] = 0.5
    else:
        Vr[:] = np.asarray(V0, dtype=np.float64).ravel()
    Vr[is_term] = term_flat[is_term]
    Vw[:] = Vr

    ctx = mp.get_context("fork")
    quiet, delta, worst_gap, sweep = 0, 0.0, 0.0, 0
    pool = ctx.Pool(nproc, _worker_init,
                    (shm_r.name, shm_w.name, size, shape, nu, nv, chunk,
                     fp_iters, fp_tol))
    try:
        for sweep in range(max_sweeps):
            res = pool.map(_worker_chunk, starts, chunksize=1)
            delta = max(r[0] for r in res)
            n_bad = sum(r[1] for r in res)
            worst_gap = max(r[2] for r in res)
            if not inplace:
                Vr[:] = Vw                  # swap by copy; ~30 ms at 10 M
            if verbose and (sweep % log_every == 0 or sweep < 3):
                und = int(np.count_nonzero((Vr > 1e-3) & (Vr < 1 - 1e-3)))
                print(f"  sweep {sweep:4d}  delta={delta:.3e}  "
                      f"nonsaddle={n_bad:8d}  undecided={und:9d}")
            if sweep + 1 >= min_sweeps and delta < tol:
                quiet += 1
                if quiet >= patience:
                    break
            else:
                quiet = 0
    finally:
        pool.close()
        pool.join()
        out = Vr.copy()
        shm_r.close(); shm_w.close()
        shm_r.unlink()
        if not inplace:
            shm_w.unlink()

    return out.reshape(shape), {
        "sweeps": sweep + 1, "delta": delta, "worst_gap": worst_gap,
        "size": size, "chunks": len(starts), "nproc": nproc,
    }


def solve_game_np_ordered(shape, n_actions, succ_fn, term_val,
                          max_sweeps=400, min_sweeps=25, tol=1e-5, patience=2,
                          fp_iters=150, fp_tol=1e-3, verbose=True, log_every=5,
                          V0=None, cd_axis=4, reverse=False):
    """Gauss-Seidel value iteration, sweeping the cooldown axis in order.

    Plain Jacobi has to carry information one step per sweep, and the state
    graph here is ~90 steps deep along the ability cooldown, so it needs ~250
    sweeps.  The direction information travels is known, though: V(c) depends
    on V(c-1), because the cooldown ticks down every step.  So iterating the
    cooldown slices in increasing order and writing in place propagates the
    whole chain inside a single sweep.

    Blocks are one cooldown slice (all other axes), which keeps them large
    enough that numpy overhead does not dominate.
    """
    nu, nv = n_actions
    ndim = len(shape)
    size = int(np.prod(shape))
    strides = tuple(int(np.prod(shape[k + 1:])) for k in range(ndim))

    V = np.full(size, 0.5, dtype=np.float64) if V0 is None else V0.ravel().copy()
    term_flat = term_val.ravel()
    is_term = ~np.isnan(term_flat)
    V[is_term] = term_flat[is_term]

    # coordinates of the non-swept axes, built once
    keep = [k for k in range(ndim) if k != cd_axis]
    rest_shape = tuple(shape[k] for k in keep)
    if keep:
        rest = [x.ravel().astype(np.int16) for x in np.indices(rest_shape)]
        nrest = rest[0].size
    else:                      # 1-D space: the swept axis is the only axis
        rest, nrest = [], 1
    NC = shape[cd_axis]
    cd_const = np.empty(nrest, dtype=np.int16)

    G = np.empty((nu, nv, nrest))
    quiet = 0
    delta = 0.0
    worst_gap = 0.0
    sweep = 0
    for sweep in range(max_sweeps):
        delta = 0.0
        n_bad_tot = 0
        order = range(NC - 1, -1, -1) if reverse else range(NC)
        for cval in order:
            cd_const[:] = cval
            parts = []
            ri = 0
            for k in range(ndim):
                if k == cd_axis:
                    parts.append(cd_const)
                else:
                    parts.append(rest[ri]); ri += 1
            coords = tuple(parts)
            flat = np.zeros(nrest, dtype=np.int64)
            ri = 0
            for k in range(ndim):
                if k == cd_axis:
                    flat += cd_const.astype(np.int64) * strides[k]
                else:
                    flat += rest[ri].astype(np.int64) * strides[k]; ri += 1
            live = ~is_term[flat]
            if not live.any():
                continue
            for i in range(nu):
                for j in range(nv):
                    idxs, ws = succ_fn(coords, i, j)
                    acc = ws[0] * V[idxs[0]]
                    for k in (1, 2, 3):
                        acc += ws[k] * V[idxs[k]]
                    G[i, j] = acc
            li = np.nonzero(live)[0]
            sub = np.ascontiguousarray(G[:, :, li])
            vals, n_bad, gap = stage_values(sub, nu, nv, fp_iters, fp_tol)
            n_bad_tot += n_bad
            if gap > worst_gap:
                worst_gap = gap
            old = V[flat[li]]
            if li.size:
                d_chunk = float(np.max(np.abs(vals - old)))
                if d_chunk > delta:
                    delta = d_chunk
            V[flat[li]] = vals          # in place: later slices see this

        V[is_term] = term_flat[is_term]
        if verbose and (sweep % log_every == 0 or sweep < 3):
            und = int(np.count_nonzero((V > 1e-3) & (V < 1 - 1e-3)))
            print(f"  sweep {sweep:4d}  delta={delta:.3e}  nonsaddle={n_bad_tot:7d}"
                  f"  undecided={und:9d}", flush=True)
        if sweep + 1 >= min_sweeps and delta < tol:
            quiet += 1
            if quiet >= patience:
                break
        else:
            quiet = 0

    return V.reshape(shape), {"sweeps": sweep + 1, "delta": delta,
                              "worst_gap": worst_gap, "size": size}
