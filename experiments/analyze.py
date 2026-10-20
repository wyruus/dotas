"""Extract the optimal policy and the structure of optimal play from V."""

from __future__ import annotations

import argparse
import numpy as np

from dota2sim.constants import UNITS_PER_METER
from solvers.matrixgame import solve as solve_matrix
from .full_model import build_shape, terminal_array, make_successor
from .run_full import make_cfg, build_actions


def policy_at(V, cfg, meta, succ, state, nu, nv):
    """Stage game at one state: returns (value, rows, cols, matrix)."""
    shape = V.shape
    coords = tuple(np.array([idx], dtype=np.int16) for idx in state)
    Vf = V.ravel()
    G = np.empty((nu, nv))
    for i in range(nu):
        for j in range(nv):
            idxs, ws = succ(coords, i, j)
            G[i, j] = sum(float(ws[k][0]) * float(Vf[idxs[k][0]]) for k in range(4))
    val, p, q, info = solve_matrix(G.ravel().tolist(), nu, nv)
    return val, p, q, G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="quick")
    ap.add_argument("--dmax", type=float, default=8.0)
    ap.add_argument("--hpq", type=float, default=400.0)
    ap.add_argument("--cdcap", type=int, default=20)
    ap.add_argument("--blink", action="store_true")
    ap.add_argument("--slow", type=float, default=0.0)
    ap.add_argument("--split", default="8,1,8")
    ap.add_argument("--instances", default="0,0,3")
    ap.add_argument("--cdcap2", dest="cdcap2", type=int, default=88)
    ap.add_argument("--room", default="0,2,4,6,8")
    ap.add_argument("--p-hp", dest="p_hp", type=float, default=3200.0)
    ap.add_argument("--i-hp", dest="i_hp", type=float, default=None)
    a = ap.parse_args()

    cfg = build_actions(make_cfg(
        d_max=a.dmax, hpq=a.hpq, cd_cap=a.cdcap, c2_cap=a.cdcap2,
        room=tuple(float(x) for x in a.room.split(",")) if a.room else None,
        split=tuple(int(x) for x in a.split.split(",")),
        instances=tuple(int(x) for x in a.instances.split(",")),
        blink=a.blink, slow_item=a.slow,
        p_hp=a.p_hp, i_hp=a.i_hp))
    shape, meta = build_shape(cfg)
    V = np.load(f"results/V_{a.tag}.npy")
    assert V.shape == shape, (V.shape, shape)
    succ = make_successor(cfg, meta)
    nu, nv = len(cfg["acts1"]), len(cfg["acts2"])
    dq = cfg["dist_q"]
    NH1, NH2, NC1, NC2, NR = (meta["NH1"], meta["NH2"], meta["NC1"],
                              meta["NC2"], meta["NR"])

    print(f"V shape {V.shape}   actions {nu}x{nv}   dist_q={dq}")
    print(f"undecided: {int(np.count_nonzero((V>1e-3)&(V<1-1e-3))):,}"
          f" / {V.size:,}")

    # ---- value vs distance, full HP, everything ready, mid room ----
    h1f, h2f = NH1 - 1, NH2 - 1
    print("\n  d(m)   V(pudge)  [full HP, both ready, room=mid]")
    for d in range(meta["ND"]):
        v = V[d, h1f, h2f, 0, 0, NR // 2]
        bar = "#" * int(round(v * 40))
        print(f"  {d*dq:5.1f}   {v:6.3f}   {bar}")

    # ---- critical distance: furthest d at which Pudge still wins ----
    print("\n  critical distance (max d with V>0.5), by Invoker HP:")
    for h2 in range(1, NH2, max(1, NH2 // 5)):
        row = V[:, h1f, h2, 0, 0, NR // 2]
        win = np.nonzero(row > 0.5)[0]
        cd = win.max() * dq if win.size else float("nan")
        print(f"    Invoker HP {h2*cfg['hpq']:6.0f}: {cd:5.1f} m"
              f"   (V at 0 m = {row[0]:.3f})")

    # ---- does the wall matter? ----
    print("\n  effect of Invoker's dodging room (d = 12 m):")
    dd = min(int(round(0.8 * cfg["d_max"] / dq)), meta["ND"] - 1)
    for r in range(NR):
        print(f"    room={cfg['room'][r]:5.1f} m -> V={V[dd,h1f,h2f,0,0,r]:.3f}")

    # ---- policy at a representative engagement state ----
    print("\n  optimal action at d=3 m, full HP, both ready, room=mid")
    st = (int(round(3.0 / dq)), h1f, h2f, 0, 0, NR // 2)
    val, p, q, G = policy_at(V, cfg, meta, succ, st, nu, nv)
    print(f"    value = {val:.3f}")
    order1 = np.argsort(p)[::-1][:4]
    order2 = np.argsort(q)[::-1][:4]
    print("    Pudge :", ", ".join(
        f"{cfg['acts1'][i]}:{p[i]:.2f}" for i in order1 if p[i] > 1e-3))
    print("    Invoker:", ", ".join(
        f"{cfg['acts2'][j]}:{q[j]:.2f}" for j in order2 if q[j] > 1e-3))


if __name__ == "__main__":
    main()
