"""
Evaluate the trained value function inside the search.

The point of training was to stop the answer depending on a hand-written
rollout policy.  This compares the two leaves directly on the same positions:
the hand policy's rollout, and the learned function.  If they disagree
systematically, the function is wrong and the disagreement localises where.

Usage:
    python3 -m solvers.evaluate --vf results/valuefn.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from dota2sim.build import make_duel
from dota2sim.constants import UNITS_PER_METER
from .valuefn import ValueFunction
from .search import (make_leaf, value, rollout, actions_for, PUDGE_KIT,
                     INVOKER_KIT)

SCENARIOS = [
    ("оба полные, 2 м", 2.0, 1.0, 1.0),
    ("оба полные, 4 м", 4.0, 1.0, 1.0),
    ("оба полные, 8 м", 8.0, 1.0, 1.0),
    ("оба полные, 15 м", 15.0, 1.0, 1.0),
    ("Pudge 50%, Invoker 100%, 4 м", 4.0, 0.5, 1.0),
    ("Pudge 100%, Invoker 50%, 4 м", 4.0, 1.0, 0.5),
    ("Pudge 25%, Invoker 100%, 8 м", 8.0, 0.25, 1.0),
    ("Pudge 100%, Invoker 25%, 8 м", 8.0, 1.0, 0.25),
]


def build(dist_m, php, ihp):
    d = make_duel(pudge_pos=(10.0, 40.0), invoker_pos=(10.0 + dist_m, 40.0))
    d.u["pudge"].hp = d.u["pudge"].max_hp * php
    d.u["invoker"].hp = d.u["invoker"].max_hp * ihp
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vf", default="results/valuefn.npz")
    ap.add_argument("--depth", type=int, default=1)
    a = ap.parse_args()

    vf = ValueFunction.load(a.vf)
    hand = make_leaf()

    print(f"{'сценарий':32s} {'лист-прогон':>12s} {'лист-функция':>13s} "
          f"{'поиск(VF)':>10s}")
    for name, dist, php, ihp in SCENARIOS:
        d = build(dist, php, ihp)
        v_hand = hand(d.clone())
        v_vf = vf(d.clone())
        v_search = value(d.clone(), depth=a.depth, leaf=vf)
        print(f"{name:32s} {v_hand:12.2f} {v_vf:13.2f} {v_search:10.2f}")

    print("\nоптимальные действия по поиску с обученной функцией:")
    for name, dist, php, ihp in SCENARIOS[:6]:
        d = build(dist, php, ihp)
        st = {}
        v = value(d.clone(), depth=a.depth, leaf=vf, stats=st)
        Ap, Ai = st["root_actions"]
        p, q = np.array(st["root_p"]), np.array(st["root_q"])
        bp = [str(Ap[i]) for i in np.argsort(p)[::-1][:2] if p[i] > 1e-3]
        bi = [str(Ai[j]) for j in np.argsort(q)[::-1][:2] if q[j] > 1e-3]
        print(f"  {name:30s} V={v:.2f}")
        print(f"      Pudge  : {', '.join(bp)}")
        print(f"      Invoker: {', '.join(bi)}")


if __name__ == "__main__":
    main()
