"""Does the DP actually want to cast everything at once?

My per-ability compression assumed optimal play casts every ready spell
together, which let one cooldown axis carry six independent cooldowns.  That is
a claim about strategy and must be an OUTPUT of the DP, not an input.

This is the smallest model where the question is answerable: Invoker gets two
spells with genuinely independent cooldowns, and the action set lets him cast
either, both, or neither.  Cold Snap is unit-target and always lands; Sun
Strike is ground-targeted with a 1.7 s delay, so it lands only on a stunned
target.  Casting Sun Strike alone should therefore be wasted, and the DP has to
discover the sequencing on its own.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import numpy as np

from solvers.npdp import solve_game_np_ordered
from dota2sim.constants import UNITS_PER_METER, TICK, armor_multiplier

SNAP_CD, SNAP_DUR = 63, 19      # 19 s cooldown, 5.7 s stun window
SUN_CD = 77                     # 23 s cooldown
SNAP_DMG = 825.0                # physical
SUN_DMG = 525.0                 # pure
ND, NH1, NH2, NR = 7, 3, 3, 2   # d 0..3 m, HP coarse, room coarse
HPQ = 800.0
SHAPE = (ND, NH1, NH2, SNAP_CD + 1, SUN_CD + 1, NR)
STRIDES = tuple(int(np.prod(SHAPE[k+1:])) for k in range(len(SHAPE)))

P_ARMOR, P_MR = 5.4333, 0.25
P_DPS = 88.0
P_ATK_RANGE_U = 175 + 48
I_ARMOR = 7.1667

A1 = [("none",), ("initiate",)]                  # Pudge
A2 = [(), ("snap",), ("sun",), ("snap", "sun")]  # Invoker


def make_succ():
    s_d, s_h1, s_h2, s_cs, s_cu, s_r = STRIDES

    def succ(coords, i, j):
        d, h1, h2, cs, cu, r = [c.astype(np.int64) for c in coords]
        a1 = A1[i][0]
        cast = A2[j]

        frozen = cs < SNAP_DUR                 # stunned by a recent Cold Snap
        cs_n = np.minimum(cs + 1, SNAP_CD)
        cu_n = np.minimum(cu + 1, SUN_CD)
        d_new = d

        loss1 = np.zeros(d.shape)              # damage to Pudge
        loss2 = np.zeros(d.shape)              # damage to Invoker
        d_m = d * 1.0                          # dist_q = 1 m

        # Pudge initiates whenever he is close enough
        if a1 == "initiate":
            hits = (~frozen) & (d_m * UNITS_PER_METER <= 400.0)
            loss2 = loss2 + np.where(hits, 746.0 * armor_multiplier(I_ARMOR), 0.0)
            d_new = np.where(hits, 1, d)

        # Invoker auto-attack
        loss1 = loss1 + np.where(d_m * UNITS_PER_METER <= 648.0,
                                 30.0 * TICK * armor_multiplier(P_ARMOR), 0.0)

        snap_ready = cs >= SNAP_CD
        if "snap" in cast:
            loss1 = loss1 + np.where(snap_ready, SNAP_DMG * armor_multiplier(P_ARMOR), 0.0)
            cs_n = np.where(snap_ready, 0, cs_n)
        if "sun" in cast:
            ready = cu >= SUN_CD
            # Sun Strike has a 1.7 s delay before it lands, so it connects if
            # the target is ALREADY stunned, or if a Cold Snap cast in the same
            # action stuns him first.  Casting it on a free target wastes it.
            stunned = frozen | ("snap" in cast and snap_ready)
            lands = ready & stunned
            loss1 = loss1 + np.where(lands, SUN_DMG, 0.0)
            cu_n = np.where(ready, 0, cu_n)     # the cooldown is spent regardless

        f1 = np.clip(h1 - loss1 / HPQ, 0.0, NH1 - 1.0)
        f2 = np.clip(h2 - loss2 / HPQ, 0.0, NH2 - 1.0)
        lo1 = np.floor(f1).astype(np.int64); w1 = f1 - lo1
        lo2 = np.floor(f2).astype(np.int64); w2 = f2 - lo2
        hi1 = np.minimum(lo1 + 1, NH1 - 1); hi2 = np.minimum(lo2 + 1, NH2 - 1)
        base = d_new * s_d + cs_n * s_cs + cu_n * s_cu + r * s_r
        idx = ((base + lo1 * s_h1 + lo2 * s_h2).astype(np.int64),
               (base + hi1 * s_h1 + lo2 * s_h2).astype(np.int64),
               (base + lo1 * s_h1 + hi2 * s_h2).astype(np.int64),
               (base + hi1 * s_h1 + hi2 * s_h2).astype(np.int64))
        return idx, ((1-w1)*(1-w2), w1*(1-w2), (1-w1)*w2, w1*w2)
    return succ


def main():
    tv = np.full(SHAPE, np.nan)
    tv[:, 0, :, :, :, :] = 0.0        # Pudge dead
    tv[:, :, 0, :, :, :] = 1.0        # Invoker dead
    V, info = solve_game_np_ordered(SHAPE, (len(A1), len(A2)), make_succ(), tv,
                                    cd_axis=4, reverse=True, max_sweeps=200,
                                    min_sweeps=30, tol=2e-3, verbose=False)
    np.save("results/V_castorder.npy", V)
    print(f"сошлось: {info['sweeps']} свипов, delta={info['delta']:.2e}, "
          f"состояний {info['size']:,}")

    # what does the DP choose when both spells are ready?
    d, h1, h2, r = 2, NH1-1, NH2-1, 0
    print("\nДП в состоянии «оба готовы» (cs=63, cu=77), полное HP, d=2м:")
    for i in range(len(A1)):
        row = []
        for j in range(len(A2)):
            s = (d, h1, h2, SNAP_CD, SUN_CD, r)
            v = 0.0
            idxs, ws = make_succ()(tuple(np.array([x], dtype=np.int16) for x in s), i, j)
            for k in range(4):
                v += ws[k][0] * V.ravel()[idxs[k][0]]
            row.append(v)
        print(f"   Pudge={A1[i][0]:9s} " + "  ".join(
            f"{str(A2[j]) or '()':14s}={row[j]:.4f}" for j in range(len(A2))))

    print("\nДП когда Cold Snap НЕ готов, а Sun Strike готов (cs=10, cu=77):")
    s = (d, h1, h2, 10, SUN_CD, r)
    for j in range(len(A2)):
        idxs, ws = make_succ()(tuple(np.array([x], dtype=np.int16) for x in s), 0, j)
        v = sum(ws[k][0] * V.ravel()[idxs[k][0]] for k in range(4))
        print(f"   Invoker {str(A2[j]) or '()':14s} -> V={v:.4f}")


if __name__ == "__main__":
    main()
