"""Runner: build the reduced duel and solve it.  Usage:
     python3 -m experiments.run_full --quick
     python3 -m experiments.run_full --blink
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from .full_model import (BASE, build_shape, terminal_array, make_successor,
                         heuristic_value)
from solvers.npdp import solve_game_np_mp, solve_game_np, solve_game_np_ordered


def orb_stats(split, instances):
    """Derived level-15 stats for an Invoker orb distribution."""
    Q, W, E = split
    nQ, nW, nE = instances
    STR, AGI, INT = 54 + Q, 42 + W, 78 + E
    from dota2sim.constants import total_attack_speed, attacks_per_second
    MS = 280.0 * (1 + 0.006 * W * nW)
    AS = total_attack_speed(AGI + W * nW)
    dmg = 22 + INT + 2 * E * nE
    return dict(STR=STR, AGI=AGI, INT=INT, HP=120 + 22 * STR, MS=MS,
                dmg=dmg, aps=attacks_per_second(AS),
                regen=0.25 + 0.09 * STR + Q * nQ)


def combo_damage(split):
    """Invoker's kit as (physical, pure, magical) plus a weighted cooldown."""
    Q, W, E = split
    snap_n = (2.6 + 0.4 * Q) / (0.83 - 0.03 * Q)          # number of procs
    parts = [
        ((20 + 8 * Q) * snap_n, "phys", 19.0),            # Cold Snap
        (125 + 50 * E,          "pure", 23.0),            # Sun Strike
        ((30 + 20 * E) * 3,     "mag",  50.0),            # Chaos Meteor
        (0.6 * (25 + 75 * W),   "mag",  27.0),            # E.M.P.
        (30 + 40 * E,           "mag",  36.0),            # Deafening Blast
        (50 + 45 * W,           "mag",  27.0),            # Tornado
    ]
    phys = sum(v for v, t, _ in parts if t == "phys")
    pure = sum(v for v, t, _ in parts if t == "pure")
    mag  = sum(v for v, t, _ in parts if t == "mag")
    dps  = sum(v / cd for v, _, cd in parts)
    cycle = (phys + pure + mag) / dps
    return phys, pure, mag, cycle


def make_cfg(d_max=12.0, dist_q=0.5, hpq=200.0, c2_cap=64,
             split=(8, 1, 8), instances=(0, 0, 3),
             p_hp=3200.0, i_hp=None, hook_cd_steps=None,
             burst_cd_steps=None,
             # 1 m buckets up to d_max.  Room only matters in the last
             # `d_max` metres before the wall: with more space behind him than
             # the largest representable distance the target is never pinned,
             # so the top bucket means "unpinned".
             room=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
             cd_cap=40, blink=False, slow_item=0.0, p_ms=None, i_ms=None):
    P = dict(BASE)
    phys, pure, mag, cycle = combo_damage(split)
    P["p_hp_max"], P["i_hp_max"] = p_hp, i_hp
    P["p_ms"] = p_ms if p_ms is not None else P["p_ms"]
    orb = orb_stats(split, instances)
    P["i_ms"] = i_ms if i_ms is not None else orb["MS"]
    P["i_dps"] = orb["dmg"] * orb["aps"]
    P["i_armor"] = (42 + split[1]) / 6.0
    P["i_hp_max"] = orb["HP"]
    if i_hp is None:
        i_hp = orb["HP"]
    P["i_damage"] = 410.0
    P["i_range"] = 950.0
    P["i_speed"] = 1000.0
    P["i_radius"] = 675.0
    P["i_delay"] = 2.9
    # Cold Snap, Quas 8: 20 + 8*Quas = 84 per proc, procs every
    # 0.83 - 0.03*Quas = 0.59 s over 5.8 s -> ~9 procs ~ 756 physical
    P["c_phys"], P["c_pure"], P["c_magic"] = phys, pure, mag
    P["combo_range"] = 1000.0
    P["p_atk_range"] = 175.0 + 2 * 24.0      # range + both bound radii
    P["i_atk_range"] = 600.0 + 2 * 24.0
    P["blink"] = blink
    P["blink_range"] = 1200.0
    if slow_item:
        P["i_ms"] = P["i_ms"] * (1.0 - slow_item)

    cd_cap = int(cd_cap)
    c2_cap = int(c2_cap)
    STEP = 0.3

    def steps(sec):
        return int(round(sec / STEP))

    # --- Invoker's kit: per ability, each with its own real cooldown -------
    # At 8/1/8 with three Exort instances.  Numbers from the 7.41f datafeed
    # sweep, see data/invoker_7.41f.md
    Q, W, E = split
    snap_n = (2.6 + 0.4 * Q) / (0.83 - 0.03 * Q)        # number of Cold Snap procs
    INV_SPELLS = [
        ("Cold Snap",       (20 + 8 * Q) * snap_n, "phys",  19.0),
        ("Sun Strike",      125 + 50 * E,          "pure",  23.0),
        ("E.M.P.",          0.6 * (25 + 75 * W),   "magic", 27.0),
        ("Tornado",         50 + 45 * W,           "magic", 27.0),
        ("Deafening Blast", 30 + 40 * E,           "magic", 36.0),
        ("Chaos Meteor",    (30 + 20 * E) * 3,     "magic", 50.0),
    ]
    PUD_SPELLS = [("Meat Hook", 360.0, "pure", 12.0),
                  ("Dismember", 140.2 * 2.75, "magic", 25.0)]

    c2_max = max(steps(cd) for _, _, _, cd in INV_SPELLS)     # Meteor, 50 s
    c1_max = max(steps(cd) for _, _, _, cd in PUD_SPELLS)     # Dismember, 25 s

    # Damage ramp: what one cast delivers, given how long since the last one.
    # This is what lets a single axis carry six independent cooldowns.
    ramp2 = {"phys": np.zeros(c2_max + 1), "pure": np.zeros(c2_max + 1),
             "magic": np.zeros(c2_max + 1)}
    for _, dmg, typ, cd in INV_SPELLS:
        ramp2[typ][steps(cd):] += dmg
    ramp1_pure = np.zeros(c1_max + 1)
    ramp1_magic = np.zeros(c1_max + 1)
    for _, dmg, typ, cd in PUD_SPELLS:
        (ramp1_pure if typ == "pure" else ramp1_magic)[steps(cd):] += dmg

    snap_cd_steps = steps(19.0)
    snap_dur_steps = min(steps(2.6 + 0.4 * Q), snap_cd_steps)
    hook_cd_steps = steps(12.0)
    dis_cd_steps = steps(25.0)

    cfg = dict(
        ramp1_pure=ramp1_pure, ramp1_magic=ramp1_magic,
        ramp2_phys=ramp2["phys"], ramp2_pure=ramp2["pure"],
        ramp2_magic=ramp2["magic"],
        d_max=d_max, dist_q=dist_q, hpq=hpq, hp1=p_hp, hp2=i_hp,
        room=tuple(room), room_step=(room[1] - room[0]),
        c1_steps=c1_max, c2_steps=c2_max,
        hook_cd_steps=hook_cd_steps, dis_cd_steps=dis_cd_steps,
        blink_cd_steps=max(steps(15.0), hook_cd_steps),
        snap_cd_steps=snap_cd_steps, snap_dur_steps=snap_dur_steps,
        params=P,
    )
    return cfg


def build_actions(cfg):
    P = cfg["params"]
    k1 = max(1, int(round(P["p_ms"] / 52.4934 * 0.3 / cfg["dist_q"])))
    k2 = max(1, int(round(P["i_ms"] / 52.4934 * 0.3 / cfg["dist_q"])))
    # Pudge: close / hold / back off, crossed with his combat options.
    # Closing is negative movement in d.
    mv1 = sorted({-k1, 0, k1})
    c1 = ["none", "hook"]
    if P.get("blink"):
        c1.append("blink")
    A1 = [(m, a) for m in mv1 for a in c1]
    mv2 = sorted({-k2, 0, k2})
    A2 = [(m, a) for m in mv2 for a in ["none", "combo"]]
    cfg["acts1"], cfg["acts2"] = A1, A2
    cfg["k1"], cfg["k2"] = k1, k2
    return cfg


def run(args):
    cfg = build_actions(make_cfg(d_max=args.dmax, hpq=args.hpq,
                                 cd_cap=args.cdcap, c2_cap=args.cdcap2, blink=args.blink,
                                 slow_item=args.slow, split=args.split,
                                 instances=args.instances,
                                 p_hp=args.p_hp, i_hp=args.i_hp,
                                 **({"room": args.room} if args.room else {})))
    shape, meta = build_shape(cfg)
    print(f"state space : {meta['size']:,}  shape={shape}")
    print(f"actions     : {len(cfg['acts1'])} x {len(cfg['acts2'])} "
          f"= {len(cfg['acts1'])*len(cfg['acts2'])} pairs")
    print(f"reach       : pudge {cfg['k1']} cells, invoker {cfg['k2']} cells")
    tv = terminal_array(shape, meta, cfg)
    print(f"terminal    : {int((~np.isnan(tv)).sum()):,}")
    succ = make_successor(cfg, meta)
    V0 = heuristic_value(cfg, shape, meta) if args.heuristic else None
    if V0 is not None:
        print("using static-race heuristic initialization")

    t0 = time.time()
    # The DP tolerance must sit ABOVE the fictitious-play bracket tolerance.
    # FP resolves the mixed states only to within fp_tol (~1e-3), so a residual
    # target of 1e-5 can never be met: the iteration hits a noise floor and
    # burns the whole sweep budget achieving nothing.
    common = dict(max_sweeps=args.sweeps, min_sweeps=args.min_sweeps,
                  tol=args.tol, fp_iters=args.fp_iters, fp_tol=1e-3,
                  verbose=True, log_every=1)
    if args.ordered:
        # sweeps the cooldown axis in order with in-place writes; one process,
        # but far fewer sweeps (see solvers.npdp.solve_game_np_ordered)
        V, info = solve_game_np_ordered(
            shape, (len(cfg["acts1"]), len(cfg["acts2"])), succ, tv,
            cd_axis=4, reverse=True, V0=V0, **common)
    else:
        V, info = solve_game_np_mp(
            shape, (len(cfg["acts1"]), len(cfg["acts2"])), succ, tv,
            chunk=args.chunk, V0=V0, **common)
    dt = time.time() - t0
    print(f"\nsolved in {dt:.1f}s  {info['sweeps']} sweeps  "
          f"delta={info['delta']:.2e}  worst_gap={info['worst_gap']:.2e}")
    np.save(f"results/V_{args.tag}.npy", V.reshape(shape))
    return cfg, shape, meta, V, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmax", type=float, default=12.0)
    ap.add_argument("--hpq", type=float, default=200.0)
    ap.add_argument("--cdcap", type=int, default=40)
    ap.add_argument("--cdcap2", dest="cdcap2", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=120)
    ap.add_argument("--min-sweeps", dest="min_sweeps", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=300_000)
    ap.add_argument("--fp-iters", dest="fp_iters", type=int, default=60)
    ap.add_argument("--blink", action="store_true")
    ap.add_argument("--slow", type=float, default=0.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--room", default="")
    ap.add_argument("--heuristic", action="store_true")
    ap.add_argument("--ordered", action="store_true")
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--p-hp", dest="p_hp", type=float, default=3200.0)
    ap.add_argument("--i-hp", dest="i_hp", type=float, default=None)
    ap.add_argument("--split", default="8,1,8")
    ap.add_argument("--instances", default="")
    a = ap.parse_args()
    if a.quick:
        # only shrink what the user did not set explicitly
        import sys as _s
        if "--dmax" not in _s.argv: a.dmax = 8.0
        if "--hpq" not in _s.argv: a.hpq = 400.0
        if "--cdcap" not in _s.argv: a.cdcap = 20
    a.room = tuple(float(x) for x in a.room.split(",")) if a.room else None
    a.split = tuple(int(x) for x in a.split.split(","))
    if a.instances:
        a.instances = tuple(int(x) for x in a.instances.split(","))
    else:
        i = max(range(3), key=lambda k: a.split[k])
        a.instances = tuple(3 if k == i else 0 for k in range(3))
    run(a)
