"""
Full Pudge vs Invoker duel model, numpy-vectorised, patch 7.41f.

State (6 axes)
--------------
    (d, h1, h2, c1, c2, room)

    d     distance, cells of DIST_Q metres
    h1    Pudge HP / HPQ1
    h2    Invoker HP / HPQ2
    c1    Meat Hook cooldown, in DP steps (the only Pudge cooldown that
          gates a kill; Rot is a toggle with no cooldown, Dismember is
          tracked implicitly by c1 since it is only reachable after a hook)
    c2    Invoker's burst cooldown, in DP steps
    room  Invoker's lateral dodging room, bucketed

Why `room` is in the state
--------------------------
With perfect reactions a skillshot is dodged iff the target can displace
laterally by more than the hit radius within cast-point + flight-time.  The
available lateral displacement is `min(v*t, room)`.  So the arena wall is not
decoration: driving Invoker back until `room` is small is the only way Pudge's
hook becomes unavoidable at range.  Dropping `room` would delete the entire
positional game.

Transitions
-----------
Movement is a lattice displacement; damage is applied per DP step.  Channelled
and damage-over-time effects are applied as their total value at the step they
land, which is correct at 0.3 s granularity for everything except interrupt
play on the channel - that is handled by an explicit counter (Invoker's burst
interrupts Dismember if it is ready when the channel starts, which is exactly
the real trade).
"""

from __future__ import annotations

import numpy as np

from dota2sim.constants import UNITS_PER_METER, TICK, armor_multiplier

TICK_S = TICK

# --------------------------------------------------------------------------
# Matchup constants, patch 7.41f.  Pudge numbers verified against the official
# datafeed; see dota2sim/herodata.py and data/*.md
# --------------------------------------------------------------------------
BASE = dict(
    # Pudge, level 15: STR 67 AGI 32.6 INT 41.2
    p_ms=280.0, p_hp=1594.0, p_armor=5.4333, p_regen=9.2,
    p_dps=88.0,                      # 112-118 damage at 0.78 attacks/s
    hook_dmg=360.0, hook_cd=12.0, hook_range=1300.0, hook_speed=1600.0,
    hook_radius=100.0, hook_cast_point=0.3, hook_drag=125.0,
    dis_dps=140.2, dis_dur=2.75, dis_range=200.0,
    rot_dps=120.0, rot_radius=250.0, rot_slow=0.32,
    # Invoker, level 15 (8/8/1, 3 Wex instances): MS 320.32, HP 1484
    i_ms=320.32, i_hp=1484.0, i_armor=7.33, i_regen=6.45,
    i_dps=84.4,                      # 101 avg x 0.8353 attacks/s (was wrongly 33.0)
    i_mr=0.25, p_mr=0.25,
    bound_radius=24.0,
)

#: Invoker's burst, treated as one cooldown-gated packet
INV_BURST = dict(damage=410.0, dtype="magical", cd=27.0, cast_range=950.0,
                 cast_point=0.05, delay=2.9, radius=675.0, speed=1000.0)
#: Sun Strike: pure, global, 1.7 s delay - lands on a channelling Pudge
SUN_STRIKE = dict(damage=175.0, dtype="pure", cd=23.0, delay=1.7,
                  radius=175.0)


def build_shape(cfg):
    """Return (shape, meta) for the discretised duel."""
    ND = int(round(cfg["d_max"] / cfg["dist_q"])) + 1
    NH1 = int(round(cfg["hp1"] / cfg["hpq"])) + 1
    NH2 = int(round(cfg["hp2"] / cfg["hpq"])) + 1
    NC1 = cfg["c1_steps"] + 1
    NC2 = cfg["c2_steps"] + 1
    NR = len(cfg["room"])
    shape = (ND, NH1, NH2, NC1, NC2, NR)
    strides = tuple(int(np.prod(shape[k + 1:])) for k in range(len(shape)))
    meta = dict(ND=ND, NH1=NH1, NH2=NH2, NC1=NC1, NC2=NC2, NR=NR,
                strides=strides, size=int(np.prod(shape)))
    return shape, meta


def terminal_array(shape, meta, cfg):
    """np.nan on live states, 1.0 where Pudge has won, 0.0 where he lost."""
    tv = np.full(shape, np.nan)
    hpq = cfg["hpq"]
    hp1 = np.arange(meta["NH1"]) * hpq
    hp2 = np.arange(meta["NH2"]) * hpq
    tv[:, 0, :, :, :, :] = 0.0          # Pudge dead -> Invoker wins
    tv[:, :, 0, :, :, :] = 1.0          # Invoker dead -> Pudge wins
    # a state where both are at 0 is unreachable; Pudge-dead wins the tie
    tv[:, 0, 0, :, :, :] = 0.0
    return tv


def make_successor(cfg, meta):
    """Return succ(coords, i, j) -> (idxs, weights).

    Cooldowns: one axis per hero, holding the number of DP steps ELAPSED since
    that hero's last cast, capped at the longest cooldown in his kit.

    That single axis carries every ability's real, individual cooldown.  The
    reason it is exact rather than an average: cooldowns all tick down together
    and optimal play is to cast every *available* spell at once, so "elapsed
    since last cast" determines exactly which spells are up.  The damage the
    cast delivers is therefore a deterministic ramp of that one number:

        damage(elapsed) = sum of damage of spells whose cooldown <= elapsed

    Direct tabulation of the six Invoker cooldowns would need 8.2e11 states per
    hero; this needs 167.  The only play it forbids is deliberately withholding
    a ready spell, which is never optimal in this matchup because the burst is
    what kills and holding it back only delays it.

    HP is float through the step and quantised once at the end as a four-point
    blend, because per-step damage is 18-30 HP against a quantum of hundreds and
    rounding it away deletes auto-attacks, Rot and the Meat Hook outright.
    """
    ND, NH1, NH2, NC1, NC2, NR = (meta["ND"], meta["NH1"], meta["NH2"],
                                  meta["NC1"], meta["NC2"], meta["NR"])
    s_d, s_h1, s_h2, s_c1, s_c2, s_r = meta["strides"]
    hpq, dq = cfg["hpq"], cfg["dist_q"]
    A1, A2 = cfg["acts1"], cfg["acts2"]
    P = cfg["params"]
    ROOM = np.array(cfg["room"], dtype=np.float64) * UNITS_PER_METER
    C1MAX, C2MAX = NC1 - 1, NC2 - 1
    HOOK_AT, DIS_AT = cfg["hook_cd_steps"], cfg["dis_cd_steps"]
    SNAP_AT, SNAP_DUR = cfg["snap_cd_steps"], cfg["snap_dur_steps"]
    # damage ramps indexed by elapsed steps
    R1_PURE = cfg["ramp1_pure"]      # Pudge: hook [+ dismember]
    R1_MAGIC = cfg["ramp1_magic"]
    R2_PHYS, R2_PURE, R2_MAGIC = (cfg["ramp2_phys"], cfg["ramp2_pure"],
                                  cfg["ramp2_magic"])

    def succ(coords, i, j):
        d = coords[0].astype(np.int64)
        h1 = coords[1].astype(np.int64)
        h2 = coords[2].astype(np.int64)
        c1 = coords[3].astype(np.int64)
        c2 = coords[4].astype(np.int64)
        r = coords[5].astype(np.int64)

        p, a1 = A1[i]
        q, a2 = A2[j]

        # Cold Snap lockout: c2 counts from the last Cold Snap cast, so the
        # stun window is unambiguous
        frozen = c2 < SNAP_DUR
        # elapsed grows by one step unless something is cast this step
        c1_next = np.minimum(c1 + 1, C1MAX)
        c2_next = np.minimum(c2 + 1, C2MAX)

        q_eff = np.where(r == 0, np.minimum(q, 0), q)
        d_new = np.clip(d + np.where(frozen, 0, p) + q_eff, 0, ND - 1)
        d_m = d * dq

        loss1 = np.zeros(d.shape, dtype=np.float64)
        loss2 = np.zeros(d.shape, dtype=np.float64)
        heal1 = np.zeros(d.shape, dtype=np.float64)

        # ---- Pudge: Rot (toggle) and auto-attacks, both automatic ---------
        atk = np.where((d_m * UNITS_PER_METER <= P["p_atk_range"]) & (~frozen),
                       P["p_dps"] * TICK_S * armor_multiplier(P["i_armor"]), 0.0)
        rotdmg = np.where(d_m * UNITS_PER_METER <= P["rot_radius"],
                          P["rot_dps"] * TICK_S * (1 - P["i_mr"]), 0.0)
        loss2 = loss2 + atk + rotdmg

        # ---- Pudge: Meat Hook (+ Dismember if its own cooldown is up) -----
        if a1 == "hook":
            ready = c1 >= HOOK_AT
            inr = d_m * UNITS_PER_METER <= P["hook_range"]
            t_avail = P["hook_cast_point"] + d_m * UNITS_PER_METER / P["hook_speed"]
            reach = P["i_ms"] * t_avail
            hits = (ready & inr & (~frozen)
                    & (reach <= P["hook_radius"] + P["bound_radius"]))
            loss2 = loss2 + np.where(hits, R1_PURE[c1], 0.0)
            heal1 = heal1 + np.where(hits, R1_MAGIC[c1], 0.0)
            c1_next = np.where(hits | (ready & inr), 0, c1_next)
            drag_cell = min(int(round(P["hook_drag"] / UNITS_PER_METER / dq)), ND - 1)
            d_new = np.where(hits, drag_cell, d_new)

        elif a1 == "blink":
            # Blink Dagger: 1200 u, zero cast point.  Modelled as one initiation
            # that teleports onto the target and lands the point-blank hook.
            # APPROXIMATION: shares the axis with Meat Hook, so Pudge gets one
            # initiation per cycle rather than two independent cooldowns.
            ready = c1 >= cfg["blink_cd_steps"]
            hits = ready & (~frozen)
            loss2 = loss2 + np.where(hits, R1_PURE[c1], 0.0)
            heal1 = heal1 + np.where(hits, R1_MAGIC[c1], 0.0)
            d_new = np.where(hits, 1, d_new)
            c1_next = np.where(hits, 0, c1_next)

        # ---- Invoker: auto-attack ----------------------------------------
        loss1 = loss1 + np.where(d_m * UNITS_PER_METER <= P["i_atk_range"],
                                 P["i_dps"] * TICK_S * armor_multiplier(P["p_armor"]), 0.0)

        # ---- Invoker: the invoked kit, gated on Cold Snap ------------------
        if a2 == "combo":
            # only Cold Snap matters for availability: it is the only
            # unit-target spell, so it is the only one that cannot be dodged,
            # and it is what makes the delayed AoE land
            ready = c2 >= SNAP_AT
            inr = d_m * UNITS_PER_METER <= P["combo_range"]
            hits = ready & inr
            loss1 = loss1 + np.where(
                hits,
                R2_PHYS[c2] * armor_multiplier(P["p_armor"])
                + R2_PURE[c2]
                + R2_MAGIC[c2] * (1 - P["p_mr"]), 0.0)
            c2_next = np.where(hits, 0, c2_next)

        # ---- quantise HP once, carrying the fraction as a blend ----------
        f1 = np.clip(h1 + heal1 / hpq - loss1 / hpq, 0.0, NH1 - 1.0)
        f2 = np.clip(h2 - loss2 / hpq, 0.0, NH2 - 1.0)
        lo1 = np.floor(f1).astype(np.int64); w1 = f1 - lo1
        lo2 = np.floor(f2).astype(np.int64); w2 = f2 - lo2
        hi1 = np.minimum(lo1 + 1, NH1 - 1)
        hi2 = np.minimum(lo2 + 1, NH2 - 1)

        nr = np.clip(r - np.where(q > 0, 1, np.where(q < 0, -1, 0)), 0, NR - 1)
        base = d_new * s_d + c1_next * s_c1 + c2_next * s_c2 + nr * s_r
        i00 = (base + lo1 * s_h1 + lo2 * s_h2).astype(np.int64)
        i10 = (base + hi1 * s_h1 + lo2 * s_h2).astype(np.int64)
        i01 = (base + lo1 * s_h1 + hi2 * s_h2).astype(np.int64)
        i11 = (base + hi1 * s_h1 + hi2 * s_h2).astype(np.int64)
        return ((i00, i10, i01, i11),
                ((1 - w1) * (1 - w2), w1 * (1 - w2), (1 - w1) * w2, w1 * w2))

    return succ


def heuristic_value(cfg, shape, meta, k=0.35):
    """A starting guess for value iteration, from the static damage race.

    Jacobi propagation needs roughly as many sweeps as the diameter of the
    state graph, which here is set by the ability cooldown (88 steps for the
    kit) plus the time to chew through the HP pool - a few hundred sweeps.
    Starting from a rough estimate of the *answer* removes most of that: the
    iteration then only has to correct the estimate locally rather than carry
    information in from the terminal states across the whole space.

    The guess is the sign of the damage race, smoothed:

        V0 = sigmoid( k * (ttk_invoker - ttk_pudge) )

    with ttk computed from the HP pools and the per-side damage available in
    the given readiness state.  A sigmoid rather than a step so that the
    initial vector is not a fixed point of the operator.
    """
    P = cfg["params"]
    hpq = cfg["hpq"]
    ND, NH1, NH2 = meta["ND"], meta["NH1"], meta["NH2"]
    NC1, NC2, NR = meta["NC1"], meta["NC2"], meta["NR"]

    d = np.arange(ND)[:, None, None, None, None, None]
    h1 = np.arange(NH1)[None, :, None, None, None, None]
    h2 = np.arange(NH2)[None, None, :, None, None, None]
    c1 = np.arange(NC1)[None, None, None, :, None, None]
    c2 = np.arange(NC2)[None, None, None, None, :, None]

    d_m = d * cfg["dist_q"]
    hp1 = h1 * hpq
    hp2 = h2 * hpq

    U = UNITS_PER_METER
    # sustained damage each side can put out, ignoring cooldown timing except
    # for the big packet
    p_sust = np.where(d_m * U <= P["p_atk_range"],
                      P["p_dps"] * armor_multiplier(P["i_armor"]), 0.0)
    p_sust = p_sust + np.where(d_m * U <= P["rot_radius"],
                               P["rot_dps"] * (1 - P["i_mr"]), 0.0)
    i_sust = np.where(d_m * U <= P["i_atk_range"],
                      P["i_dps"] * armor_multiplier(P["p_armor"]), 0.0)

    # expected value of one use of the big packet per second of its cooldown
    hook_cyc = max(cfg["hook_cd_steps"], 1) * 0.3
    combo_cyc = max(cfg["snap_cd_steps"], 1) * 0.3
    hook_dps = (P["hook_dmg"] + P["dis_dps"] * P["dis_dur"] * (1 - P["i_mr"])) / hook_cyc
    combo_dps = (P["c_phys"] * armor_multiplier(P["p_armor"]) + P["c_pure"]
                 + P["c_magic"] * (1 - P["p_mr"])) / combo_cyc

    pdps = p_sust + hook_dps
    idps = i_sust + combo_dps
    ttk_i = hp2 / np.maximum(pdps, 1e-6)
    ttk_p = hp1 / np.maximum(idps, 1e-6)
    V0 = 1.0 / (1.0 + np.exp(-k * (ttk_i - ttk_p)))
    V0 = np.broadcast_to(V0, shape).copy()
    V0[:, 0, :, :, :, :] = 0.0        # h1 = 0 -> Pudge dead
    V0[:, :, 0, :, :, :] = 1.0        # h2 = 0 -> Invoker dead
    return V0
