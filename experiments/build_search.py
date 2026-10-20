"""
Analytical build screening.

The DP is far too expensive to run per build pair: one run is 11-23 minutes and
the build space is thousands of multisets.  So the search has to be inverted -
screen the whole space analytically in milliseconds, then spend DP runs only on
the Pareto front.

The screen is the damage race the DP rediscovered:

    does Pudge survive one Invoker combo, and can he then kill before the next?

        HP_p - combo_after_mitigation  >  HP_i / pudge_dps * invoker_sustained_dps

This is exactly the inequality that produced the ~2771 HP threshold, matching
the DP.  It is obviously a lower-fidelity model than the DP - it ignores
positioning, cooldown state and the Cold Snap lock - so it must never be quoted
as an answer.  Its job is to rank and discard.

Usage:
    python3 -m experiments.build_search
    python3 -m experiments.build_search --slots 6 --top 20
"""

from __future__ import annotations

import argparse
import itertools
import numpy as np
from dataclasses import dataclass, field

from data.items import ITEMS, profile_from_build
from dota2sim.constants import (armor_multiplier, attacks_per_second,
                                total_attack_speed, magic_resistance)

# --------------------------------------------------------------------------
# Level-15 baselines, no items (patch 7.41f)
# --------------------------------------------------------------------------
PUDGE_BASE = dict(level=15, str=67.0, agi=32.6, int=41.2, hp=1594.0,
                  armor=5.4333, aps=0.7800, dmg=115.0)
INVOKER_BASE = dict(level=15, str=54.0, agi=42.0, int=78.0, hp=1484.0,
                    armor=7.1667, aps=0.8412, dmg=156.0)

#: Pudge's kit at level 15
HOOK_DMG = 360.0          # pure, ignores armour and magic resistance
DISMEMBER_RAW = 385.6     # 140.2 dps x 2.75 s, magical
HOOK_CYCLE = 12.0
ROT_DPS = 120.0           # magical

#: Invoker's kit at 8/1/8 with three Exort instances
COMBO_PHYS = 823.0        # Cold Snap, ~9.8 procs
COMBO_PURE = 525.0        # Sun Strike, ignores everything
COMBO_MAGIC = 1075.0      # Meteor + EMP + Deafening + Tornado
COMBO_CYCLE = 26.0


#: items whose *active* changes the race; everything else is just stats
_FLAGGED = {"Aeon Disk", "Blink Dagger", "Linken's Sphere", "Lotus Orb",
            "Octarine Core", "Abyssal Blade", "Scythe of Vyse",
            "Eul's Scepter", "Shiva's Guard", "Assault Cuirass", "Blade Mail"}


@dataclass
class Side:
    name: str
    hp: float
    armor: float
    mr: float
    dps: float                       # sustained attack damage per second
    combo_phys: float = 0.0
    combo_pure: float = 0.0
    combo_magic: float = 0.0
    combo_cycle: float = 1e9
    burst_phys: float = 0.0
    burst_magic: float = 0.0
    burst_cycle: float = 1e9
    flags: frozenset = frozenset()          # item actives/passives that matter

    def combo_after(self, target: "Side") -> float:
        return (self.combo_phys * armor_multiplier(target.armor)
                + self.combo_pure
                + self.combo_magic * (1.0 - target.mr))

    def burst_after(self, target: "Side") -> float:
        return (self.burst_phys * armor_multiplier(target.armor)
                + self.burst_magic * (1.0 - target.mr))

    def dps_after(self, target: "Side") -> float:
        return self.dps * armor_multiplier(target.armor)


def make_pudge(items) -> Side:
    b = PUDGE_BASE
    p = profile_from_build(items, dict(str=0, agi=0, int=0, hp=0, mana=0,
                                       hp_regen=0, mana_regen=0, armor=0, mr=0,
                                       damage=0, attack_speed=0, ms_flat=0))
    st = b["str"] + p.get("str", 0) + p.get("all_attr", 0) + p.get("primary", 0)
    ag = b["agi"] + p.get("agi", 0) + p.get("all_attr", 0)
    it = b["int"] + p.get("int", 0) + p.get("all_attr", 0)
    hp = 120 + 22 * st + p.get("hp", 0)
    armor = ag / 6.0 + p.get("armor", 0)
    mr = magic_resistance(0.0, *([p["mr"]] if p.get("mr") else []))
    dmg = b["dmg"] + st - b["str"] + p.get("damage", 0)   # +1 per STR
    aps = attacks_per_second(total_attack_speed(ag, p.get("attack_speed", 0)))
    return Side("Pudge", hp, armor, mr, dmg * aps,
                burst_phys=HOOK_DMG, burst_magic=DISMEMBER_RAW,
                burst_cycle=HOOK_CYCLE,
                flags=frozenset(x for x in items if x in _FLAGGED))


def make_invoker(items, split=(8, 1, 8), instances=(0, 0, 3)) -> Side:
    b = INVOKER_BASE
    p = profile_from_build(items, dict(str=0, agi=0, int=0, hp=0, mana=0,
                                       hp_regen=0, mana_regen=0, armor=0, mr=0,
                                       damage=0, attack_speed=0, ms_flat=0))
    st = b["str"] + p.get("str", 0) + p.get("all_attr", 0)
    ag = b["agi"] + p.get("agi", 0) + p.get("all_attr", 0)
    hp = 120 + 22 * st + p.get("hp", 0)
    armor = ag / 6.0 + p.get("armor", 0)
    mr = magic_resistance(0.0, *([p["mr"]] if p.get("mr") else []))
    dmg = b["dmg"] + p.get("damage", 0)
    aps = attacks_per_second(total_attack_speed(ag, p.get("attack_speed", 0)))
    cycle = COMBO_CYCLE * (1.0 - min(0.9, p.get("cdr", 0.0)))
    return Side("Invoker", hp, armor, mr, dmg * aps,
                combo_phys=COMBO_PHYS, combo_pure=COMBO_PURE,
                combo_magic=COMBO_MAGIC, combo_cycle=cycle,
                flags=frozenset(x for x in items if x in _FLAGGED))


def screen(pudge: Side, invoker: Side) -> dict:
    """Race numbers plus a crude win indicator, with the decisive actives.

    Actives are what make item choice non-degenerate - a pure-stat screen just
    stacks the densest stat item on both sides and produces a meaningless
    equilibrium.  The ones that matter in this matchup:

      Aeon Disk   below 70% HP: strong dispel + 2.5 s of zero damage both ways,
                  which blanks the entire opening combo.  Turns a guaranteed
                  loss into a guaranteed survival.
      Blink       the only way Pudge closes against an equal-speed opponent
                  who is kiting; without it the race never starts.
      Linken's    blocks Dismember (unit target), cutting Pudge's burst.  Does
                  NOT block Meat Hook, which is a target-point ability.
      Lotus Orb   reflects Dismember.
      Octarine    -25% cooldown, so the combo comes round more often.
    """
    combo = invoker.combo_after(pudge)
    aeon = "Aeon Disk" in pudge.flags
    survives = aeon or pudge.hp > combo
    leftover = pudge.hp - (0.0 if aeon else combo)
    leftover = max(0.0, leftover)

    inv_sustained = invoker.dps_after(pudge)
    # Aeon blanks 2.5 s of incoming damage
    time_to_die = leftover / inv_sustained if inv_sustained > 0 else 1e9
    if aeon and inv_sustained > 0:
        time_to_die += 2.5

    burst = pudge.burst_after(invoker)
    if "Linken's Sphere" in invoker.flags:
        # Dismember is the magical half of the burst; the hook is pure and
        # passes through Spell Block entirely
        burst -= DISMEMBER_RAW * (1.0 - invoker.mr)
    burst = max(0.0, burst)
    pudge_dps = pudge.dps_after(invoker) + burst / pudge.burst_cycle
    time_to_kill = invoker.hp / pudge_dps if pudge_dps > 0 else 1e9

    # without a gap closer Pudge cannot start the race at all - equal move
    # speed, Invoker kites from 12.3 m, Rot reaches 4.76 m
    # Only Blink actually closes the gap: 1200 u with a zero cast point.
    # Abyssal (150), Scythe (800) and Eul's (550) all require Pudge to already
    # BE in range, so they win the fight once it has started but do nothing to
    # start it.
    can_close = "Blink Dagger" in pudge.flags
    return dict(combo=combo, survives=survives, ttk_invoker=time_to_kill,
                ttk_pudge=time_to_die,
                wins=bool(survives and can_close and time_to_kill < time_to_die))


def enumerate_builds(pool, slots):
    """Non-dominated multisets of at most `slots` items, by raw stat vector."""
    for k in range(1, slots + 1):
        for combo in itertools.combinations(sorted(pool), k):
            yield list(combo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--pool", default="", help="comma-separated item names")
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--slots2", type=int, default=4)
    a = ap.parse_args()

    # a small, relevant pool: the search over all 37 items x 6 slots is 2.3M
    # multisets, which is fine for a formula but the interesting items are few
    default_pool = [
        "Heart of Tarrasque", "Satanic", "Aeon Disk", "Blade Mail",
        "Eternal Shroud", "Shiva's Guard", "Assault Cuirass", "Black King Bar",
        "Linken's Sphere", "Lotus Orb", "Blink Dagger", "Phase Boots",
        "Power Treads", "Boots of Travel 2", "Octarine Core", "Scythe of Vyse",
        "Eul's Scepter", "Eye of Skadi", "Ghost Scepter", "Abyssal Blade",
        "Monkey King Bar", "Daedalus", "Mage Slayer", "Heaven's Halberd",
    ]
    pool = [x.strip() for x in a.pool.split(",")] if a.pool else default_pool

    if a.meta:
        meta(PUDGE_POOL, INVOKER_POOL, a.slots2, a.top)
        return
    pool = [x for x in pool if x in ITEMS]

    n = 0
    results = []
    for build in enumerate_builds(pool, a.slots):
        p = make_pudge(build)
        i = make_invoker([])          # Invoker bare, for the first cut
        r = screen(p, i)
        r["build"] = build
        results.append(r)
        n += 1
    print(f"проверено билдов Pudge: {n:,}")

    wins = [r for r in results if r["wins"]]
    print(f"проходят порог: {len(wins):,} ({100*len(wins)/max(1,n):.1f}%)")
    # rank by margin
    wins.sort(key=lambda r: -(r["ttk_pudge"] - r["ttk_invoker"]))
    print(f"\nлучшие билды Pudge (маржа = t_death − t_kill):")
    for r in wins[:a.top]:
        m = r["ttk_pudge"] - r["ttk_invoker"]
        print(f"  {m:+6.2f} с  комбо {r['combo']:5.0f}  "
              f"убить {r['ttk_invoker']:4.1f}  умереть {r['ttk_pudge']:5.1f}  "
              f"{', '.join(r['build'])}")
    if not wins:
        best = sorted(results, key=lambda r: r["ttk_pudge"] - r["ttk_invoker"])[-a.top:]
        print("  ни один не проходит; ближайшие:")
        for r in reversed(best):
            print(f"  {r['ttk_pudge']-r['ttk_invoker']:+6.2f} с  "
                  f"комбо {r['combo']:5.0f}  {', '.join(r['build'])}")




# ==========================================================================
# Two-sided search: both heroes get six slots, so the question is the
# equilibrium of the build meta-game, not "Pudge versus a bare Invoker".
# ==========================================================================
PUDGE_POOL = ["Heart of Tarrasque", "Satanic", "Aeon Disk", "Blade Mail",
              "Eternal Shroud", "Shiva's Guard", "Assault Cuirass",
              "Eye of Skadi", "Abyssal Blade", "Blink Dagger",
              "Phase Boots", "Boots of Travel 2"]
INVOKER_POOL = ["Octarine Core", "Scythe of Vyse", "Eul's Scepter",
                "Linken's Sphere", "Lotus Orb", "Aeon Disk", "Ghost Scepter",
                "Eye of Skadi", "Shiva's Guard", "Black King Bar",
                "Boots of Travel 2", "Heart of Tarrasque"]


def multisets(pool, slots):
    """Multisets of exactly `slots` items (repeats allowed)."""
    for combo in itertools.combinations_with_replacement(sorted(pool), slots):
        yield list(combo)


def meta(pool_p, pool_i, slots, top):
    pb = list(multisets(pool_p, slots))
    ib = list(multisets(pool_i, slots))
    print(f"билдов: Pudge {len(pb):,}  Invoker {len(ib):,}  "
          f"матрица {len(pb)*len(ib):,}")

    P = [make_pudge(b) for b in pb]
    I = [make_invoker(b) for b in ib]

    # Screen: does Pudge win the race?  Reduce each side to the builds that are
    # not dominated in the race, then solve the resulting 0/1 matrix game.
    A = np.zeros((len(pb), len(ib)))
    for x, p in enumerate(P):
        for y, i in enumerate(I):
            r = screen(p, i)
            A[x, y] = 1.0 if r["wins"] else 0.0

    print(f"доля побед Pudge в матрице: {A.mean()*100:.1f}%")
    val, prow, icol = _solve_binary(A)
    print(f"\nравновесие: цена игры V = {val:.3f}")
    psup = [(pb[k], prow[k]) for k in np.argsort(prow)[::-1] if prow[k] > 1e-3][:top]
    isup = [(ib[k], icol[k]) for k in np.argsort(icol)[::-1] if icol[k] > 1e-3][:top]
    print(f"\nбилды Pudge в поддержке равновесия ({len(psup)}):")
    for b, w in psup:
        print(f"   {w:.3f}  {', '.join(b)}")
    print(f"\nбилды Invoker в поддержке равновесия ({len(isup)}):")
    for b, w in isup:
        print(f"   {w:.3f}  {', '.join(b)}")


def _solve_binary(A, iters=2000):
    """Fictitious play on a 0/1 payoff matrix."""
    m, n = A.shape
    cp, cq = np.ones(m), np.ones(n)
    for _ in range(iters):
        p = cp / cp.sum()
        q = cq / cq.sum()
        cp[int(np.argmax(A @ q))] += 1.0
        cq[int(np.argmin(p @ A))] += 1.0
    p, q = cp / cp.sum(), cq / cq.sum()
    return float(p @ A @ q), p, q


if __name__ == "__main__":
    main()
