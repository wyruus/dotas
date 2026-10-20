"""
Construct the two duelists from the data tables.

Every number comes from the official datafeed sweep (patch 7.41f); see
dota2sim/herodata.py, data/invoker_7.41f.md and data/abilities_741f.py.
"""

from __future__ import annotations

from .constants import UNITS_PER_METER
from .forward import Unit, Duel
from data.abilities_741f import invoker_kit, INVOKE_FIXED, INVOKE_INNATE
from .herodata import PUDGE, PUDGE_ABILITIES_15
from data.items import ITEMS as _ITEMS


def item_stats(items, cdr_out=None):
    """Fold a set of items into a stat delta dict.

    Boots collapse as a group (highest flat move speed wins, as in the game);
    every other flat bonus stacks.  Cooldown reduction is multiplicative and
    returned separately because it multiplies existing cooldowns rather than
    adding to a stat.
    """
    out = dict(str=0.0, agi=0.0, int=0.0, all_attr=0.0, hp=0.0, mana=0.0,
               hp_regen=0.0, mana_regen=0.0, armor=0.0, mr=0.0, damage=0.0,
               attack_speed=0.0, ms_flat=0.0, ms_pct=0.0, evasion=0.0,
               lifesteal=0.0, spell_amp=0.0, status_resist=0.0, cdr=0.0,
               cast_range=0.0)
    boots = []
    for name in items or ():
        it = _ITEMS.get(name)
        if it is None:
            raise KeyError(f"unknown item: {name}")
        if "ms_flat" in it:
            boots.append(it["ms_flat"])
        for k in out:
            if k in it and k != "ms_flat":
                out[k] += it[k]
    out["ms_flat"] += max(boots) if boots else 0.0
    return out


#: item actives expressed in the same shape the forward engine casts from.
#: Only the actives that matter in this duel are implemented; the rest of the
#: 37 items contribute stats only, which is honest as long as it is stated.
ITEM_ACTIVES = {
    "Blink Dagger": dict(mana=0.0, cooldown=15.0, cast_point=0.0,
                         cast_range=1400.0, target="point", active="blink"),
    "Scythe of Vyse": dict(mana=250.0, cooldown=20.0, cast_point=0.0,
                           cast_range=800.0, target="unit", active="hex"),
    "Eul's Scepter": dict(mana=175.0, cooldown=23.0, cast_point=0.0,
                          cast_range=550.0, target="unit", active="cyclone"),
    "Orchid Malevolence": dict(mana=125.0, cooldown=18.0, cast_point=0.0,
                               cast_range=900.0, target="unit", active="silence"),
    "Abyssal Blade": dict(mana=75.0, cooldown=35.0, cast_point=0.0,
                          cast_range=150.0, target="unit", active="stun"),
    "Blade Mail": dict(mana=25.0, cooldown=25.0, cast_point=0.0,
                       cast_range=0.0, target="self", active="blademail"),
}


def pudge_abilities(st: float) -> dict:
    """Pudge's kit in the shape the forward engine expects.

    Dismember's damage scales with strength, so it is resolved here rather
    than stored as a constant.
    """
    src = PUDGE_ABILITIES_15
    h = src["meat_hook"]
    dis = src["dismember"]
    dps = dis["base_dps"] + dis["str_mult"] * st
    return {
        "meat_hook": dict(
            mana=h["mana"], cooldown=h["cooldown"], cast_point=h["cast_point"],
            cast_backswing=h["cast_backswing"], cast_range=h["cast_range"],
            travel_distance=h["travel_distance"], speed=h["speed"],
            radius=h["radius"], damage=h["damage"],
            damage_type=h["dtype"], pierces_immunity=h["pierces_immunity"],
            drag_stop=h["drag_stop_distance"],
            self_disable_factor=h["self_disable"],
        ),
        "rot": dict(
            mana=0.0, cooldown=0.0, cast_point=0.0,
            radius=src["rot"]["radius"],
            damage_per_second=src["rot"]["dps"],
            damage_type=src["rot"]["dtype"],
            self_damage=src["rot"]["self_damage"],
            slow=src["rot"]["slow"], tick=src["rot"]["tick"], toggle=True,
        ),
        "meat_shield": dict(
            mana=src["meat_shield"]["mana"],
            cooldown=src["meat_shield"]["cooldown"],
            cast_point=0.0, duration=src["meat_shield"]["duration"],
            block=src["meat_shield"]["block"],
        ),
        "dismember": dict(
            mana=dis["mana"], cooldown=dis["cooldown"],
            cast_point=dis["cast_point"], cast_range=dis["cast_range"],
            damage_type=dis["dtype"], duration=dis["duration"],
            intervals=dis["intervals"],
            damage_per_interval=dps * dis["duration"] / dis["intervals"],
            dps=dps, heals=True, channel=True,
            pierces_immunity=dis["pierces_immunity"],
        ),
    }


def _apply_items(stats, abilities, items):
    """Fold items into a resolved hero: stats into the profile, actives into
    the ability table so the ordinary cast machinery picks them up."""
    if not items:
        return
    d = item_stats(items)
    stats["max_hp"] += d["hp"] + 22.0 * (d["str"] + d["all_attr"])
    stats["max_mana"] += d["mana"] + 12.0 * (d["int"] + d["all_attr"])
    stats["hp_regen"] += d["hp_regen"] + 0.1 * (d["str"] + d["all_attr"])
    stats["mana_regen"] += d["mana_regen"] + 0.05 * (d["int"] + d["all_attr"])
    stats["armor"] += d["armor"] + (d["agi"] + d["all_attr"]) / 6.0
    if d["mr"]:
        from .constants import stack_resist
        stats["magic_resist"] = stack_resist(stats["magic_resist"], d["mr"])
    stats["attack_damage"] += d["damage"] + d["str"] + d["agi"] + d["int"] \
        + 3.0 * d["all_attr"]
    stats["attack_speed_bonus"] += d["attack_speed"] + d["agi"] + d["all_attr"]
    stats["ms_flat"] = stats.get("ms_flat", 0.0) + d["ms_flat"]
    stats["ms_pct"] = stats.get("ms_pct", 0.0) + d["ms_pct"]
    stats["evasion"] = d["evasion"]
    stats["items"] = tuple(items)
    stats["lifesteal"] = d["lifesteal"]
    stats["status_resist"] = d["status_resist"]
    if d["cdr"]:
        # multiplicative and applied to what is already there, as in the game
        for name, ab in abilities.items():
            if ab.get("cooldown", 0.0) > 0 and name not in (
                    "meat_hook", "blink"):
                ab["cooldown"] *= (1.0 - min(0.9, d["cdr"]))
    for name in items:
        act = ITEM_ACTIVES.get(name)
        if act:
            abilities[name] = dict(act)
            abilities[name]["item"] = True


# --------------------------------------------------------------------------
# Pudge, level 15: STR 67 / AGI 32.6 / INT 41.2
# --------------------------------------------------------------------------
def build_pudge(x_m=0.0, y_m=0.0, hp_bonus=0.0, armor_bonus=0.0,
                mr_bonus=0.0, ms_bonus=0.0, level=15, items=()) -> Unit:
    st = PUDGE.base_str + PUDGE.str_gain * (level - 1)
    ag = PUDGE.base_agi + PUDGE.agi_gain * (level - 1)
    it = PUDGE.base_int + PUDGE.int_gain * (level - 1)
    stats = dict(
        max_hp=120.0 + 22.0 * st + hp_bonus,
        max_mana=75.0 + 12.0 * it,
        hp_regen=2.5 + 0.1 * st,
        mana_regen=0.05 * it,
        armor=ag / 6.0 + armor_bonus,
        magic_resist=0.25 + mr_bonus,
        attack_damage=(PUDGE.base_attack_damage_min
                       + PUDGE.base_attack_damage_max) / 2.0 + st,
        attack_speed_bonus=ag,
        base_attack_time=1.7,
        ms_base=PUDGE.base_move_speed,
        ms_flat=ms_bonus,
        ms_pct=0.0,
        attack_range=PUDGE.attack_range,
        projectile_speed=0.0,            # melee: the attack lands at the attack point
        attack_point=PUDGE.attack_point,
        turn_rate=PUDGE.turn_rate,
        collision=24.0,
        status_resist=0.0,
    )
    ab = pudge_abilities(st)
    _apply_items(stats, ab, items)
    u = Unit("pudge", x_m * UNITS_PER_METER, y_m * UNITS_PER_METER,
                stats, ab)
    u.items = stats.get("items", ())
    return u


# --------------------------------------------------------------------------
# Invoker, level 15, with a real orb distribution
# --------------------------------------------------------------------------
def build_invoker(Q=8, W=1, E=8, inst_q=0, inst_w=0, inst_e=3,
                  x_m=0.0, y_m=0.0, hp_bonus=0.0, armor_bonus=0.0,
                  mr_bonus=0.0, ms_bonus=0.0, level=15,
                  talent_25=False, items=()) -> Unit:
    st = 19.0 + 2.5 * (level - 1) + Q
    ag = 14.0 + 2.0 * (level - 1) + W
    it = 22.0 + 4.0 * (level - 1) + E
    t = 2 if talent_25 else 1
    stats = dict(
        max_hp=120.0 + 22.0 * st + hp_bonus,
        max_mana=75.0 + 12.0 * it,
        hp_regen=0.25 + 0.09 * st + Q * inst_q * t,
        mana_regen=0.05 * it,
        armor=ag / 6.0 + armor_bonus,
        magic_resist=0.25 + mr_bonus,
        attack_damage=22.0 + it + 2.0 * E * inst_e * t,
        attack_speed_bonus=ag + W * inst_w * t,
        base_attack_time=1.7,
        ms_base=280.0 * (1.0 + 0.006 * W * inst_w * t),
        ms_flat=ms_bonus,
        ms_pct=0.0,
        attack_range=600.0,
        projectile_speed=900.0,          # datafeed: Invoker's attack projectile
        attack_point=0.4,
        turn_rate=0.6,
        collision=24.0,
        status_resist=0.0,
    )
    kit = invoker_kit(Q, W, E, quas_inst=inst_q, wex_inst=inst_w,
                      exort_inst=inst_e, talent_25=talent_25)
    kit["invoke"] = dict(INVOKE_INNATE)
    kit["invoke"]["cooldown"] = max(
        0.0, INVOKE_INNATE["base_cooldown"]
        - INVOKE_INNATE["per_orb"] * (Q + W + E))
    kit["invoke"]["cast_point"] = 0.0
    kit["invoke"]["cast_range"] = 0.0
    _apply_items(stats, kit, items)
    u = Unit("invoker", x_m * UNITS_PER_METER, y_m * UNITS_PER_METER, stats, kit)
    u.items = stats.get("items", ())
    u.orbs = (Q, W, E)
    u.invoked = []
    return u


def make_duel(pudge_kw=None, invoker_kw=None, arena=(30.0, 80.0),
              pudge_pos=(2.0, 40.0), invoker_pos=(8.0, 40.0)) -> Duel:
    p = build_pudge(x_m=pudge_pos[0], y_m=pudge_pos[1], **(pudge_kw or {}))
    i = build_invoker(x_m=invoker_pos[0], y_m=invoker_pos[1], **(invoker_kw or {}))
    return Duel(p, i, *arena)
