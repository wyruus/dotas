"""
Item profiles for the duel, patch 7.41f.

Source: Valve's official item datafeed, transcribed into data/items_7.41f.md.
Only items that can matter in a Pudge vs Invoker 1v1 are listed.

An item build is a set of at most 6 items plus one neutral.  With unlimited
gold the objective is NOT cost - it is **slots**, so the search is over
multisets of at most 6 items from the pool below.

Two kinds of effect are modelled, and they are very different in the solver:

* **stat items** just move the hero's profile (HP, armour, MR, move speed,
  damage, attack speed).  They cost nothing in state space.
* **active items** add an action and a mechanic (a gap close, a hex, a spell
  block).  Each one needs its own branch in the transition, and some need a
  state axis.  Those are listed with `active=...` and implemented explicitly.

Cooldown reduction is multiplicative across sources and has no documented cap;
Octarine Core is the only realistic source in a 6-slot duel build.
"""

from __future__ import annotations

# name -> stat deltas and optional active
ITEMS = {
    # ---------------- boots (the boots group does not stack) -------------
    "Boots of Speed":      dict(ms_flat=45),
    "Phase Boots":         dict(ms_flat=50, damage=18, armor=4,
                                active="phase", active_cd=8.0,
                                note="+20% MS for 3 s"),
    "Power Treads":        dict(ms_flat=55, attack_speed=25, primary=10),
    "Boots of Travel 2":   dict(ms_flat=110),
    "Tranquil Boots":      dict(ms_flat=65, hp_regen=14),

    # ---------------- mobility -------------------------------------------
    "Blink Dagger":        dict(cost=2250, active="blink", active_cd=15.0,
                                note="1200 u, 0 cast point, 3 s mute on damage, "
                                     "rootdisables, disjoints"),
    "Overwhelming Blink":  dict(cost=6800, str=25, active="blink_ob",
                                active_cd=15.0),
    "Swift Blink":         dict(cost=6800, agi=25, active="blink_sb",
                                active_cd=15.0),
    "Force Staff":         dict(cost=2200, int=10, hp=175, active="force",
                                active_cd=19.0, note="600 u push"),

    # ---------------- survivability --------------------------------------
    "Black King Bar":      dict(cost=4050, str=10, damage=24, active="bkb",
                                active_cd=95.0,
                                note="USELESS here: Meat Hook, Dismember and "
                                     "Sun Strike all pierce debuff immunity; "
                                     "only Rot is blocked"),
    "Linken's Sphere":     dict(cost=4800, all_attr=16, hp_regen=6.5,
                                mana_regen=4.25,
                                passive="spellblock",
                                note="blocks Dismember (unit target). Does NOT "
                                     "block Meat Hook (target point)"),
    "Lotus Orb":           dict(cost=3850, armor=10, hp_regen=6.5,
                                mana_regen=4, mana=250, active="lotus",
                                active_cd=15.0,
                                note="reflects Dismember back at Pudge, and the "
                                     "reflected cast resolves FIRST"),
    "Ghost Scepter":       dict(cost=1500, all_attr=5, active="ghost",
                                active_cd=22.0,
                                note="+30% magic vulnerability - bad here, "
                                     "Pudge's damage is pure/magical"),
    "Aeon Disk":           dict(cost=3000, hp=250, mana=300, passive="aeon",
                                note="below 70% HP: strong dispel + 2.5 s of "
                                     "zero damage both ways"),
    "Heart of Tarrasque":  dict(cost=5300, str=40, passive="behemoth",
                                note="+1% max HP regen, +1.5% of missing HP"),
    "Satanic":             dict(cost=5050, str=25, damage=25, lifesteal=0.25,
                                active="unholy", active_cd=40.0),
    "Shiva's Guard":       dict(cost=4500, armor=17,
                                passive="freezing_aura",
                                note="-40 attack speed aura to enemies"),
    "Assault Cuirass":     dict(cost=5125, armor=10, attack_speed=30,
                                passive="assault_aura",
                                note="-5 armour to enemies in 1200"),
    "Eternal Shroud":      dict(cost=3900, mr=0.20, str=10, hp=250,
                                passive="shroud",
                                note="restores mana = 25% of spell damage taken"),

    # ---------------- control --------------------------------------------
    "Scythe of Vyse":      dict(cost=5200, int=30, mana_regen=8.5,
                                active="hex", active_cd=20.0, active_range=800,
                                note="2.8 s hex: silence + mute + disarm, base MS "
                                     "set to 140. Interrupts Dismember"),
    "Eul's Scepter":       dict(cost=2600, int=10, mana_regen=2.5, ms_flat=20,
                                active="cyclone", active_cd=23.0,
                                active_range=550,
                                note="2.5 s cyclone; duration NOT reduced by "
                                     "status resistance. Interrupts Dismember"),
    "Wind Waker":          dict(cost=6800, int=35, mana_regen=3, ms_flat=30,
                                active="cyclone", active_cd=19.0),
    "Orchid Malevolence":  dict(cost=3275, int=12, damage=20, attack_speed=35,
                                mana_regen=2.5, active="silence",
                                active_cd=18.0, active_range=900,
                                note="5 s silence; interrupts Dismember"),
    "Bloodthorn":          dict(cost=6400, int=25, damage=20, attack_speed=70,
                                mana_regen=4, active="silence",
                                active_cd=15.0, active_range=900),
    "Abyssal Blade":       dict(cost=6250, str=30, damage=35,
                                active="stun", active_cd=35.0,
                                active_range=150,
                                note="1.6 s stun, PIERCES debuff immunity"),
    "Rod of Atos":         dict(cost=2250, int=12, hp=275, active="root",
                                active_cd=18.0, active_range=1100,
                                note="2 s root - does NOT interrupt Dismember"),
    "Blade Mail":          dict(cost=2400, damage=15, armor=7,
                                active="blademail", active_cd=25.0,
                                note="returns 85% of incoming damage for 5.5 s"),

    # ---------------- damage / utility -----------------------------------
    "Octarine Core":       dict(cost=5100, cdr=0.25, hp=450, mana=450,
                                mana_regen=6,
                                note="-25% cooldown, multiplicative, no cap"),
    "Refresher Orb":       dict(cost=5000, hp_regen=14, mana_regen=7,
                                active="refresh", active_cd=180.0,
                                note="180 s - outside the horizon of a duel"),
    "Eye of Skadi":        dict(cost=5900, all_attr=35, passive="skadi",
                                note="attacks slow MS 25/50% and cut health "
                                     "restoration 50%"),
    "Dagon 5":             dict(cost=7400, all_attr=10, hp=240, mana=450,
                                active="dagon", active_cd=15.0,
                                active_range=640, note="800 magic damage"),
    "Monkey King Bar":     dict(cost=5000, damage=50, attack_speed=50),
    "Daedalus":            dict(cost=5200, damage=88),
    "Mage Slayer":         dict(cost=3100, mr=0.18, damage=12, hp_regen=5.5,
                                mana_regen=2.5, passive="mage_slayer",
                                note="-40% spell damage dealt by the target"),
    "Vladmir's Offering":  dict(cost=2200, armor=1, mana_regen=0.75,
                                passive="vladmir"),
    "Aghanim's Scepter":   dict(cost=4200, all_attr=10, hp=175, mana=175,
                                passive="aghs"),
    "Aghanim's Shard":     dict(cost=1400, passive="shard"),
}

#: neutral items that are actually obtainable in 7.41f (tiers 0-4 only; the
#: tier-5 artifacts players remember have item_cost 0 and appear in no tier).
NEUTRALS = {
    "Stygian Desolator":  dict(tier=4, passive="greater_corruption",
                               note="-13 armour for 7 s on attack"),
    "Minotaur Horn":      dict(tier=4, active="mini_bkb", active_cd=40.0,
                               note="2 s Lesser Avatar: basic dispel, +60% MR"),
    "Fallen Sky":         dict(tier=4, active="fallen_sky", active_cd=25.0,
                               note="meteor, 1.6 s stun in 315 AoE, disjoints"),
    "Spider Legs":        dict(tier=4, active="skitter", active_cd=15.0,
                               note="+20% MS, free pathing 12 s"),
    "Witchbane":          dict(tier=4, active="cleanse", active_cd=30.0),
    "Dandelion Amulet":   dict(tier=3, passive="magic_barrier"),
    "Idol of Scree'auk":  dict(tier=3, active="scree", active_cd=30.0),
}


def profile_from_build(items, base):
    """Apply a list of item names to a hero profile dict.

    Only the boots group collapses (highest flat MS wins); every other flat
    bonus stacks.  Percentage move speed stacks additively.
    """
    out = dict(base)
    flat_ms = []
    for name in items:
        it = ITEMS.get(name)
        if it is None:
            continue
        if "ms_flat" in it:
            flat_ms.append(it["ms_flat"])
            continue                      # boots handled as a group below
        for k_src, k_dst in (("str", "str"), ("agi", "agi"), ("int", "int"),
                             ("all_attr", "all_attr"), ("hp", "hp"),
                             ("mana", "mana"), ("hp_regen", "hp_regen"),
                             ("mana_regen", "mana_regen"), ("armor", "armor"),
                             ("mr", "mr"), ("damage", "damage"),
                             ("attack_speed", "attack_speed"),
                             ("lifesteal", "lifesteal"), ("cdr", "cdr")):
            if k_src in it:
                out[k_dst] = out.get(k_dst, 0.0) + it[k_src]
    out["ms_flat"] = out.get("ms_flat", 0.0) + (max(flat_ms) if flat_ms else 0.0)
    return out


#: which actives are actually implemented in the tactical model.  The rest are
#: transcribed from the datafeed but not yet wired into the transition.
IMPLEMENTED_ACTIVES = {"blink"}
