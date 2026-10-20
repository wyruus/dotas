"""
Forward model of the duel - exact rules, driven by data/abilities_741f.py.

This is the ground truth that search runs on.  Unlike the tabulated DP it has
no state-space limit, so it keeps everything the reduction had to throw away:
a two-dimensional arena with the loss-on-exit rule, ten Invoker abilities each
with its own cooldown and cast point, projectiles, delayed ground effects,
channels and damage types.

Time is integrated at the server tick (1/30 s).  Decisions are taken every
0.3 s, matching the user's grid.

Ordering inside one tick follows the game: buffs and cooldowns tick, casts
progress and fire, regeneration applies, then movement, then attacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .constants import (
    SERVER_TICK, UNITS_PER_METER, armor_multiplier, magic_resistance,
    attacks_per_second, total_attack_speed, move_speed, turn_time,
    stack_resist, STATUS_RESIST_CAP,
)

# --------------------------------------------------------------------------
# Disable flags
# --------------------------------------------------------------------------
STUN, ROOT, SILENCE, MUTE, DISARM, HEX, CYCLONE = (1 << i for i in range(7))
TAUNT, FEAR, ETHEREAL, SPELL_IMMUNE, INVISIBLE, BREAK, KNOCKBACK = (
    1 << i for i in range(7, 14))

CANNOT_MOVE = STUN | ROOT | CYCLONE | HEX | TAUNT | FEAR
CANNOT_TURN = STUN | CYCLONE | HEX | TAUNT | FEAR
CANNOT_ATTACK = STUN | CYCLONE | HEX | DISARM | TAUNT | FEAR | ETHEREAL
CANNOT_CAST = STUN | CYCLONE | HEX | SILENCE | TAUNT | FEAR
CANNOT_ITEM = STUN | CYCLONE | HEX | MUTE | TAUNT | FEAR


# --------------------------------------------------------------------------
# Buffs
# --------------------------------------------------------------------------
@dataclass
class Buff:
    name: str
    duration: float
    disables: int = 0
    #: periodic damage
    dps: float = 0.0
    dtype: str = "magical"
    dps_tick: float = 0.0          # seconds between damage applications
    dps_accum: float = 0.0
    #: stat modifiers
    bonus_armor: float = 0.0
    bonus_magic_resist: float = 0.0
    bonus_attack_speed: float = 0.0
    bonus_move_speed_pct: float = 0.0
    move_slow: float = 0.0
    #: resistances applied to the WEARER, extra
    bonus_attack_damage: float = 0.0
    #: behaviour
    dispellable: bool = True
    source: str = ""
    #: Cold Snap style: procs on taking damage
    proc_on_damage: bool = False
    proc_cooldown: float = 0.0
    proc_time_left: float = 0.0
    proc_stun: float = 0.0
    proc_damage: float = 0.0
    proc_dtype: str = "physical"
    on_expire_damage: float = 0.0
    on_expire_dtype: str = "magical"
    #: flat damage negation per instance (Meat Shield, Stout Shield, Vanguard)
    damage_block: float = 0.0
    #: forced movement, units per second, away from the source
    knockback: float = 0.0

    def active(self) -> bool:
        return self.duration > 0


# --------------------------------------------------------------------------
# Unit
# --------------------------------------------------------------------------
class Unit:
    def __init__(self, name, x, y, stats, abilities):
        self.name = name
        self.x = float(x)                  # units
        self.y = float(y)
        self.facing = 0.0
        self.stats = stats                 # resolved dict
        self.abilities = abilities         # name -> resolved numbers
        self.hp = stats["max_hp"]
        self.mana = stats["max_mana"]
        self.cooldowns: dict[str, float] = {}
        self.buffs: list[Buff] = []
        self.alive = True
        # casting
        self.cast_ability: str | None = None
        self.cast_left = 0.0
        self.cast_target = None            # ("point", x, y) | ("unit", name)
        # channel
        self.channel_ability: str | None = None
        self.channel_left = 0.0
        # attack
        self.attack_target = None
        self.attack_cooldown = 0.0
        self.attack_windup = 0.0
        # orders
        self.move_target = None
        self.rot_on = False
        self.invoked: list[str] = []
        self.items: tuple = ()
        self.spellblock_cd = 0.0
        self.aeon_cd = 0.0          # the duel starts with every item ready

    # -- state queries -----------------------------------------------------
    def _has(self, flag: int) -> bool:
        return any(b.disables & flag for b in self.buffs if b.active())

    def buff(self, flag: int) -> bool:
        return self._has(flag)

    def can_move(self):
        return self.alive and not self._has(CANNOT_MOVE)

    def can_turn(self):
        return self.alive and not self._has(CANNOT_TURN)

    def can_attack(self):
        return self.alive and not self._has(CANNOT_ATTACK)

    def can_cast(self):
        return self.alive and not self._has(CANNOT_CAST)

    def is_spell_immune(self):
        return self._has(SPELL_IMMUNE)

    def is_invisible(self):
        return self._has(INVISIBLE)

    # -- resolved stats ----------------------------------------------------
    def _sum(self, attr):
        return sum(getattr(b, attr) for b in self.buffs if b.active())

    @property
    def max_hp(self):
        return self.stats["max_hp"]

    @property
    def max_mana(self):
        return self.stats["max_mana"]

    def armor(self):
        return self.stats["armor"] + self._sum("bonus_armor")

    def magic_resist(self):
        base = self.stats.get("magic_resist", 0.25)
        mods = [b.bonus_magic_resist for b in self.buffs
                if b.active() and b.bonus_magic_resist]
        return stack_resist(base, *mods) if mods else base

    def status_resist(self):
        return min(STATUS_RESIST_CAP, self.stats.get("status_resist", 0.0))

    def move_speed(self):
        slows = [b.move_slow for b in self.buffs if b.active() and b.move_slow > 0]
        return move_speed(self.stats["ms_base"], self.stats.get("ms_flat", 0.0),
                          self.stats.get("ms_pct", 0.0) + self._sum("bonus_move_speed_pct"),
                          slows)

    def attack_damage(self):
        return self.stats["attack_damage"] + self._sum("bonus_attack_damage")

    def attacks_per_second(self):
        ias = self.stats.get("attack_speed_bonus", 0.0) + self._sum("bonus_attack_speed")
        return attacks_per_second(total_attack_speed(0.0, ias),
                                  self.stats["base_attack_time"])

    def hp_regen(self):
        return self.stats.get("hp_regen", 0.0)

    def mana_regen(self):
        return self.stats.get("mana_regen", 0.0)

    def attack_range(self):
        return self.stats["attack_range"] + self.stats.get("collision", 0) + 24.0

    def dist_to(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)


# --------------------------------------------------------------------------
# Projectiles and ground effects
# --------------------------------------------------------------------------
@dataclass
class Projectile:
    kind: str
    source: str
    x: float
    y: float
    vx: float
    vy: float
    remaining: float                # distance left to travel
    radius: float
    payload: dict = field(default_factory=dict)


@dataclass
class Summon:
    """A summoned unit.  Only what the duel needs: it chases, it attacks, and
    its attacks shred armour (Forge Spirits' Melting Strike)."""
    owner: str
    x: float
    y: float
    hp: float
    max_hp: float
    damage: float
    armor: float
    attack_range: float
    bat: float
    attack_point: float
    ms: float
    duration: float
    attack_cd: float = 0.0
    windup: float = 0.0
    #: Melting Strike
    shred_per_stack: float = 0.0
    stacks: int = 0
    max_stacks: int = 20
    stack_time: float = 0.0


@dataclass
class GroundEffect:
    kind: str
    source: str
    x: float
    y: float
    radius: float
    delay: float                    # time until it fires
    payload: dict = field(default_factory=dict)
    fired: bool = False
    #: travelling meteors carry their own velocity
    vx: float = 0.0
    vy: float = 0.0
    remaining_travel: float = 0.0
    duration: float = 0.0
    time_left: float = 0.0


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Duel:
    """Two heroes, an empty rectangle, exact ability rules."""

    def __init__(self, pudge: Unit, invoker: Unit,
                 arena_w_m=30.0, arena_h_m=80.0):
        self.u = {"pudge": pudge, "invoker": invoker}
        self.w = arena_w_m * UNITS_PER_METER
        self.h = arena_h_m * UNITS_PER_METER
        self.t = 0.0
        self.projectiles: list[Projectile] = []
        self.effects: list[GroundEffect] = []
        self.summons: list[Summon] = []
        self.log: list[tuple[float, str]] = []
        self.winner: str | None = None
        self.reason = ""

    # -- helpers -----------------------------------------------------------
    # -- vision ------------------------------------------------------------
    #: A fight lasts well under the 300 s day/night half-cycle, so vision is
    #: constant across a duel and the only thing that matters is the radius.
    #: It is 1800 units (34 m) by day, 800 (15 m) by night.
    VISION_DAY = 1800.0
    VISION_NIGHT = 800.0

    @property
    def vision_range(self):
        return self.VISION_DAY if (self.t % 600.0) < 300.0 else self.VISION_NIGHT

    def visible(self, observer: Unit, target: Unit) -> bool:
        """Can `observer` see `target`?  An invisible target is seen only with
        True Sight, which nothing in this duel grants."""
        if not target.alive:
            return False
        if target.is_invisible():
            return False
        return observer.dist_to(target) <= self.vision_range

    def other(self, u: Unit) -> Unit:
        return self.u["invoker"] if u is self.u["pudge"] else self.u["pudge"]

    def note(self, msg):
        self.log.append((round(self.t, 3), msg))

    def inside(self, x, y):
        return 0.0 <= x <= self.w and 0.0 <= y <= self.h

    # -- damage ------------------------------------------------------------
    def deal(self, source: Unit, target: Unit, amount: float, dtype: str,
             pierces_immunity=False, interrupts=True, label="",
             _no_reflect=False):
        """Apply one damage instance.  Returns the damage actually dealt."""
        if not target.alive or amount <= 0:
            return 0.0
        # A cycloned unit is invulnerable: nothing at all gets through, of any
        # damage type.  Without this, Tornado read as a free disable with no
        # downside and the search rated it as an ordinary nuke.
        if target.buff(CYCLONE):
            return 0.0
        if target.is_spell_immune() and not pierces_immunity and dtype != "physical":
            return 0.0

        if dtype == "physical":
            amount *= armor_multiplier(target.armor())
        elif dtype == "magical":
            if target.buff(ETHEREAL):
                amount *= 1.40
            amount *= (1.0 - target.magic_resist())
        # pure: untouched

        # Meat Shield and the damage-block items negate a flat amount from
        # EVERY incoming instance, after all reductions.  It was declared in
        # the ability table but never applied, so Pudge's third ability was a
        # no-op - the same class of error as the Tornado invulnerability.
        blk = target._sum("damage_block")
        if blk:
            amount = max(0.0, amount - blk)
        if any(x.name == "aeon_disk" and x.active() for x in target.buffs):
            return 0.0
        before = target.hp
        target.hp = max(0.0, target.hp - amount)
        dealt = before - target.hp
        if (dealt > 0 and not _no_reflect and any(
                x.name == "blade_mail" and x.active() for x in target.buffs)
                and source is not target):
            # returns the damage increased by 85%, before any reduction
            self.deal(target, source, dealt * 1.85, dtype,
                      pierces_immunity=True, interrupts=False,
                      label="blade_mail", _no_reflect=True)
        # Aeon Disk: below 70% health, a strong dispel and 2.5 s in which
        # neither side deals or takes damage.  It is the standard answer to a
        # single burst combo, so it has to exist before burst builds can be
        # compared fairly.
        if ("Aeon Disk" in target.items and target.aeon_cd <= 0.0
                and target.alive and target.hp <= 0.70 * target.max_hp):
            target.aeon_cd = 105.0
            target.buffs = [x for x in target.buffs if not x.disables]
            target.buffs.append(Buff("aeon_disk", 2.5, source=target.name))
            self.note(f"{target.name}: Aeon Disk triggered")
        if target.hp <= 0:
            self._kill(target)
        # Cold Snap and similar proc on taking a damage instance
        if dealt >= 10.0:
            self._on_damage_taken(target, source, dealt)
        return dealt

    def _on_damage_taken(self, target, source, dealt):
        for b in target.buffs:
            if b.active() and b.proc_on_damage and b.proc_time_left <= 0:
                b.proc_time_left = b.proc_cooldown
                if b.proc_stun > 0:
                    target.buffs.append(Buff(f"{b.name} stun", b.proc_stun,
                                             disables=STUN, source=b.source))
                if b.proc_damage > 0:
                    self.deal(target, target, b.proc_damage, b.proc_dtype,
                              label="proc")

    def _kill(self, u):
        u.alive = False
        if self.winner is None:
            self.winner = self.other(u).name
            self.reason = f"{u.name} died"
        self.note(f"{u.name} died")

    # -- buffs -------------------------------------------------------------
    def apply_buff(self, target: Unit, b: Buff, pierces_immunity=False):
        if target.is_spell_immune() and not pierces_immunity and b.disables:
            return
        if b.disables:
            b.duration *= (1.0 - target.status_resist())
        target.buffs.append(b)

    # ------------------------------------------------------------------
    # One server tick
    # ------------------------------------------------------------------
    def step(self, dt=SERVER_TICK):
        if self.winner is not None:
            return
        self.t += dt
        for u in self.u.values():
            self._tick_buffs(u, dt)
            self._tick_cooldowns(u, dt)
        for u in self.u.values():
            self._tick_cast(u, dt)
            self._tick_channel(u, dt)
        for u in self.u.values():
            self._regen(u, dt)
            self._tick_aura(u, dt)
        for u in self.u.values():
            self._tick_move(u, dt)
        self._tick_summons(dt)
        self._tick_forced_movement(dt)
        self._tick_projectiles(dt)
        self._tick_effects(dt)
        for u in self.u.values():
            self._tick_attack(u, dt)
        self._check_end()

    def _tick_buffs(self, u, dt):
        keep = []
        for b in u.buffs:
            b.duration -= dt
            if b.proc_time_left > 0:
                b.proc_time_left -= dt
            if b.active():
                if b.dps > 0 and b.dps_tick > 0:
                    b.dps_accum += dt
                    while b.dps_accum >= b.dps_tick:
                        b.dps_accum -= b.dps_tick
                        self.deal(self.u.get(b.source, u), u,
                                  b.dps * b.dps_tick, b.dtype,
                                  interrupts=False, label=b.name)
                if u.hp <= 0:
                    break
                keep.append(b)
            else:
                if b.on_expire_damage > 0:
                    self.deal(self.u.get(b.source, u), u, b.on_expire_damage,
                              b.on_expire_dtype, label=b.name)
        u.buffs = keep

    def _tick_cooldowns(self, u, dt):
        for k in list(u.cooldowns):
            u.cooldowns[k] -= dt
            if u.cooldowns[k] <= 0:
                del u.cooldowns[k]
        if u.attack_cooldown > 0:
            u.attack_cooldown -= dt
        if u.spellblock_cd > 0:
            u.spellblock_cd -= dt
        if u.aeon_cd > 0:
            u.aeon_cd -= dt

    def _regen(self, u, dt):
        if not u.alive:
            return
        u.hp = min(u.max_hp, u.hp + u.hp_regen() * dt)
        u.mana = min(u.max_mana, u.mana + u.mana_regen() * dt)

    def _tick_aura(self, u, dt):
        """Rot: a toggle, so it keeps burning through stuns and channels."""
        if u.name != "pudge" or not u.rot_on or not u.alive:
            return
        rot = u.abilities["rot"]
        e = self.other(u)
        # Rot damages in discrete 0.2 s ticks, not continuously.  The instance
        # SIZE matters, not just the rate: on-damage effects such as Cold Snap
        # only trigger on instances of 10 or more, so a smooth 3 HP per server
        # tick would silently never proc it while the real 18 HP tick does.
        tick = rot.get("tick", 0.2)
        rot["_acc"] = rot.get("_acc", 0.0) + dt
        if e.alive and u.dist_to(e) <= rot["radius"]:
            if rot.get("slow"):
                self.apply_buff(e, Buff("rot slow", 0.5, move_slow=rot["slow"],
                                        source="pudge"))
            while rot["_acc"] >= tick:
                rot["_acc"] -= tick
                self.deal(u, e, rot["damage_per_second"] * tick,
                          rot["damage_type"], interrupts=False, label="rot")
        rot["_acc_self"] = rot.get("_acc_self", 0.0) + dt
        while rot["_acc_self"] >= tick:
            rot["_acc_self"] -= tick
            self.deal(u, u, rot["self_damage"] * (1.0 - u.magic_resist()) * tick,
                      "pure", interrupts=False, label="rot_self")
            if u.hp < 1.0:
                u.hp = 1.0
        if False:
            # Rot's self-damage is magical and IS reduced by magic resistance,
            # so at base 25% Pudge burns 90/s rather than the full 120/s.
            #
            # It goes through deal() rather than subtracting HP directly,
            # because it is a real damage instance and therefore triggers
            # on-damage effects - most importantly Cold Snap, which is why an
            # Invoker who lands Cold Snap on a Rotting Pudge gets a free
            # machine-gun of procs.  It is never lethal: Pudge cannot deny
            # himself with Rot.
            self.deal(u, u, rot["self_damage"] * (1.0 - u.magic_resist()) * dt,
                      "pure", interrupts=False, label="rot_self")
            if u.hp < 1.0:
                u.hp = 1.0

    def _tick_move(self, u, dt):
        if not u.alive or not u.can_move() or u.move_target is None:
            return
        tx, ty = u.move_target
        dx, dy = tx - u.x, ty - u.y
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return
        want = math.atan2(dy, dx)
        if u.can_turn():
            max_turn = (u.stats["turn_rate"] / 0.03) * dt
            diff = (want - u.facing + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > max_turn:
                u.facing += math.copysign(max_turn, diff)
                return
            u.facing = want
        step = min(dist, (u.move_speed() * dt))
        u.x += dx / dist * step
        u.y += dy / dist * step

    def _tick_cast(self, u, dt):
        if u.cast_ability is None:
            return
        u.cast_left -= dt
        if u.cast_left <= 0:
            ab, tgt = u.cast_ability, u.cast_target
            u.cast_ability = None
            u.cast_target = None
            self._fire(u, ab, tgt)

    def _tick_channel(self, u, dt):
        if u.channel_ability is None:
            return
        if not u.can_cast():
            self.note(f"{u.name}: channel of {u.channel_ability} broken")
            u.channel_ability = None
            return
        u.channel_left -= dt
        ab = u.abilities[u.channel_ability]
        target = self.other(u)
        # Dismember ticks its damage and heal over the channel
        total = ab["duration"]
        tick = ab.get("intervals", 6)
        per_tick = ab.get("damage_per_interval")
        if per_tick is not None and u.channel_left > 0:
            accum = ab.setdefault("_accum", 0.0) + dt
            period = total / tick
            while accum >= period:
                accum -= period
                dmg = per_tick
                dealt = self.deal(u, target, dmg, ab["damage_type"],
                                  pierces_immunity=ab.get("pierces_immunity", False),
                                  label="dismember")
                u.hp = min(u.max_hp, u.hp + dealt)
                if not target.alive:
                    break
            ab["_accum"] = accum
        if u.channel_left <= 0:
            u.channel_ability = None

    # -- casting -----------------------------------------------------------
    def try_cast(self, u: Unit, ability_name: str, target) -> bool:
        """Begin a cast if legal.  target is ('point', x, y) or ('unit', name)."""
        if not u.alive or not u.can_cast() or u.rot_on is None:
            pass
        if not u.alive or not u.can_cast():
            return False
        if u.cast_ability is not None or u.channel_ability is not None:
            return False
        ab = u.abilities.get(ability_name)
        if ab is None:
            return False
        if u.cooldowns.get(ability_name, 0.0) > 0:
            return False
        if u.mana < ab.get("mana", 0.0):
            return False
        # Unit-target spells need vision of the target; point-target and
        # skillshot abilities do not (Meat Hook explicitly hits invisible
        # units, which is why invisibility is not a hard counter to it).
        if target is not None and target[0] == "unit":
            tgt = self.u[target[1]]
            if tgt is not u and not self.visible(u, tgt):
                return False
            # Linken's Sphere blocks the first unit-targeted spell every 14 s.
            # It is the correct answer to Dismember - but NOT to Meat Hook,
            # which is a target-point skillshot and therefore never triggers
            # Spell Block at all.
            if (tgt is not u and "Linken's Sphere" in tgt.items
                    and tgt.spellblock_cd <= 0.0):
                tgt.spellblock_cd = 14.0
                self.note(f"Linken's blocked {ability_name}")
                u.mana -= ab.get("mana", 0.0)
                u.cooldowns[ability_name] = ab["cooldown"] if ab["cooldown"] > 0 else 0.0
                return True
        rng = ab.get("cast_range", 0.0)
        if rng > 0 and target is not None and target[0] == "point":
            d = math.hypot(target[1] - u.x, target[2] - u.y)
            if d > rng + 24.0:
                return False
        elif rng > 0 and target is not None and target[0] == "unit":
            t = self.u[target[1]]
            if u.dist_to(t) > rng + 24.0:
                return False
        u.mana -= ab.get("mana", 0.0)
        u.cooldowns[ability_name] = ab["cooldown"] if ab["cooldown"] > 0 else 0.0
        cp = ab.get("cast_point", 0.0)
        if cp <= 0:
            self._fire(u, ability_name, target)
        else:
            u.cast_ability = ability_name
            u.cast_left = cp
            u.cast_target = target
        return True

    # -- effects -----------------------------------------------------------
    def _fire(self, u, ability_name, target):
        ab = u.abilities[ability_name]
        # item actives register under their own names, which contain spaces and
        # apostrophes, so they dispatch through a tag instead of the method name
        tag = ab.get("active")
        fn = getattr(self, f"_item_{tag}", None) if tag else None
        if fn is None:
            fn = getattr(self, f"_fx_{ability_name}", None)
        if fn is None:
            self.note(f"{u.name}: {ability_name} has no implementation")
            return
        fn(u, ab, target)

    # Items ---------------------------------------------------------------
    def _item_blink(self, u, ab, target):
        """Blink Dagger: 1200 units, zero cast point, and it DISJOINTS.

        The disjoint is the point.  A projectile already in flight - a ranged
        attack, a hook - loses its target, which is what makes Blink a
        defensive tool and not merely a gap closer.  Modelling it as pure
        movement would have understated it in exactly the situations the duel
        cares about.
        """
        if not target or target[0] != "point":
            return
        dx, dy = target[1] - u.x, target[2] - u.y
        d = math.hypot(dx, dy)
        if d < 1.0:
            return
        step = min(d, ab["cast_range"])
        u.x += dx / d * step
        u.y += dy / d * step
        # disjoint: anything homing on this unit is dropped
        self.projectiles = [p for p in self.projectiles
                            if p.kind != "attack" or p.source != u.name]

    def _item_hex(self, u, ab, target):
        if not target or target[0] != "unit":
            return
        t = self.u[target[1]]
        self.apply_buff(t, Buff("hex", 2.8,
                                disables=HEX | SILENCE | MUTE | DISARM,
                                move_slow=0.0, source=u.name))
        t.stats_set = getattr(t, "stats_set", None)

    def _item_cyclone(self, u, ab, target):
        if not target or target[0] != "unit":
            return
        t = self.u[target[1]]
        self.apply_buff(t, Buff("cyclone", 2.5, disables=CYCLONE,
                                source=u.name))

    def _item_silence(self, u, ab, target):
        if not target or target[0] != "unit":
            return
        t = self.u[target[1]]
        self.apply_buff(t, Buff("silence", 5.0, disables=SILENCE,
                                source=u.name))

    def _item_stun(self, u, ab, target):
        if not target or target[0] != "unit":
            return
        t = self.u[target[1]]
        # Overwhelm pierces debuff immunity
        self.apply_buff(t, Buff("overwhelm", 1.6, disables=STUN,
                                source=u.name), pierces_immunity=True)

    def _item_blademail(self, u, ab, target):
        self.apply_buff(u, Buff("blade_mail", 5.5, source=u.name))

    # Pudge ---------------------------------------------------------------
    def _fx_meat_hook(self, u, ab, target):
        e = self.other(u)
        if target and target[0] == "point":
            ang = math.atan2(target[2] - u.y, target[1] - u.x)
        else:
            ang = math.atan2(e.y - u.y, e.x - u.x)
        self.projectiles.append(Projectile(
            kind="meat_hook", source=u.name, x=u.x, y=u.y,
            vx=math.cos(ang) * ab["speed"], vy=math.sin(ang) * ab["speed"],
            remaining=ab["travel_distance"], radius=ab["radius"], payload=dict(ab)))

    def _fx_rot(self, u, ab, target):
        u.rot_on = not u.rot_on
        u.cooldowns.pop("rot", None)

    def _fx_meat_shield(self, u, ab, target):
        self.apply_buff(u, Buff("meat_shield", ab["duration"], source=u.name,
                                damage_block=ab["block"]))

    def _fx_dismember(self, u, ab, target):
        t = self.other(u)
        if not t.alive or u.dist_to(t) > ab["cast_range"] + t.stats.get("collision", 0):
            return
        u.channel_ability = "dismember"
        u.channel_left = ab["duration"]
        ab["_accum"] = 0.0
        self.apply_buff(t, Buff("dismember", ab["duration"], disables=STUN,
                                source=u.name), pierces_immunity=True)

    # Invoker -------------------------------------------------------------
    def _fx_cold_snap(self, u, ab, target):
        t = self.u[target[1]] if target and target[0] == "unit" else self.other(u)
        self.apply_buff(t, Buff(
            "cold_snap", ab["duration"], source=u.name,
            proc_on_damage=True, proc_cooldown=ab["proc_interval"],
            proc_stun=ab["stun_per_proc"], proc_damage=ab["damage_per_proc"],
            proc_dtype=ab["damage_type"]))

    def _fx_sun_strike(self, u, ab, target):
        x, y = (target[1], target[2]) if target else (self.other(u).x, self.other(u).y)
        self.effects.append(GroundEffect(
            "sun_strike", u.name, x, y, ab["radius"], ab["delay"], dict(ab)))

    def _fx_chaos_meteor(self, u, ab, target):
        if not target:
            return
        ang = math.atan2(target[2] - u.y, target[1] - u.x)
        e = GroundEffect("chaos_meteor", u.name, u.x, u.y, ab["radius"],
                         ab["impact_delay"], dict(ab))
        e.vx = math.cos(ang) * ab["speed"]
        e.vy = math.sin(ang) * ab["speed"]
        e.remaining_travel = ab["travel_distance"]
        e.duration = ab["travel_distance"] / ab["speed"]
        e.time_left = e.duration
        self.effects.append(e)

    def _fx_emp(self, u, ab, target):
        if not target:
            return
        self.effects.append(GroundEffect(
            "emp", u.name, target[1], target[2], ab["radius"], ab["delay"],
            dict(ab)))

    def _fx_tornado(self, u, ab, target):
        if not target:
            return
        ang = math.atan2(target[2] - u.y, target[1] - u.x)
        p = Projectile("tornado", u.name, u.x, u.y,
                       math.cos(ang) * ab["speed"], math.sin(ang) * ab["speed"],
                       ab["travel_distance"], ab["radius"], dict(ab))
        self.projectiles.append(p)

    def _fx_deafening_blast(self, u, ab, target):
        if not target:
            return
        ang = math.atan2(target[2] - u.y, target[1] - u.x)
        self.projectiles.append(Projectile(
            "deafening", u.name, u.x, u.y,
            math.cos(ang) * ab["speed"], math.sin(ang) * ab["speed"],
            ab["travel_distance"], ab["end_radius"], dict(ab)))

    def _fx_ice_wall(self, u, ab, target):
        if not target:
            return
        ang = math.atan2(target[2] - u.y, target[1] - u.x)
        cx = u.x + math.cos(ang) * 200.0
        cy = u.y + math.sin(ang) * 200.0
        e = GroundEffect("ice_wall", u.name, cx, cy, ab["segments"] * ab["spacing"] / 2,
                         0.0, dict(ab))
        e.duration = ab["duration"]
        e.time_left = ab["duration"]
        e.payload["angle"] = ang + math.pi / 2      # wall runs perpendicular
        e.payload["width"] = ab["width"]
        self.effects.append(e)

    def _fx_alacrity(self, u, ab, target):
        t = self.u[target[1]] if target and target[0] == "unit" else u
        self.apply_buff(t, Buff("alacrity", ab["duration"],
                                bonus_attack_speed=ab["attack_speed"],
                                bonus_attack_damage=ab["damage"], source=u.name))

    def _fx_ghost_walk(self, u, ab, target):
        self.apply_buff(u, Buff("ghost_walk", ab["invisibility"],
                                disables=INVISIBLE, source=u.name,
                                bonus_move_speed_pct=ab["self_speed"]))

    def _fx_forge_spirit(self, u, ab, target):
        """Summon 1-3 Forge Spirits.

        They were previously a no-op that only wrote to the log, which made
        Invoker's armour-shredding option invisible to the search - and that
        option competes directly with the armour-shredding items, so it had to
        exist before builds could be compared.
        """
        # casting kills the previous spirits, as in the game
        self.summons = [x for x in self.summons if x.owner != u.name]
        n = ab.get("count", 1)
        for k in range(n):
            ang = 2 * math.pi * k / max(n, 1)
            self.summons.append(Summon(
                owner=u.name,
                x=u.x + math.cos(ang) * 100.0,
                y=u.y + math.sin(ang) * 100.0,
                hp=ab["hp"], max_hp=ab["hp"], damage=ab["damage"],
                armor=ab["armor"], attack_range=ab["attack_range"],
                bat=1.35, attack_point=0.2, ms=320.0,
                duration=ab["duration"],
                shred_per_stack=ab.get("armor_reduction_per_stack", 0.0),
            ))
        self.note(f"{u.name}: {n} forge spirits summoned")
        # the aura follows Invoker: spirits re-target whatever he fights

    def _tick_summons(self, dt):
        live = []
        for sp in self.summons:
            sp.duration -= dt
            if sp.duration <= 0 or sp.hp <= 0:
                continue
            owner = self.u[sp.owner]
            tgt = self.other(owner)
            if not tgt.alive:
                live.append(sp)
                continue
            d = math.hypot(tgt.x - sp.x, tgt.y - sp.y)
            if sp.attack_cd > 0:
                sp.attack_cd -= dt
            if d > sp.attack_range + tgt.stats.get("collision", 24):
                ang = math.atan2(tgt.y - sp.y, tgt.x - sp.x)
                step = min(sp.ms * dt, d - sp.attack_range)
                if step > 0:
                    sp.x += math.cos(ang) * step
                    sp.y += math.sin(ang) * step
            else:
                sp.windup += dt
                if sp.windup >= sp.attack_point and sp.attack_cd <= 0:
                    sp.windup = 0.0
                    sp.attack_cd = sp.bat
                    # Melting Strike: every hit adds a stack of armour
                    # reduction on the target, capped and with its own timer
                    sp.stacks = min(sp.stacks + 1, sp.max_stacks)
                    sp.stack_time = 5.0
                    for b in tgt.buffs:
                        if b.name == "melting_strike":
                            b.duration = 5.0
                            b.bonus_armor = -sp.shred_per_stack * sp.stacks
                            break
                    else:
                        tgt.buffs.append(Buff(
                            "melting_strike", 5.0,
                            bonus_armor=-sp.shred_per_stack * sp.stacks,
                            source=sp.owner))
                    self.deal(sp, tgt, sp.damage, "physical",
                              label="forge_spirit")
            live.append(sp)
        self.summons = live

    # -- ticking projectiles and effects -----------------------------------
    def _tick_forced_movement(self, dt):
        """Knockback and Dismember's pull move units, they do not just flag them.

        A disable flag alone leaves the target in place, which understates
        every effect whose whole point is displacement: Deafening Blast pushes
        the target out of its own follow-up, Dismember drags it to point blank.
        """
        for u in self.u.values():
            if not u.alive:
                continue
            for b in u.buffs:
                if not b.active():
                    continue
                if b.knockback:
                    src = self.u.get(b.source)
                    if src is not None:
                        ang = math.atan2(u.y - src.y, u.x - src.x)
                        u.x += math.cos(ang) * b.knockback * dt
                        u.y += math.sin(ang) * b.knockback * dt
            if u.channel_ability is None and u.buffs:
                pass
        # Dismember drags the target toward Pudge at 75 u/s, stopping at 125
        p = self.u["pudge"]
        if p.channel_ability == "dismember" and p.alive:
            ab = p.abilities["dismember"]
            t = self.other(p)
            d = p.dist_to(t)
            stop = ab.get("pull_min", 125.0)
            if t.alive and d > stop:
                step = min(ab.get("pull_rate", 75.0) * dt, d - stop)
                ang = math.atan2(t.y - p.y, t.x - p.x)
                t.x += math.cos(ang) * step
                t.y += math.sin(ang) * step

    def _tick_projectiles(self, dt):
        keep = []
        for p in self.projectiles:
            step = math.hypot(p.vx, p.vy) * dt
            if step <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.remaining -= step
            hit = self._projectile_hit(p)
            if hit:
                continue
            if p.remaining <= 0 or not self.inside(p.x, p.y):
                if not self.inside(p.x, p.y):
                    self.note(f"{p.kind} left the arena")
                keep.append(p) if p.remaining > 0 and self.inside(p.x, p.y) else None
                continue
            keep.append(p)
        self.projectiles = keep

    def _projectile_hit(self, p) -> bool:
        src = self.u[p.source]
        tgt = self.other(src)
        if not tgt.alive:
            return False
        # Meat Hook explicitly hits invisible units; the other projectiles here
        # are ground-targeted spells that do not home, so they are unaffected.
        if tgt.is_invisible() and p.kind != "meat_hook":
            return False
        if p.kind == "meat_hook":
            if math.hypot(tgt.x - p.x, tgt.y - p.y) <= p.radius + tgt.stats.get("collision", 24):
                ab = p.payload
                self.deal(src, tgt, ab["damage"], ab["damage_type"],
                          pierces_immunity=ab.get("pierces_immunity", False),
                          label="meat_hook")
                drag = ab["drag_stop"]
                ang = math.atan2(tgt.y - src.y, tgt.x - src.x)
                tgt.x = src.x + math.cos(ang) * drag
                tgt.y = src.y + math.sin(ang) * drag
                stun = ab["travel_distance"] / ab["speed"] * ab.get("self_disable_factor", 0.65)
                self.apply_buff(tgt, Buff("hooked", max(0.4, stun), disables=STUN,
                                          source=src.name), pierces_immunity=True)
                self.note("Meat Hook connected")
                return True
        elif p.kind == "attack":
            if math.hypot(tgt.x - p.x, tgt.y - p.y) <= p.radius:
                self.deal(src, tgt, p.payload["damage"], "physical", label="attack")
                return True
        elif p.kind == "tornado":
            if math.hypot(tgt.x - p.x, tgt.y - p.y) <= p.radius + tgt.stats.get("collision", 24):
                ab = p.payload
                self.deal(src, tgt, ab["damage"], ab["damage_type"], label="tornado")
                self.apply_buff(tgt, Buff("tornado", ab["lift"], disables=CYCLONE,
                                          source=src.name))
                return True
        elif p.kind == "deafening":
            if math.hypot(tgt.x - p.x, tgt.y - p.y) <= p.radius + tgt.stats.get("collision", 24):
                ab = p.payload
                self.deal(src, tgt, ab["damage"], ab["damage_type"], label="deafening")
                self.apply_buff(tgt, Buff("deafening_disarm", ab["disarm"],
                                          disables=DISARM, source=src.name))
                ang = math.atan2(tgt.y - src.y, tgt.x - src.x)
                dist = ab.get("knockback_distance", 200.0)
                self.apply_buff(tgt, Buff("deafening_knockback", ab["knockback_duration"],
                                          disables=KNOCKBACK, source=src.name,
                                          knockback=dist / max(ab["knockback_duration"], 0.1)))
                return True
        return False

    def _tick_effects(self, dt):
        keep = []
        for e in self.effects:
            if e.delay > 0:
                e.delay -= dt
                if e.delay <= 0:
                    self._effect_impact(e)
                if e.time_left > 0:
                    e.time_left -= dt
                keep.append(e)
                continue
            # travelling or persistent
            if e.remaining_travel > 0:
                step = math.hypot(e.vx, e.vy) * dt
                e.x += e.vx * dt
                e.y += e.vy * dt
                e.remaining_travel -= step
                self._effect_tick(e, dt)
            if e.time_left > 0:
                e.time_left -= dt
                if e.time_left <= 0 and e.kind == "ice_wall":
                    continue
            keep.append(e)
        self.effects = keep

    def _effect_impact(self, e):
        src = self.u[e.source]
        tgt = self.other(src)
        if e.kind == "sun_strike":
            ab = e.payload
            if tgt.alive and math.hypot(tgt.x - e.x, tgt.y - e.y) <= e.radius + tgt.stats.get("collision", 24):
                self.deal(src, tgt, ab["damage"], ab["damage_type"],
                          label="sun_strike")
                self.note("Sun Strike hit")
            return
        if e.kind == "emp":
            ab = e.payload
            if tgt.alive and math.hypot(tgt.x - e.x, tgt.y - e.y) <= e.radius:
                burn = min(tgt.mana, ab["mana_burn"])
                tgt.mana -= burn
                self.deal(src, tgt, burn * ab["burn_to_damage"], ab["damage_type"],
                          label="emp")
                src.mana = min(src.max_mana, src.mana + burn * ab["self_restore"])
                self.note("EMP hit")
            return

    def _effect_tick(self, e, dt):
        src = self.u[e.source]
        tgt = self.other(src)
        if not tgt.alive:
            return
        inside = math.hypot(tgt.x - e.x, tgt.y - e.y) <= e.radius
        if e.kind == "chaos_meteor" and inside:
            ab = e.payload
            e.payload.setdefault("_acc", 0.0)
            e.payload["_acc"] += dt
            period = ab["main_interval"]
            while e.payload["_acc"] >= period:
                e.payload["_acc"] -= period
                self.deal(src, tgt, ab["damage_per_interval"], ab["damage_type"],
                          label="chaos_meteor")
        elif e.kind == "ice_wall":
            ab = e.payload
            ang = ab["angle"]
            # distance from the target to the wall line through (e.x, e.y)
            dx, dy = tgt.x - e.x, tgt.y - e.y
            perp = abs(-math.sin(ang) * dx + math.cos(ang) * dy)
            along = abs(math.cos(ang) * dx + math.sin(ang) * dy)
            if perp <= ab["width"] / 2 and along <= e.radius:
                self.apply_buff(tgt, Buff("ice_wall", 0.5, move_slow=ab["slow"],
                                          source=src.name))
                self.deal(src, tgt, ab["dps"] * dt, ab["damage_type"],
                          interrupts=False, label="ice_wall")

    # -- attacks -----------------------------------------------------------
    def _tick_attack(self, u, dt):
        t = self.u.get(u.attack_target) if isinstance(u.attack_target, str) else None
        if t is None or not t.alive or not u.can_attack():
            u.attack_windup = 0.0
            return
        # Attacking needs vision, and an invisible unit needs True Sight,
        # which nothing in this duel grants.
        if not self.visible(u, t):
            u.attack_windup = 0.0
            return
        if u.attack_cooldown > 0 or u.cast_ability is not None or u.channel_ability:
            return
        if u.dist_to(t) > u.attack_range():
            return
        want = math.atan2(t.y - u.y, t.x - u.x)
        diff = (want - u.facing + math.pi) % (2 * math.pi) - math.pi
        max_turn = (u.stats["turn_rate"] / 0.03) * dt
        if abs(diff) > max_turn:
            u.facing += math.copysign(max_turn, diff)
            return
        u.facing = want
        # attack point then damage
        u.attack_windup += dt
        if u.attack_windup < u.stats["attack_point"]:
            return
        u.attack_windup = 0.0
        if u.stats.get("attack_range", 0) > 200.0:
            # Ranged attacks travel.  That matters well beyond cosmetics: a
            # projectile in flight is what Blink Dagger disjoints, so modelling
            # the attack as instant would delete a core use of the item.
            ang = math.atan2(t.y - u.y, t.x - u.x)
            sp = u.stats.get("projectile_speed", 900.0)
            self.projectiles.append(Projectile(
                kind="attack", source=u.name, x=u.x, y=u.y,
                vx=math.cos(ang) * sp, vy=math.sin(ang) * sp,
                remaining=u.dist_to(t) + 100.0, radius=t.stats.get("collision", 24),
                payload=dict(damage=u.attack_damage())))
        else:
            self.deal(u, t, max(0.0, u.attack_damage()), "physical", label="attack")
        # The attack point is the first part of the attack cycle, not extra
        # time on top of it: the next attack may begin one full cycle after
        # this one began, i.e. after (1/aps - attack_point) more seconds.
        cycle = 1.0 / u.attacks_per_second()
        u.attack_cooldown = max(0.0, cycle - u.stats["attack_point"])

    # -- end conditions ----------------------------------------------------
    def _check_end(self):
        if self.winner is not None:
            return
        for u in self.u.values():
            if not u.alive:
                self.winner = self.other(u).name
                self.reason = f"{u.name} died"
                return
            if not self.inside(u.x, u.y):
                self.winner = self.other(u).name
                self.reason = f"{u.name} left the arena"
                self.note(f"{u.name} left the arena")
                return

    # ------------------------------------------------------------------
    # Cloning, so the search can branch
    # ------------------------------------------------------------------
    def clone(self) -> "Duel":
        import copy
        d = Duel.__new__(Duel)
        d.w, d.h = self.w, self.h
        d.t = self.t
        d.winner = self.winner
        d.reason = self.reason
        d.log = []
        d.summons = [copy.deepcopy(x) for x in self.summons]
        d.projectiles = [copy.deepcopy(p) for p in self.projectiles]
        d.effects = [copy.deepcopy(e) for e in self.effects]
        d.u = {}
        for k, u in self.u.items():
            n = Unit.__new__(Unit)
            n.name = u.name
            n.x, n.y, n.facing = u.x, u.y, u.facing
            n.stats = u.stats                       # read-only after build
            n.abilities = copy.deepcopy(u.abilities)
            n.hp, n.mana = u.hp, u.mana
            n.cooldowns = dict(u.cooldowns)
            n.buffs = copy.deepcopy(u.buffs)
            n.alive = u.alive
            n.cast_ability = u.cast_ability
            n.cast_left = u.cast_left
            n.cast_target = u.cast_target
            n.channel_ability = u.channel_ability
            n.channel_left = u.channel_left
            n.attack_target = u.attack_target
            n.attack_cooldown = u.attack_cooldown
            n.attack_windup = u.attack_windup
            n.move_target = u.move_target
            n.rot_on = u.rot_on
            n.items = u.items
            n.spellblock_cd = u.spellblock_cd
            n.aeon_cd = u.aeon_cd
            n.invoked = list(u.invoked)
            if hasattr(u, "orbs"):
                n.orbs = u.orbs
            d.u[k] = n
        return d

    # ------------------------------------------------------------------
    def run(self, policy_pudge, policy_invoker, max_time=120.0,
            decision_dt=0.3):
        """Run the duel.  Each policy is called every `decision_dt` seconds
        with (duel, unit) and sets orders on the unit."""
        next_decision = 0.0
        while self.winner is None and self.t < max_time:
            if self.t >= next_decision - 1e-9:
                if self.u["pudge"].alive:
                    policy_pudge(self, self.u["pudge"])
                if self.u["invoker"].alive:
                    policy_invoker(self, self.u["invoker"])
                next_decision += decision_dt
            self.step(SERVER_TICK)
        return self.winner, self.reason
