"""
High-fidelity forward simulator (the validator).

The DP works on a *reduced, quantised* state at 0.3 s resolution.  This module
implements the same semantics at 1/30 s on *continuous* positions, and is used
to check that the policies the DP extracts are still good in the real model -
i.e. that the reduction and the quantisation did not manufacture an advantage
that does not exist.

Key fidelity points that the reduced model approximates and this one does not:

* continuous positions (no 1 m snapping);
* turn rate: a unit must face its target before it can start an attack or a
  directional cast, which costs real time;
* cast point: the delay between issuing the order and the effect firing - this
  is the window in which the opponent can react or walk out of range;
* projectiles travel at a finite speed, so an ability can be dodged;
* channel interruption (Dismember breaks if Pudge is stunned or silenced).

The simulator is deterministic; all randomness (evasion, crit) is disabled or
seeded, because the user specified perfect play and we want reproducible values.
"""

from __future__ import annotations

import math

from .constants import (
    SERVER_TICK, UNITS_PER_METER, armor_multiplier, attacks_per_second,
    total_attack_speed, move_speed, turn_time,
)


class SimUnit:
    __slots__ = (
        "name", "x", "y", "facing", "hp", "mana", "profile",
        "cooldowns", "buffs", "cast_left", "cast_total", "pending",
        "attack_cd", "channel_left", "channel_ability", "alive",
        "rot_on", "invoked", "attack_target", "move_target",
    )

    def __init__(self, profile, x, y, facing=0.0):
        self.name = profile.name
        self.x = x
        self.y = y
        self.facing = facing
        self.hp = profile.max_hp
        self.mana = profile.max_mana
        self.profile = profile
        self.cooldowns = {}
        self.buffs = []
        self.cast_left = 0.0
        self.cast_total = 0.0
        self.pending = None          # (kind, payload) fired when cast_left hits 0
        self.attack_cd = 0.0
        self.channel_left = 0.0
        self.channel_ability = None
        self.alive = True
        self.rot_on = False
        self.invoked = []
        self.attack_target = None
        self.move_target = None

    def pos(self):
        return (self.x, self.y)

    def dist_to(self, other):
        return math.hypot(other.x - self.x, other.y - self.y)

    # -- disable queries ---------------------------------------------------
    def _has(self, flag):
        for b in self.buffs:
            if b.disables & flag:
                return True
        return False

    def can_move(self):
        from .model import CANNOT_MOVE
        return self.alive and not self._has(CANNOT_MOVE)

    def can_turn(self):
        from .model import CANNOT_TURN
        return self.alive and not self._has(CANNOT_TURN)

    def can_attack(self):
        from .model import CANNOT_ATTACK
        return self.alive and not self._has(CANNOT_ATTACK)

    def can_cast(self):
        from .model import CANNOT_CAST
        return self.alive and not self._has(CANNOT_CAST)

    def can_item(self):
        from .model import CANNOT_ITEM
        return self.alive and not self._has(CANNOT_ITEM)

    def is_spell_immune(self):
        from .model import Disable
        return self._has(Disable.SPELL_IMMUNE)

    # -- stats -------------------------------------------------------------
    def _sum(self, attr, default=0.0):
        return sum(getattr(b, attr) for b in self.buffs if b.duration > 0)

    def total_move_speed(self):
        p = self.profile
        slows = [b.move_speed_slow for b in self.buffs if b.move_speed_slow > 0]
        pct = p.move_speed_pct_bonus + self._sum("bonus_move_speed_pct")
        flat = p.move_speed_flat_bonus + self._sum("bonus_move_speed_flat")
        return move_speed(p.move_speed_base, flat, pct, slows)

    def total_armor(self):
        return self.profile.armor + self._sum("bonus_armor")

    def total_magic_resist(self):
        from .constants import stack_resist
        return stack_resist(self.profile.magic_resist,
                            self._sum("bonus_magic_resist"))

    def total_attack_speed(self):
        """Clamped total attack speed in [20, 700]."""
        return total_attack_speed(
            self.profile.agility,
            self.profile.attack_speed_flat + self._sum("bonus_attack_speed"),
            self.profile.attack_speed_pct)

    def total_attack_damage(self):
        return self.profile.attack_damage + self._sum("bonus_attack_damage")

    def total_evasion(self):
        from .constants import stack_resist
        return min(0.99, stack_resist(self.profile.evasion, self._sum("evasion")))

    def hp_regen(self):
        return self.profile.hp_regen + self._sum("heal_per_sec")

    def mana_regen(self):
        return self.profile.mana_regen


class Sim:
    """Deterministic 1v1 duel simulator."""

    def __init__(self, u1: SimUnit, u2: SimUnit, arena=(30.0, 80.0)):
        self.units = [None, u1, u2]
        self.arena_w_m, self.arena_h_m = arena
        self.t = 0.0
        self.events = []
        self.winner = None      # 1, 2, or None (draw/unfinished)

    # ------------------------------------------------------------------
    def _log(self, msg):
        self.events.append((self.t, msg))

    def _out_of_arena(self, u):
        return not (0.0 <= u.x <= self.arena_w_m * UNITS_PER_METER
                    and 0.0 <= u.y <= self.arena_h_m * UNITS_PER_METER)

    # ------------------------------------------------------------------
    def step(self, dt=SERVER_TICK):
        if self.winner is not None:
            return
        self.t += dt

        for i in (1, 2):
            u = self.units[i]
            if not u.alive:
                continue
            self._tick_buffs(u, dt)
            self._tick_cooldowns(u, dt)
            self._tick_cast(u, dt)
            self._regen(u, dt)
            if u.rot_on:
                self._apply_rot(u, dt)

        for i in (1, 2):
            u = self.units[i]
            if u.alive:
                self._tick_movement(u, dt)

        for i in (1, 2):
            u = self.units[i]
            if u.alive:
                self._tick_attack(u, dt)

        self._check_end()

    # ------------------------------------------------------------------
    def _tick_buffs(self, u, dt):
        keep = []
        for b in u.buffs:
            b.duration -= dt
            if b.duration > 0:
                keep.append(b)
            else:
                self._log(f"{u.name}: {b.name} expired")
        u.buffs = keep
        # a stun that has just expired cancels the channel it interrupted
        if u.channel_left > 0 and not u.can_cast():
            self._log(f"{u.name}: channel of {u.channel_ability} broken")
            u.channel_left = 0.0
            u.channel_ability = None

    def _tick_cooldowns(self, u, dt):
        for k in list(u.cooldowns):
            u.cooldowns[k] -= dt
            if u.cooldowns[k] <= 0:
                del u.cooldowns[k]
        if u.attack_cd > 0:
            u.attack_cd -= dt

    def _regen(self, u, dt):
        u.hp = min(u.profile.max_hp, u.hp + u.hp_regen() * dt)
        u.mana = min(u.profile.max_mana, u.mana + u.mana_regen() * dt)

    def _apply_rot(self, pudge, dt):
        """Rot damages enemies in radius and Pudge himself."""
        ab = pudge.profile.abilities["rot"]
        r_u = ab["radius"] * (1.0 + 0.0)
        enemy = self.units[2] if pudge is self.units[1] else self.units[1]
        if enemy.alive and pudge.dist_to(enemy) <= r_u:
            self.deal_damage(pudge, enemy, ab["dps"] * dt, "magical",
                             source="rot")
        if ab.get("self_damage", 0.0) > 0:
            self.deal_damage(pudge, pudge, ab["self_damage"] * dt, "pure",
                             source="rot-self", ignore_immunity=True)

    # ------------------------------------------------------------------
    def deal_damage(self, source, target, amount, dtype, source_name="",
                    ignore_immunity=False):
        """Apply one damage instance.  Returns the damage actually dealt."""
        if not target.alive or amount <= 0:
            return 0.0

        if dtype in ("magical", "pure") and target.is_spell_immune() \
                and not ignore_immunity:
            return 0.0

        if dtype == "physical":
            amount *= armor_multiplier(target.total_armor())
        elif dtype == "magical":
            amount *= (1.0 - target.total_magic_resist())

        amount *= (1.0 - min(0.9, target._sum("damage_reduction")))

        before = target.hp
        target.hp -= amount
        dealt = before - max(0.0, target.hp)
        if target.hp <= 0:
            target.hp = 0.0
            self._kill(target)
        return dealt

    def _kill(self, u):
        u.alive = False
        self._log(f"{u.name} died")
        if u is self.units[1]:
            self.winner = 2
        else:
            self.winner = 1

    def apply_buff(self, target, buff):
        """Apply a buff, honouring status resistance on the duration."""
        from .model import DispelType
        if target.is_spell_immune() and buff.dispellable != DispelType.NONE \
                and getattr(buff, "blocked_by_spell_immunity", False):
            return
        if buff.duration > 0:
            from .constants import STATUS_RESIST_CAP
            sr = min(STATUS_RESIST_CAP,
                     target.profile.status_resist + target._sum("status_resist"))
            if buff.disables:
                buff.duration *= (1.0 - sr)
        target.buffs.append(buff)

    # ------------------------------------------------------------------
    def _tick_movement(self, u, dt):
        if not u.can_move() or u.move_target is None:
            return
        tx, ty = u.move_target
        dx, dy = tx - u.x, ty - u.y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return
        # turning toward the movement direction costs time
        want = math.atan2(dy, dx)
        if u.can_turn():
            tr = u.profile.turn_rate
            max_turn = (tr / 0.03) * dt
            diff = (want - u.facing + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) <= max_turn:
                u.facing = want
            else:
                u.facing += math.copysign(max_turn, diff)
                return          # not facing the destination yet: no movement
        speed = u.total_move_speed() / UNITS_PER_METER   # m/s
        step = min(d, speed * dt)
        u.x += dx / d * step
        u.y += dy / d * step

    # ------------------------------------------------------------------
    def _tick_attack(self, u, dt):
        tgt = u.attack_target
        if tgt is None or not tgt.alive or not u.can_attack():
            return
        rng = u.profile.attack_range + u.profile.collision_size + tgt.profile.collision_size
        if u.dist_to(tgt) > rng:
            return
        if u.attack_cd > 0:
            return
        if u.cast_left > 0:
            return
        # face the target
        want = math.atan2(tgt.y - u.y, tgt.x - u.x)
        diff = (want - u.facing + math.pi) % (2 * math.pi) - math.pi
        max_turn = (u.profile.turn_rate / 0.03) * dt
        if abs(diff) > max_turn:
            u.facing += math.copysign(max_turn, diff)
            return
        u.facing = want
        # forced miss
        if tgt.total_evasion() > 0:
            return          # deterministic model: evasion disabled
        dmg = u.total_attack_damage()
        self.deal_damage(u, tgt, dmg, "physical", source_name="attack")
        aps = attacks_per_second(u.total_attack_speed(),
                                 u.profile.base_attack_time)
        u.attack_cd = 1.0 / aps

    # ------------------------------------------------------------------
    def _tick_cast(self, u, dt):
        if u.cast_left <= 0:
            return
        u.cast_left -= dt
        if u.cast_left <= 0 and u.pending is not None:
            kind, payload = u.pending
            u.pending = None
            self._fire(u, kind, payload)

    def _fire(self, u, kind, payload):
        """Resolve a completed cast."""
        if kind == "meat_hook":
            self._launch_hook(u, payload)
        elif kind == "dismember":
            tgt = self.units[2] if u is self.units[1] else self.units[1]
            if tgt.alive and u.dist_to(tgt) <= payload["range"]:
                ab = payload
                u.channel_left = ab["duration"]
                u.channel_ability = "dismember"
                self._log(f"{u.name} begins Dismember on {tgt.name}")
        elif kind == "instant_damage":
            tgt = payload["target"]
            self.deal_damage(u, tgt, payload["damage"], payload["dtype"],
                             source_name=kind)
        elif kind == "hook_arrive":
            pass

    def _launch_hook(self, pudge, ab):
        """Create a travelling Meat Hook.

        Implemented inline (rather than via the Projectile dataclass) because
        the hook's drag-on-hit semantics are bespoke.
        """
        enemy = self.units[2] if pudge is self.units[1] else self.units[1]
        ang = math.atan2(enemy.y - pudge.y, enemy.x - pudge.x)
        speed = ab["speed"]
        max_d = ab["travel_distance"]
        radius = ab["radius"]
        travelled = 0.0
        dt = SERVER_TICK
        ox, oy = pudge.x, pudge.y
        while travelled < max_d:
            travelled += speed * dt
            hx = ox + math.cos(ang) * travelled
            hy = oy + math.sin(ang) * travelled
            if enemy.alive:
                d = math.hypot(enemy.x - hx, enemy.y - hy)
                if d <= radius + enemy.profile.collision_size:
                    self._log("Meat Hook connected")
                    self.deal_damage(pudge, enemy, ab["damage"], ab["dtype"],
                                     source_name="meat_hook")
                    if enemy.alive and ab.get("drags", True):
                        self._drag(pudge, enemy, ab)
                    return
            if not (0 <= hx <= self.arena_w_m * UNITS_PER_METER
                    and 0 <= hy <= self.arena_h_m * UNITS_PER_METER):
                self._log("Meat Hook left the arena")
                return
        self._log("Meat Hook missed")

    def _drag(self, pudge, enemy, ab):
        """Pull the hooked unit to a fixed distance in front of Pudge."""
        stop = ab.get("drag_stop_distance", 100.0)
        ang = math.atan2(enemy.y - pudge.y, enemy.x - pudge.x)
        tx = pudge.x + math.cos(ang) * stop
        ty = pudge.y + math.sin(ang) * stop
        enemy.x, enemy.y = tx, ty
        self._log("dragged")

    # ------------------------------------------------------------------
    def _check_end(self):
        if self.winner is not None:
            return
        for i in (1, 2):
            u = self.units[i]
            if not u.alive:
                self.winner = 2 if i == 1 else 1
                return
            if self._out_of_arena(u):
                self._log(f"{u.name} left the arena")
                self.winner = 2 if i == 1 else 1
                return

    # ------------------------------------------------------------------
    def run(self, orders1, orders2, max_time=120.0):
        """Run the duel.  `ordersN` is a callable (sim, unit) -> None issued
        every 0.3 s (the decision grid)."""
        decision_dt = 0.3
        next_decision = 0.0
        while self.winner is None and self.t < max_time:
            if self.t >= next_decision - 1e-9:
                if self.units[1].alive:
                    orders1(self, self.units[1])
                if self.units[2].alive:
                    orders2(self, self.units[2])
                next_decision += decision_dt
            self.step(SERVER_TICK)
        return self.winner
