"""
Search over the forward model.

Tabulation is impossible for a real Invoker (the cooldown vector alone is
1.6e17 states, and the reachable subset is nearly as large), so the decision is
made by search instead:

    value(node) = Val_{a in A1, b in A2} [ value(f(node, a, b)) ]

with `Val` the value of the stage matrix game (the moves are simultaneous) and
a leaf evaluator closing the recursion.

The leaf is a **rollout**, not a formula: play both sides out to the end with a
fixed policy and return the outcome.  That keeps the leaf honest - it is
evaluated by the exact rules rather than by a hand-written approximation - and
the rollout policy can be improved by policy iteration, using the search's own
output as the next rollout policy.

Cost is |A1| x |A2| rollouts per node, so depths stay shallow and the action
set is pruned.  That is the right trade: the mechanics are exact, only the
policy at the leaves is approximate.
"""

from __future__ import annotations

import math
import time as _time

from dota2sim.constants import UNITS_PER_METER
from .matrixgame import solve as solve_matrix

#: each hero issues orders every 250 ms - the grid the user asked for,
#: matching average human reaction time
DECISION_DT = 0.25
#: alternating plies.  Each hero still acts once per DECISION_DT, but the two
#: are offset by half a grid, which is what lets alpha-beta apply: with truly
#: simultaneous moves the recursion is a matrix game at every node and there is
#: no bound to prune against.
PLY_DT = 0.25          # the user fixed the HALF-step at 250 ms, so a full
                       # round (both heroes acting) is 500 ms and "6 steps each"
                       # is 12 plies = 3 s of game time


# --------------------------------------------------------------------------
# Action representation
# --------------------------------------------------------------------------
class Action:
    """Movement plus at most one ability, both issued at the same instant."""

    __slots__ = ("mvx", "mvy", "ability", "target", "attack", "rot")

    def __init__(self, mvx=0.0, mvy=0.0, ability=None, target=None,
                 attack=False, rot=None):
        self.mvx = mvx          # metres, unit direction * 1
        self.mvy = mvy
        self.ability = ability
        self.target = target    # "enemy" | "self" | "point_enemy"
        self.attack = attack
        self.rot = rot          # None = leave alone, True/False = toggle

    def key(self):
        return (round(self.mvx, 3), round(self.mvy, 3), self.ability,
                self.target, self.attack, self.rot)

    def __repr__(self):
        bits = []
        if self.mvx or self.mvy:
            bits.append(f"mv({self.mvx:+.2f},{self.mvy:+.2f})")
        if self.ability:
            bits.append(self.ability)
        if self.attack:
            bits.append("attack")
        if self.rot is not None:
            bits.append("rot_on" if self.rot else "rot_off")
        return "+".join(bits) or "idle"


#: eight compass directions plus standing still
DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
        (0, 0)]


def _dirs():
    out = []
    for dx, dy in DIRS:
        n = math.hypot(dx, dy) or 1.0
        out.append((dx / n, dy / n))
    return out


UNIT_DIRS = _dirs()


def _castable(u, name):
    ab = u.abilities.get(name)
    if ab is None:
        return False
    if u.cooldowns.get(name, 0.0) > 0:
        return False
    return u.mana >= ab.get("mana", 0.0)


#: which abilities each side bothers to consider.  Everything else is either
#: irrelevant to a duel or dominated by one of these.
PUDGE_KIT = ["meat_hook", "dismember"]
INVOKER_KIT = ["cold_snap", "sun_strike", "chaos_meteor", "emp", "tornado",
               "deafening_blast", "ice_wall", "alacrity", "forge_spirit",
               "ghost_walk"]


def actions_for(duel, who: str, kit, moves=None, allow_attack=True):
    """Enumerate (movement, ability) pairs.

    Movement is relative to the enemy - toward, away, strafe, hold - because
    absolute compass directions are mostly redundant in a duel.  Rot appears
    as an explicit choice for Pudge: whether to keep burning is a real decision
    here, since his own Rot ticks are what feed Cold Snap's procs.
    """
    u = duel.u[who]
    e = duel.other(u)
    if moves is None:
        moves = relative_dirs(u, e)
    rots = [None] if who != "pudge" else [None, True, False]
    out = []
    for mx, my in moves:
      for rot in rots:
        out.append(Action(mx, my, rot=rot))
        if allow_attack:
            out.append(Action(mx, my, attack=True, rot=rot))
        for name in kit:
            if not _castable(u, name):
                continue
            ab = u.abilities[name]
            tgt = ab.get("target", "point")
            if tgt in ("self",):
                out.append(Action(mx, my, name, "self", rot=rot))
            elif tgt == "unit":
                if u.dist_to(e) <= ab.get("cast_range", 1e9) + 100:
                    out.append(Action(mx, my, name, "enemy", rot=rot))
            elif tgt == "unit_ally":
                out.append(Action(mx, my, name, "self", rot=rot))
            else:                                    # point / global
                rng = ab.get("cast_range", 0.0)
                if rng <= 0 or u.dist_to(e) <= rng + 100:
                    out.append(Action(mx, my, name, "point_enemy", rot=rot))
    return out


def relative_dirs(u, e):
    """Five movement options relative to the enemy, plus standing still."""
    ang = math.atan2(e.y - u.y, e.x - u.x)
    rel = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (0.0, 0.0)]
    out = []
    for fwd, side in rel:
        a = ang if fwd > 0 else (ang + math.pi if fwd < 0 else ang + math.pi / 2)
        if fwd == 0 and side == 0:
            out.append((0.0, 0.0))
            continue
        aa = a + (math.pi / 2 if side > 0 else (-math.pi / 2 if side < 0 else 0))
        out.append((math.cos(aa), math.sin(aa)))
    return out


def apply_action(duel, who: str, a: Action):
    """Issue `a` for `who`.  Movement is a direction; the simulator handles
    speed and turning."""
    u = duel.u[who]
    e = duel.other(u)
    if a.mvx or a.mvy:
        speed = u.move_speed()
        dist = speed * DECISION_DT * 3.0           # aim a little beyond one step
        u.move_target = (u.x + a.mvx * dist, u.y + a.mvy * dist)
    else:
        u.move_target = None
    if a.attack:
        u.attack_target = e.name
    else:
        u.attack_target = None
    if a.rot is not None and who == "pudge":
        u.rot_on = a.rot
    if a.ability:
        if a.target == "self":
            duel.try_cast(u, a.ability, ("unit", u.name))
        elif a.target == "enemy":
            duel.try_cast(u, a.ability, ("unit", e.name))
        else:
            duel.try_cast(u, a.ability, ("point", e.x, e.y))


def advance(duel, dt=DECISION_DT):
    """Run the simulator forward one decision step."""
    n = max(1, int(round(dt / (1.0 / 30.0))))
    for _ in range(n):
        duel.step(1.0 / 30.0)
        if duel.winner is not None:
            break


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------
def default_policy(duel, who: str, aggressive=True):
    """A fixed, sensible policy used to evaluate leaves.

    Pudge closes and uses his kit; Invoker kites and casts.  Deliberately
    simple - its job is to be a stable baseline, and policy iteration replaces
    it with something better once the search has an opinion.
    """
    u = duel.u[who]
    e = duel.other(u)
    if not u.alive:
        return
    d = u.dist_to(e)
    ang = math.atan2(e.y - u.y, e.x - u.x)
    if who == "pudge":
        u.attack_target = e.name
        u.move_target = (e.x, e.y)
        # Rot is a liability while Cold Snap is on him: his own Rot ticks are
        # damage instances and therefore feed Cold Snap's procs, which is how
        # most of the damage he takes in this matchup arrives.  Burning while
        # stunned is strictly self-harm, so the policy switches it off and
        # back on once the debuff drops.
        cold_snapped = any(b.name == "cold_snap" for b in u.buffs if b.active())
        u.rot_on = not cold_snapped
        if d <= u.abilities["dismember"]["cast_range"] + 100:
            if duel.try_cast(u, "dismember", ("unit", e.name)):
                return
        if d <= u.abilities["meat_hook"]["cast_range"]:
            duel.try_cast(u, "meat_hook", ("point", e.x, e.y))
    else:
        # Kite, but never into the wall: the arena is finite and stepping out
        # is an instant loss, so a policy that merely runs away is suicidal.
        # The target is clamped inside, which turns "flee" into "flee along
        # the wall" once the wall is reached.
        u.attack_target = e.name
        tx = u.x + math.cos(ang) * -600.0
        ty = u.y + math.sin(ang) * -600.0
        margin = 2.0 * UNITS_PER_METER
        tx = min(max(tx, margin), duel.w - margin)
        ty = min(max(ty, margin), duel.h - margin)
        u.move_target = (tx, ty)
        for name in ("cold_snap", "chaos_meteor", "sun_strike", "emp",
                     "deafening_blast", "tornado"):
            ab = u.abilities.get(name)
            if ab is None or u.cooldowns.get(name, 0.0) > 0 or u.mana < ab.get("mana", 0):
                continue
            tgt = ab.get("target")
            if tgt == "unit":
                if duel.try_cast(u, name, ("unit", e.name)):
                    return
            else:
                rng = ab.get("cast_range", 0.0)
                if rng <= 0 or d <= rng:
                    if duel.try_cast(u, name, ("point", e.x, e.y)):
                        return


def rollout(duel, max_time=30.0, policy=None):
    """Play the duel out with `policy`; return Pudge's payoff."""
    policy = policy or default_policy
    next_decision = duel.t
    while duel.winner is None and duel.t < max_time:
        if duel.t >= next_decision - 1e-9:
            if duel.u["pudge"].alive:
                policy(duel, "pudge")
            if duel.u["invoker"].alive:
                policy(duel, "invoker")
            next_decision += DECISION_DT
        duel.step(1.0 / 30.0)
    if duel.winner is None:
        return 0.5
    return 1.0 if duel.winner == "pudge" else 0.0


def greedy_vf_policy(vf, eps=0.0, rng=None):
    """A rollout policy that is greedy with respect to the value function.

    This is what makes long-horizon play reachable without deep tree search.
    A rollout runs to the end of the duel, so it sees a Tornado lift followed
    by a meteor landing several seconds later - a sequence a depth-1 or even
    depth-6 tree would have to unroll ply by ply and could not afford.
    """
    def policy(duel, who):
        u = duel.u[who]
        if not u.alive:
            return
        kit = PUDGE_KIT if who == "pudge" else INVOKER_KIT
        best, best_v = None, -1e9
        for a in actions_for(duel, who, kit):
            d2 = duel.clone()
            apply_action(d2, who, a)
            advance(d2, dt=DECISION_DT)
            v = vf(d2)
            if who == "invoker":
                v = 1.0 - v
            if v > best_v:
                best, best_v = a, v
        if best is not None:
            apply_action(duel, who, best)
    return policy


def make_leaf(policy=None, max_time=30.0):
    def leaf(duel):
        return rollout(duel, max_time=max_time, policy=policy)
    return leaf


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------
def state_key(duel):
    """Hash of everything that determines the future.  The game is stationary,
    so wall-clock time is not part of it - only the cooldowns, buffs and
    positions are.  Same state reached by different move orders is the same
    position, which is what makes a transposition table pay."""
    parts = []
    for who in ("pudge", "invoker"):
        u = duel.u[who]
        parts.append((
            round(u.x, 1), round(u.y, 1), round(u.hp, 1), round(u.mana, 1),
            u.alive, u.rot_on,
            tuple(sorted((k, round(v, 2)) for k, v in u.cooldowns.items())),
            tuple(sorted((b.name, round(b.duration, 2), b.disables)
                         for b in u.buffs if b.active())),
        ))
    return (tuple(parts),
            tuple(sorted((e.kind, round(e.x, 1), round(e.y, 1),
                          round(e.delay, 2), round(e.time_left, 2))
                         for e in duel.effects)),
            tuple(sorted((p.kind, round(p.x, 1), round(p.y, 1))
                         for p in duel.projectiles)))


def prune_dominated(A, m, n):
    """Remove strictly dominated rows and columns from a payoff matrix.

    Row i is dominated if some other row is <= it everywhere (the row player
    maximises); column j likewise.  Dominance is exact - a dominated action can
    never be part of an optimal mixed strategy - so this is a free reduction.
    """
    def keep_rows():
        keep = []
        for i in range(m):
            base = i * n
            dominated = False
            for k in range(m):
                if k == i:
                    continue
                kb = k * n
                if all(A[kb + j] <= A[base + j] + 1e-12 for j in range(n)) and \
                   any(A[kb + j] < A[base + j] - 1e-12 for j in range(n)):
                    dominated = True
                    break
            keep.append(i) if not dominated else None
        return keep or list(range(m))

    def keep_cols(rows):
        keep = []
        for j in range(n):
            dominated = False
            for l in range(n):
                if l == j:
                    continue
                if all(A[i * n + l] >= A[i * n + j] - 1e-12 for i in rows) and \
                   any(A[i * n + l] > A[i * n + j] + 1e-12 for i in rows):
                    dominated = True
                    break
            keep.append(j) if not dominated else None
        return keep or list(range(n))

    rows = keep_rows()
    return rows, keep_cols(rows)


def value(duel, depth, leaf, acts_p=None, acts_i=None, stats=None,
          tt=None, prune=True):
    """Depth-limited simultaneous-move search with a transposition table.

    Unlike chess there is no alpha-beta here: the moves are simultaneous, so
    the recursion is a matrix game at every node and the branching multiplies
    on each ply.  What is available instead is
      * a transposition table - the game is deterministic, so repeated
        positions collapse onto one entry, and
      * exact dominance pruning of the stage matrix, which a simultaneous game
        admits and an alternating one does not use.
    Together they buy a few plies, not ten.
    """
    if duel.winner is not None:
        return 1.0 if duel.winner == "pudge" else 0.0
    if depth <= 0:
        return leaf(duel)

    key = None
    if tt is not None:
        key = state_key(duel)
        hit = tt.get(key)
        if hit is not None:
            if stats is not None:
                stats["tt_hits"] = stats.get("tt_hits", 0) + 1
            return hit

    Ap = acts_p if acts_p is not None else actions_for(duel, "pudge", PUDGE_KIT)
    Ai = acts_i if acts_i is not None else actions_for(duel, "invoker", INVOKER_KIT)
    if not Ap or not Ai:
        return leaf(duel)

    m, n = len(Ap), len(Ai)
    A = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            d2 = duel.clone()
            apply_action(d2, "pudge", Ap[i])
            apply_action(d2, "invoker", Ai[j])
            advance(d2)
            A[i * n + j] = value(d2, depth - 1, leaf, stats=stats, tt=tt,
                                 prune=prune)
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + m * n

    si, sj = list(range(m)), list(range(n))
    if prune and depth > 1 and m > 1 and n > 1:
        si, sj = prune_dominated(A, m, n)
        if stats is not None:
            stats["pruned"] = stats.get("pruned", 0) + (m * n - len(si) * len(sj))
    sub = [A[i * n + j] for i in si for j in sj]
    val, p, q, _ = solve_matrix(sub, len(si), len(sj))
    if stats is not None:
        stats["root_p"] = p
        stats["root_q"] = q
        stats["root_actions"] = ([Ap[i] for i in si], [Ai[j] for j in sj])
    if tt is not None and key is not None:
        tt[key] = val
    return val


def _old_value(duel, depth, leaf, acts_p=None, acts_i=None, stats=None):
    """Depth-limited simultaneous-move search."""
    if duel.winner is not None:
        return 1.0 if duel.winner == "pudge" else 0.0
    if depth <= 0:
        return leaf(duel)

    Ap = acts_p if acts_p is not None else actions_for(duel, "pudge", PUDGE_KIT)
    Ai = acts_i if acts_i is not None else actions_for(duel, "invoker", INVOKER_KIT)
    if not Ap or not Ai:
        return leaf(duel)

    m, n = len(Ap), len(Ai)
    A = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            d2 = duel.clone()
            apply_action(d2, "pudge", Ap[i])
            apply_action(d2, "invoker", Ai[j])
            advance(d2)
            A[i * n + j] = value(d2, depth - 1, leaf, stats=stats)
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + m * n
    val, p, q, _ = solve_matrix(A, m, n)
    if stats is not None:
        stats["root_p"], stats["root_q"] = p, q
        stats["root_actions"] = (Ap, Ai)
    return val


def best_response(duel, who: str, leaf, stats=None):
    """Value of the position for `who`, searching only his own choices while
    the opponent plays the leaf policy.  Cheap alternative to a full game
    solve when only one side is being optimised."""
    me, other = (("pudge", "invoker") if who == "pudge" else ("invoker", "pudge"))
    Ap = actions_for(duel, me, PUDGE_KIT if who == "pudge" else INVOKER_KIT)
    best, best_a = None, None
    for a in Ap:
        d2 = duel.clone()
        apply_action(d2, me, a)
        default_policy(d2, other)
        advance(d2)
        v = value(d2, 0, leaf)
        if who == "invoker":
            v = 1.0 - v
        if best is None or v > best:
            best, best_a = v, a
    if stats is not None:
        stats["best_action"] = best_a
    return best, best_a

# --------------------------------------------------------------------------
# Alternating-play search with alpha-beta
# --------------------------------------------------------------------------
def order_actions(duel, who, acts, vf=None, ply_dt=PLY_DT, cheap=True,
                  hint=None):
    """Sort candidate actions so that alpha-beta cuts early.

    This is move ORDERING, not move selection.  Alpha-beta returns exactly the
    same value whatever the order - only the number of branches that can be cut
    depends on it, and with a good order the tree costs b^(d/2) instead of b^d.
    Nothing is discarded: a move that looks bad one ply ahead is still
    searched, and if it wins in five plies the recursion finds that, because
    the value returned comes from the recursion and never from this estimate.

    (Discarding on this score would be the greedy beam search instead, and it
    would silently delete the very setups we are trying to price - a Tornado
    lift looks bad one ply ahead and good three plies later.)

    The estimate only needs to correlate with the true value.  The first
    version evaluated every candidate by cloning the duel and running a ply,
    which cost ~5 ms per node and was paid at EVERY visited node - pruning does
    not save ordering work - so depth 4 alone took 17 s of sorting.  A static
    priority is nearly free, and being cheap matters far more than being
    accurate: a mediocre order that costs nothing beats a good one that
    dominates the runtime.
    """
    if not cheap and vf is not None:
        scored = []
        for a in acts:
            d2 = duel.clone()
            apply_action(d2, who, a)
            advance(d2, dt=ply_dt)
            v = vf(d2)
            scored.append((v if who == "pudge" else -v, a))
        scored.sort(key=lambda x: -x[0])
        return [a for _, a in scored]

    u = duel.u[who]
    e = duel.other(u)
    d_units = u.dist_to(e)

    def priority(a):
        p = 0.0
        if a.ability:
            ab = u.abilities.get(a.ability, {})
            rng = ab.get("cast_range", 0.0)
            # an ability that can actually reach the target is worth trying
            if rng <= 0 or d_units <= rng + 100:
                p += 10.0
            if ab.get("damage") or ab.get("damage_per_proc") or ab.get("dps"):
                p += 5.0
            if ab.get("disables") or ab.get("stun_per_proc"):
                p += 4.0
        if a.attack:
            p += 3.0 if d_units <= u.attack_range() else 0.0
        if who == "pudge":
            # closing is Pudge's plan; opening up is Invoker's
            p += (3.0 * (-a.mvx * (e.x - u.x) - a.mvy * (e.y - u.y))
                  / max(1.0, d_units))
        else:
            p += (3.0 * (a.mvx * (e.x - u.x) + a.mvy * (e.y - u.y))
                  / max(1.0, d_units))
        return p

    ranked = sorted(acts, key=priority, reverse=True)
    if hint is not None:
        # the best move from a shallower search goes first: an iterative
        # deepening pass turns a mediocre static order into a near-perfect one,
        # and ordering quality is the whole of alpha-beta's speedup
        hk = hint.key()
        ranked.sort(key=lambda x: 0 if x.key() == hk else 1)
    return ranked


def alpha_beta(duel, depth, leaf, turn="pudge", alpha=-1e9, beta=1e9,
               vf=None, tt=None, stats=None, ply_dt=PLY_DT, kit=None):
    """Minimax with alpha-beta over alternating plies.

    `turn` is the hero to move at this ply.  Returns Pudge's win probability.
    """
    if duel.winner is not None:
        return 1.0 if duel.winner == "pudge" else 0.0
    if depth <= 0:
        return leaf(duel)

    key = None
    key = None
    hint = None
    if tt is not None:
        key = (state_key(duel), depth, turn)
        hit = tt.get(key)
        if hit is not None:
            _, _, val, hint = hit
            if hit[0] <= alpha and hit[1] >= beta:
                if stats is not None:
                    stats["tt_hits"] = stats.get("tt_hits", 0) + 1
                return val
        # a shallower entry for the same position still gives a good first move
        if hint is None:
            for dd in range(depth - 1, 0, -1):
                alt = tt.get((state_key(duel), dd, turn))
                if alt is not None:
                    hint = alt[3]
                    break

    who = turn
    acts = actions_for(duel, who, kit if kit is not None
                       else (PUDGE_KIT if who == "pudge" else INVOKER_KIT))
    if not acts:
        return leaf(duel)
    acts = order_actions(duel, who, acts, vf, ply_dt, cheap=True, hint=hint)

    nxt = "invoker" if who == "pudge" else "pudge"
    maximizing = (who == "pudge")
    best = -1e9 if maximizing else 1e9
    a_best = alpha
    b_best = beta
    a_bestmove = None
    for a in acts:
        d2 = duel.clone()
        apply_action(d2, who, a)
        advance(d2, dt=ply_dt)
        v = alpha_beta(d2, depth - 1, leaf, nxt, alpha, beta, vf, tt, stats,
                       ply_dt, kit)
        if stats is not None:
            stats["nodes"] = stats.get("nodes", 0) + 1
        if maximizing:
            if v > best:
                best = v
                a_bestmove = a
            if best > alpha:
                alpha = best
        else:
            if v < best:
                best = v
                a_bestmove = a
            if best < beta:
                beta = best
        if alpha >= beta:
            if stats is not None:
                stats["cutoffs"] = stats.get("cutoffs", 0) + 1
            break
    if tt is not None and key is not None:
        tt[key] = (a_best, b_best, best, a_bestmove)
    return best


def search_iterative(duel, max_depth, leaf, vf=None, stats=None, ply_dt=PLY_DT,
                     kit=None, verbose=False):
    """Iterative deepening: search 2, 4, ... plies, feeding each result's best
    move forward as the ordering hint for the next.  Each pass is cheap next to
    the last, and the hint makes the deeper pass prune far harder."""
    tt = {}
    st = stats if stats is not None else {}
    val = None
    for depth in range(2, max_depth + 1, 2):
        t0 = _time.time()
        val = alpha_beta(duel, depth, leaf, "pudge", -1e9, 1e9, vf=vf, tt=tt,
                         stats=st, ply_dt=ply_dt, kit=kit)
        if verbose:
            print(f"    глубина {depth}: V={val:.3f}  узлов {st.get('nodes',0)}"
                  f"  отсечений {st.get('cutoffs',0)}  {_time.time()-t0:.2f}с",
                  flush=True)
    return val, st


def best_line(duel, depth, leaf, vf, stats=None, ply_dt=PLY_DT):
    """Run the search and report the principal variation."""
    tt = {}
    st = stats if stats is not None else {}
    v = alpha_beta(duel, depth, leaf, "pudge", -1e9, 1e9, vf=vf, tt=tt,
                   stats=st, ply_dt=ply_dt)
    return v, st
