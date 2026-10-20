"""
Self-play training of the value function.

Loop:
    1. play games with the current policy
    2. record (features, outcome)
    3. refit the value function
    4. the new function plays the next round

Iteration 0 plays the hand-written rollout policy, so the first function
approximates *its* value; from then on the policy is greedy with respect to the
learned function, which is what removes the dependence on my hand-written
policy.  This is fitted value iteration with Monte-Carlo targets.

Self-play uses a 1-step greedy policy over the value function rather than the
full matrix-game search: |A| evaluations per decision instead of |A1| x |A2|,
sixty times cheaper, and the data it produces is just as usable.  The full
search is reserved for evaluating the finished function.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time

import numpy as np

from dota2sim.build import make_duel
from dota2sim.constants import UNITS_PER_METER
from .valuefn import ValueFunction, features
from .search import (actions_for, apply_action, advance, PUDGE_KIT,
                     INVOKER_KIT, default_policy, DECISION_DT)

MAX_TIME = 25.0


def random_duel(rng, hp_lo=0.25, hp_hi=1.0):
    """A random position: both heroes somewhere in the arena at partial HP.

    Randomisation is what makes the function general.  Starting every game at
    full HP and 4 m apart would only ever show the function a narrow slice of
    the state space, and the search would be badly wrong everywhere else.
    """
    d = make_duel(pudge_pos=(0.0, 0.0), invoker_pos=(0.0, 0.0))
    w_m, h_m = d.w / UNITS_PER_METER, d.h / UNITS_PER_METER
    for u in d.u.values():
        u.x = rng.uniform(1.0, w_m - 1.0) * UNITS_PER_METER
        u.y = rng.uniform(1.0, h_m - 1.0) * UNITS_PER_METER
    d.u["pudge"].hp = d.u["pudge"].max_hp * rng.uniform(hp_lo, hp_hi)
    d.u["invoker"].hp = d.u["invoker"].max_hp * rng.uniform(hp_lo, hp_hi)
    d.u["pudge"].rot_on = rng.random() < 0.5
    return d


def greedy_action(duel, who, vf, rng, eps, kit, force_ability=None,
                  force_p=0.0):
    """One-step greedy with respect to `vf`, with two kinds of exploration.

    `eps` is the usual uniform-random action.  That alone is not enough here:
    the greedy policy concentrates on Cold Snap, so positions that arise after
    Tornado or Ghost Walk are almost never visited and the learned function is
    simply wrong there - which then fed back into the search and produced the
    absurd ranking where Tornado scored worse than doing nothing.

    `force_ability` therefore casts a specific under-used ability with
    probability `force_p`, so every ability gets real coverage in the data
    rather than only the ones the policy already likes.
    """
    u = duel.u[who]
    acts = actions_for(duel, who, kit)
    if not acts:
        return None
    if force_ability and rng.random() < force_p:
        forced = [x for x in acts if x.ability == force_ability]
        if forced:
            return rng.choice(forced)
    if rng.random() < eps:
        return rng.choice(acts)
    best, best_v = None, -1e9
    for a in acts:
        d2 = duel.clone()
        apply_action(d2, who, a)
        advance(d2)
        v = vf(d2)
        if who == "invoker":
            v = 1.0 - v
        if v > best_v:
            best, best_v = a, v
    return best


#: abilities the greedy policy avoids, so they need to be injected on purpose
UNDERUSED = ["tornado", "ghost_walk", "sun_strike", "forge_spirit", "ice_wall",
             "emp", "alacrity", "deafening_blast"]


def play_game(vf, rng, eps=0.15, use_hand_seed=False, max_time=MAX_TIME,
              record_every=1, pool=None, force_p=0.0):
    """One self-play game.  Returns (features_list, outcome).

    `pool` is a list of past value functions.  When given, each side picks its
    policy from the pool uniformly at random rather than both playing the
    latest one.  Naive self-play trains against the current opponent only, and
    the pair then chases each other's weaknesses in a cycle - the win share
    swung between 0.07 and 0.81 across rounds with accuracy never improving.
    Playing against a mixture of past selves is what turns that chase into
    convergence towards a value that holds against the whole history.
    """
    d = random_duel(rng)
    vf_p = vf_i = vf
    if pool:
        vf_p = pool[rng.randrange(len(pool))]
        vf_i = pool[rng.randrange(len(pool))]
    traj = []
    step = 0
    next_decision = d.t
    while d.winner is None and d.t < max_time:
        if d.t >= next_decision - 1e-9:
            if step % record_every == 0:
                traj.append(features(d))
            if use_hand_seed:
                if d.u["pudge"].alive:
                    default_policy(d, "pudge")
                if d.u["invoker"].alive:
                    default_policy(d, "invoker")
            else:
                ap = greedy_action(d, "pudge", vf_p, rng, eps, PUDGE_KIT)
                forced = rng.choice(UNDERUSED) if force_p > 0 else None
                ai = greedy_action(d, "invoker", vf_i, rng, eps, INVOKER_KIT,
                                   force_ability=forced, force_p=force_p)
                if ap is not None:
                    apply_action(d, "pudge", ap)
                if ai is not None:
                    apply_action(d, "invoker", ai)
            next_decision += DECISION_DT
            step += 1
        d.step(1.0 / 30.0)
    outcome = 0.5 if d.winner is None else (1.0 if d.winner == "pudge" else 0.0)
    return traj, outcome


def collect(vf, n_games, rng, eps=0.15, use_hand_seed=False, verbose=False,
            pool=None):
    X, y = [], []
    t0 = time.time()
    for g in range(n_games):
        traj, outcome = play_game(vf, rng, eps=eps, use_hand_seed=use_hand_seed,
                                  pool=pool, force_p=force_p)
        for phi in traj:
            X.append(phi)
            y.append(outcome)
        if verbose and (g + 1) % max(1, n_games // 10) == 0:
            print(f"    {g+1}/{n_games} игр, {len(X)} позиций, "
                  f"{time.time()-t0:.1f}с", flush=True)
    return np.asarray(X), np.asarray(y)


def collect_games(vf, n_games, rng, eps=0.15, use_hand_seed=False,
                  verbose=False, pool=None, force_p=0.0):
    """Play games and return a list of per-game (features, outcome) groups.

    Grouping by game matters: positions inside one trajectory are strongly
    correlated, so a random split by position leaks and reports an accuracy
    that does not survive fresh data.  Splitting by game is the honest split.
    """
    games = []
    t0 = time.time()
    for g in range(n_games):
        traj, outcome = play_game(vf, rng, eps=eps, use_hand_seed=use_hand_seed,
                                  pool=pool)
        if traj:
            games.append((np.asarray(traj), outcome))
        if verbose and (g + 1) % max(1, n_games // 5) == 0:
            print(f"    {g+1}/{n_games} игр, "
                  f"{sum(len(t) for t, _ in games)} позиций, "
                  f"{time.time()-t0:.1f}с", flush=True)
    return games


def split_games(games, rng, val_frac=0.25):
    idx = list(range(len(games)))
    rng.shuffle(idx)
    k = max(1, int(len(idx) * val_frac))
    val = [games[i] for i in idx[:k]]
    tr = [games[i] for i in idx[k:]]
    return tr, val


def flatten(groups):
    if not groups:
        return np.zeros((0, 0)), np.zeros(0)
    X = np.vstack([g for g, _ in groups])
    y = np.concatenate([np.full(len(g), o) for g, o in groups])
    return X, y


def train(rounds=12, games_per_round=200, seed=0, out="results/valuefn.npz",
          verbose=True, l2=0.05, val_frac=0.25):
    rng = random.Random(seed)
    vf = ValueFunction()
    pool = []                       # past policies to play against
    games = []                      # accumulated across rounds, not per round
    for r in range(rounds):
        t0 = time.time()
        seed_round = (r == 0)
        new = collect_games(vf, games_per_round, rng,
                            eps=0.05 if seed_round else 0.15,
                            use_hand_seed=seed_round, verbose=verbose,
                            pool=(pool if len(pool) >= 2 else None),
                            force_p=0.0 if seed_round else 0.25)
        games.extend(new)
        if len(games) > 4000:               # bounded memory, keep the recent past
            games = games[-4000:]
        tr, va = split_games(games, rng, val_frac)
        Xtr, ytr = flatten(tr)
        Xva, yva = flatten(va)
        vf.fit(Xtr, ytr, Xval=Xva, yval=yva, l2=l2)
        pool.append(copy.deepcopy(vf))
        if len(pool) > 6:
            pool.pop(0)
        acc_tr = vf.accuracy(Xtr, ytr)
        acc_va = vf.accuracy(Xva, yva)
        if verbose:
            print(f"  раунд {r:2d}: игр {len(games):5d}  "
                  f"обуч {acc_tr:.3f}  ОТЛОЖ {acc_va:.3f}  "
                  f"доля побед Pudge {float((ytr>0.5).mean()):.2f}  "
                  f"{time.time()-t0:.1f}с", flush=True)
    vf.save(out)
    if verbose:
        Xva, yva = flatten(split_games(games, rng, val_frac)[1])
        print(f"\nсохранено в {out}")
        print("калибровка на отложенной выборке "
              "(предсказано -> фактически):")
        for lo, hi, actual, n in vf.calibration(Xva, yva):
            print(f"  {lo:.1f}-{hi:.1f} -> {actual:.2f}   ({n} позиций)")
    return vf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/valuefn.npz")
    a = ap.parse_args()
    train(a.rounds, a.games, a.seed, a.out)
