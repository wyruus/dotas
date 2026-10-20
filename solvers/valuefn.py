"""
A learned value function for the leaf of the search.

The search is only as good as its leaf.  With a hand-written rollout policy the
answer moved every time I edited that policy, which is not an answer at all.
So the leaf is trained instead: play games, record positions and outcomes, fit a
function, and let the improved function play better games.

Representation is linear in a hand-chosen feature vector, trained by logistic
regression.  Linear is deliberate: the dataset is small, the signal is strong
(most of the outcome is decided by HP and cooldown state), and a linear model
cannot silently memorise noise the way a large network would on a few thousand
games.  A numpy-only MLP is a drop-in upgrade once there is enough data to
justify it.

Bootstrapping: iteration 0 collects games played by the hand-written rollout
policy, so the first function approximates *that* policy's value.  From then on
the search uses the learned function, plays stronger games, and refits - the
usual self-play improvement loop, with the hand policy only as a seed.
"""

from __future__ import annotations

import math

import numpy as np

from dota2sim.constants import UNITS_PER_METER

#: abilities whose readiness materially changes the position
TRACKED = ["meat_hook", "dismember", "cold_snap", "sun_strike",
           "chaos_meteor", "emp", "tornado", "deafening_blast", "ice_wall"]


def _ready(u, name):
    return 1.0 if u.cooldowns.get(name, 0.0) <= 0.0 else 0.0


def _frac_ready(u, name):
    """How far through the cooldown the ability is: 0 just used, 1 ready."""
    ab = u.abilities.get(name)
    if ab is None or ab.get("cooldown", 0.0) <= 0:
        return 1.0
    left = u.cooldowns.get(name, 0.0)
    return float(np.clip(1.0 - left / ab["cooldown"], 0.0, 1.0))


def features(duel) -> np.ndarray:
    """Feature vector for the value function.  Order is fixed and must match
    the trained weights."""
    p = duel.u["pudge"]
    i = duel.u["invoker"]
    d = p.dist_to(i) / UNITS_PER_METER
    f = []
    # --- health, the single strongest signal ---------------------------
    f.append(p.hp / p.max_hp)
    f.append(i.hp / i.max_hp)
    f.append((p.hp / p.max_hp) - (i.hp / i.max_hp))
    # --- mana -----------------------------------------------------------
    f.append(p.mana / max(p.max_mana, 1))
    f.append(i.mana / max(i.max_mana, 1))
    # --- geometry -------------------------------------------------------
    f.append(min(d, 30.0) / 30.0)
    f.append(1.0 / (1.0 + d))                     # sharp near-range term
    f.append(1.0 if d * UNITS_PER_METER <= 250 else 0.0)    # Pudge's Rot
    f.append(1.0 if d * UNITS_PER_METER <= 223 else 0.0)    # Pudge's attack
    f.append(1.0 if d * UNITS_PER_METER <= 648 else 0.0)    # Invoker's attack
    # wall proximity, normalised by the arena half-diagonal
    half = 0.5 * math.hypot(duel.w, duel.h)
    f.append(min(p.x, duel.w - p.x, p.y, duel.h - p.y) / half)
    f.append(min(i.x, duel.w - i.x, i.y, duel.h - i.y) / half)
    # --- cooldown readiness: how close each key ability is --------------
    for name in TRACKED:
        owner = p if name in p.abilities else i
        f.append(_frac_ready(owner, name))
    # --- live effects ---------------------------------------------------
    snap = next((b for b in p.buffs if b.active() and b.name == "cold_snap"), None)
    f.append(snap.duration / 5.8 if snap else 0.0)
    f.append(1.0 if p.rot_on else 0.0)
    burn = next((b for b in i.buffs if b.active() and b.name == "chaos_meteor"), None)
    f.append(1.0 if burn else 0.0)
    f.append(1.0 if not p.alive else 0.0)
    f.append(1.0 if not i.alive else 0.0)
    # --- Pudge's own kill threat ---------------------------------------
    f.append(1.0 if _ready(p, "meat_hook") and d * UNITS_PER_METER <= 1400 else 0.0)
    f.append(1.0 if _ready(i, "cold_snap") and d * UNITS_PER_METER <= 1000 else 0.0)

    # --- interaction terms ---------------------------------------------
    # The outcome hinges on conjunctions - "Cold Snap is up AND the target is
    # in range AND the hook is down" - and a purely linear model cannot form
    # them.  With only main effects the fit plateaued at ~0.70 accuracy on
    # deterministic games, which is the signature of a missing basis rather
    # than of noise.  These products supply the conjunctions explicitly.
    F = np.asarray(f, dtype=np.float64)
    hp_p, hp_i = F[0], F[1]
    near_rot = F[7]
    near_atk_p = F[8]
    near_atk_i = F[9]
    cs_ready = _ready(i, "cold_snap")
    hook_ready = _ready(p, "meat_hook")
    dis_ready = _ready(p, "dismember")
    sun_ready = _ready(i, "sun_strike")
    met_ready = _ready(i, "chaos_meteor")
    f += [
        hp_p * hp_i,
        hp_p - hp_i * 2.0,
        hp_p / max(hp_i, 1e-3),
        near_rot * hp_i,
        near_atk_p * hp_p,
        near_atk_i * hp_i,
        cs_ready * near_rot,                 # Cold Snap will land and punish
        cs_ready * hp_p,
        hook_ready * hp_i,
        dis_ready * hook_ready * hp_i,
        sun_ready * cs_ready,                # the follow-up is only real together
        met_ready * cs_ready,
        (1.0 - hp_p) * cs_ready,             # low Pudge + Cold Snap = lethal
        (1.0 - hp_i) * hook_ready,           # low Invoker + hook = lethal
        near_rot * (1.0 - hp_i),
        F[19] * F[20],                       # both ultimates ready
    ]
    return np.asarray(f, dtype=np.float64)


N_FEATURES = len(features.__wrapped__() if hasattr(features, "__wrapped__")
                 else []) or None


class ValueFunction:
    """Logistic-linear estimate of P(pudge wins | state)."""

    def __init__(self, n_features=None, seed=0):
        self.n = n_features
        self.w = None
        self.mean = None
        self.scale = None
        self.val_loss = None
        self.platt = None

    def __call__(self, duel) -> float:
        phi = features(duel)
        if self.w is None:
            return 0.5
        z = (phi - self.mean) / self.scale
        return float(1.0 / (1.0 + math.exp(-float(np.dot(self.w, z)))))

    def fit(self, X, y, Xval=None, yval=None, l2=0.05, iters=3000,
            lr=0.15, patience=200):
        """Logistic regression with a held-out set and early stopping.

        The first version reported accuracy on the data it had just been fit
        on and showed 0.75; on fresh data the same model scored 0.52, i.e.
        chance, with a non-monotone calibration curve.  In-sample accuracy on
        correlated trajectory data is meaningless, so the fit now stops on
        validation loss and the caller is handed a number it can trust.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n = X.shape[1]
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        Z = (X - self.mean) / self.scale
        Zv = ((np.asarray(Xval) - self.mean) / self.scale
              if Xval is not None else None)

        def loss(ZZ, yy, w):
            p = 1.0 / (1.0 + np.exp(-np.clip(ZZ @ w, -30, 30)))
            eps = 1e-9
            return float(-(yy * np.log(p + eps)
                           + (1 - yy) * np.log(1 - p + eps)).mean()
                         + 0.5 * l2 * float(w @ w))

        self.w = np.zeros(self.n)
        best_w = self.w.copy()
        best = loss(Zv if Zv is not None else Z, yval if Zv is not None else y,
                    self.w)
        bad = 0
        m = len(y)
        for it in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w, -30, 30)))
            g = Z.T @ (p - y) / m + l2 * self.w
            self.w -= lr * g
            cur = loss(Zv if Zv is not None else Z,
                       yval if Zv is not None else y, self.w)
            if cur < best - 1e-6:
                best, best_w, bad = cur, self.w.copy(), 0
            else:
                bad += 1
                if bad >= patience:
                    break
        self.w = best_w
        self.val_loss = best
        # Platt scaling: regularised logistic regression is systematically
        # under-confident, so predicted 0.5 landed on an observed rate of 0.28.
        # A one-dimensional refit on the held-out scores restores the scale
        # without touching the ranking.
        self.platt = None
        if Zv is not None and yval is not None and len(yval) > 50:
            z = Zv @ self.w
            a_, b_ = 1.0, 0.0
            for _ in range(2000):
                p = 1.0 / (1.0 + np.exp(-np.clip(a_ * z + b_, -30, 30)))
                ga = float(((p - yval) * z).mean())
                gb = float((p - yval).mean())
                a_ -= 0.5 * ga
                b_ -= 0.5 * gb
            self.platt = (a_, b_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Z = (X - self.mean) / self.scale
        z = Z @ self.w
        if getattr(self, "platt", None):
            a_, b_ = self.platt
            z = a_ * z + b_
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def accuracy(self, X, y):
        return float(((self.predict(X) > 0.5) == (np.asarray(y) > 0.5)).mean())

    def calibration(self, X, y, bins=5):
        """Fraction of Pudge wins actually observed in each predicted band.

        A trustworthy value function must be monotone here.  The first
        version was not: it predicted 0.8+ on positions Pudge went on to win
        only a quarter of the time.
        """
        p = self.predict(X)
        y = np.asarray(y)
        edges = np.linspace(0.0, 1.0, bins + 1)
        rows = []
        for i in range(bins):
            m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1
                                   else p <= edges[i + 1])
            if m.sum() > 5:
                rows.append((edges[i], edges[i + 1], float(y[m].mean()),
                             int(m.sum())))
        return rows

    def save(self, path):
        np.savez(path, w=self.w, mean=self.mean, scale=self.scale,
                 platt=np.asarray(self.platt if self.platt else (1.0, 0.0)))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        v = cls(len(z["w"]))
        v.w, v.mean, v.scale = z["w"], z["mean"], z["scale"]
        v.platt = tuple(z["platt"]) if "platt" in z else None
        return v
