"""
Mechanic-by-mechanic validation of the forward model against closed-form
values.

This exists because the model produced degenerate answers four times in a row
and every time the cause was an unverified assumption.  Anything the search
later reports is only as good as these checks, so each one compares the
simulator against a number computed independently.

Run:  python3 -m dota2sim.test_forward
"""

from __future__ import annotations

from .build import make_duel
from .constants import SERVER_TICK, armor_multiplier

TOL = 0.06          # mechanics are checked to 6%, discretisation is the floor


def _fresh(dist_m=2.0):
    return make_duel(pudge_pos=(10.0, 40.0), invoker_pos=(10.0 + dist_m, 40.0))


def check(name, got, exp, tol=TOL):
    ok = abs(got - exp) <= tol * max(abs(exp), 1e-9)
    print(f"  {'OK ' if ok else 'FAIL'} {name:44s} модель {got:9.1f}  "
          f"аналитика {exp:9.1f}  откл {100*(got-exp)/max(abs(exp),1e-9):+6.1f}%")
    return ok


def test_auto_attacks():
    print("Автоатаки Pudge по Invoker")
    ok = True
    for n, tol in ((200, TOL), (400, TOL)):
        d = _fresh(2.0)
        p, i = d.u["pudge"], d.u["invoker"]
        p.attack_target = "invoker"
        hp0 = i.hp
        for _ in range(n):
            d.step()
        T = n * SERVER_TICK
        exp = (p.attack_damage() * armor_multiplier(i.armor())
               * p.attacks_per_second() * T - i.hp_regen() * T)
        ok &= check(f"за {T:.1f} с", hp0 - i.hp, exp, tol)
    return ok


def test_rot():
    print("Rot: скорость, размер импульса, самоурон")
    d = _fresh(2.0)
    p, i = d.u["pudge"], d.u["invoker"]
    p.rot_on = True
    hp0i, hp0p = i.hp, p.hp
    ticks = [0]
    orig = d.deal

    def traced(src, tgt, amt, dtype, **kw):
        if kw.get("label") == "rot":
            ticks[0] += 1
        return orig(src, tgt, amt, dtype, **kw)

    d.deal = traced
    n = 200
    for _ in range(n):
        d.step()
    T = n * SERVER_TICK
    a = p.abilities["rot"]
    ok = check("урон по Invoker", hp0i - i.hp,
               a["damage_per_second"] * (1 - i.magic_resist()) * T
               - i.hp_regen() * T)
    ok &= check("число импульсов", ticks[0], T / a["tick"], 0.10)
    ok &= check("самоурон (режется MR)", hp0p - p.hp,
                a["self_damage"] * (1 - p.magic_resist()) * T - p.hp_regen() * T)
    return ok


def test_cold_snap_procs():
    print("Cold Snap: частота проков (самоурон Rot их запускает)")
    d = _fresh(2.0)
    p, i = d.u["pudge"], d.u["invoker"]
    p.rot_on = True
    d.try_cast(i, "cold_snap", ("unit", "pudge"))
    procs = [0]
    orig = d.deal

    def traced(src, tgt, amt, dtype, **kw):
        if kw.get("label") == "proc":
            procs[0] += 1
        return orig(src, tgt, amt, dtype, **kw)

    d.deal = traced
    for _ in range(int(6.0 / SERVER_TICK)):
        d.step()
    a = i.abilities["cold_snap"]
    return check("проков за 6 с", procs[0], a["duration"] / a["proc_interval"], 0.25)


def test_meat_hook():
    print("Meat Hook: урон и время до попадания")
    d = _fresh(10.0)
    p, i = d.u["pudge"], d.u["invoker"]
    hp0, t0 = i.hp, d.t
    d.try_cast(p, "meat_hook", ("point", i.x, i.y))
    for _ in range(600):
        d.step()
        if i.hp < hp0:
            break
    ab = p.abilities["meat_hook"]
    flight = (10.0 * 52.4934 - (ab["radius"] + i.stats["collision"])) / ab["speed"]
    ok = check("урон", hp0 - i.hp, ab["damage"])
    ok &= check("время (каст 0.3 + полёт)", d.t - t0, 0.3 + flight, 0.15)
    return ok


def test_arena_exit():
    print("Выход за арену = поражение")
    d = make_duel(pudge_pos=(1.0, 40.0), invoker_pos=(8.0, 40.0))
    d.u["pudge"].move_target = (-5000.0, 40.0 * 52.4934)
    for _ in range(400):
        d.step()
        if d.winner:
            break
    ok = d.winner == "invoker" and "left the arena" in d.reason
    print(f"  {'OK ' if ok else 'FAIL'} победитель {d.winner}: {d.reason}")
    return ok


def test_cold_snap_breaks_channel():
    print("Cold Snap прерывает канал Dismember")
    d = _fresh(2.0)
    p, i = d.u["pudge"], d.u["invoker"]
    # put Pudge into a Dismember channel, then stun him
    p.channel_ability = "dismember"
    p.channel_left = p.abilities["dismember"]["duration"]
    p.abilities["dismember"]["_accum"] = 0.0
    # Cold Snap only stuns through its procs, and a proc needs a damage
    # instance of 10 or more.  Pudge's own Rot supplies them - which is
    # exactly why the interaction works in a real fight and why a bare
    # Cold Snap on an undamaged target does nothing.
    p.rot_on = True
    d.try_cast(i, "cold_snap", ("unit", "pudge"))
    broke = False
    for _ in range(int(2.0 / SERVER_TICK)):
        d.step()
        if p.channel_ability is None:
            broke = True
            break
    print(f"  {'OK ' if broke else 'FAIL'} канал прерван")
    return broke


def main():
    print("Проверка вперёд-модели против аналитики\n")
    results = [
        test_auto_attacks(),
        test_rot(),
        test_cold_snap_procs(),
        test_meat_hook(),
        test_arena_exit(),
        test_cold_snap_breaks_channel(),
    ]
    print(f"\nпройдено {sum(results)} из {len(results)}")
    return all(results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
