"""Correctness tests for the DP core, on toy pursuit games with known values."""

from .dp import solve_game


class Trans:
    __slots__ = ("terminal", "n_u", "n_v", "succ")

    def __init__(self, terminal, n_u, n_v, succ):
        self.terminal = terminal
        self.n_u = n_u
        self.n_v = n_v
        self.succ = succ


def make_pursuit(N, speed1, speed2, draw_value=0.5):
    """1-D pursuit on d in 0..N.

    d=0 -> player 1 wins (1.0).  d=N -> player 2 wins (0.0).
    Player 1 shifts d by -u (u in 0..speed1), player 2 shifts d by +v.
    Both move simultaneously; the net is clamped into [0, N].
    """
    states = list(range(N + 1))
    moves1 = list(range(speed1 + 1))     # how far player 1 closes
    moves2 = list(range(speed2 + 1))     # how far player 2 opens

    def transitions(d):
        if d == 0:
            return Trans(1.0, 0, 0, [])
        if d == N:
            return Trans(0.0, 0, 0, [])
        succ = []
        for u in moves1:
            row = []
            for v in moves2:
                nd = d - u + v
                nd = 0 if nd < 0 else (N if nd > N else nd)
                row.append(nd)
            succ.append(row)
        return Trans(None, len(moves1), len(moves2), succ)

    return states, transitions


def test_faster_player_always_wins():
    states, tr = make_pursuit(N=6, speed1=2, speed2=1)
    res = solve_game(states, tr, tol=1e-9, max_sweeps=200)
    for d in range(6):
        assert abs(res.V[d] - 1.0) < 1e-6, (d, res.V[d])
    assert abs(res.V[6] - 0.0) < 1e-6
    print("  faster player always wins: OK")


def test_equal_speed_is_a_draw():
    """With equal speeds in 1-D the chaser cannot close: value stays at the
    draw value in the interior."""
    states, tr = make_pursuit(N=6, speed1=1, speed2=1, draw_value=0.5)
    res = solve_game(states, tr, tol=1e-9, max_sweeps=300)
    for d in range(1, 6):
        assert abs(res.V[d] - 0.5) < 1e-6, (d, res.V[d])
    print("  equal speed is a draw: OK")


def test_hopeless_chaser():
    """Chaser slower than the runner: the runner wins from every distance."""
    states, tr = make_pursuit(N=6, speed1=1, speed2=2)
    res = solve_game(states, tr, tol=1e-9, max_sweeps=200)
    for d in range(1, 7):
        assert abs(res.V[d] - 0.0) < 1e-6, (d, res.V[d])
    print("  hopeless chaser loses: OK")


def test_discount_biases_win_value():
    """A discount < 1 must *not* be used for a reach-avoid game: it turns a
    certain win into gamma^k, so a win in one step scores 0.999 instead of 1."""
    states, tr = make_pursuit(N=6, speed1=2, speed2=1)
    a = solve_game(states, tr, tol=1e-9, max_sweeps=400)
    b = solve_game(states, tr, gamma=0.999, tol=1e-9, max_sweeps=400)
    assert abs(a.V[1] - 1.0) < 1e-9, a.V[1]
    assert abs(b.V[1] - 0.999) < 1e-6, b.V[1]
    print("  discount biases the win value (undiscounted is correct): OK")


def test_policy_is_optimal_best_response():
    """Verify the extracted policy by brute force: no unilateral deviation of
    either player may improve their payoff against the opponent's policy."""
    states, tr = make_pursuit(N=6, speed1=2, speed2=1)
    res = solve_game(states, tr, tol=1e-9, max_sweeps=200)
    for s in states:
        t = tr(s)
        if t.terminal is not None:
            continue
        p, q = res.policy1[s], res.policy2[s]
        v_star = res.V[s]
        # player 1 deviation
        for i in range(t.n_u):
            v = sum(q[j] * res.V[t.succ[i][j]] for j in range(t.n_v))
            assert v <= v_star + 1e-6, ("P1 can improve", s, i, v, v_star)
        # player 2 deviation
        for j in range(t.n_v):
            v = sum(p[i] * res.V[t.succ[i][j]] for i in range(t.n_u))
            assert v >= v_star - 1e-6, ("P2 can improve", s, j, v, v_star)
    print("  extracted policy is a verified equilibrium: OK")


if __name__ == "__main__":
    print("DP core tests")
    test_faster_player_always_wins()
    test_equal_speed_is_a_draw()
    test_hopeless_chaser()
    test_discount_biases_win_value()
    test_policy_is_optimal_best_response()
    print("all DP tests passed")
