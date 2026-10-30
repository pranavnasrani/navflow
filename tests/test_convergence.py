"""Tests for solver.foam.convergence: force-file parsing and the 'small increment' stopping rule."""
import numpy as np

from solver.foam.convergence import is_converged, is_diverged, load_forces, parse_force_dat

WINDOW, TOL, MIN_ITERS = 50, 0.002, 200


def _forces(n, fx, fz=0.0, noise=0.0, seed=0):
    """n rows of [iteration, Fx, Fy, Fz]; Fx/Fz may be arrays or scalars."""
    rng = np.random.default_rng(seed)
    it = np.arange(1, n + 1, dtype=float)
    out = np.zeros((n, 4))
    out[:, 0] = it
    out[:, 1] = fx + rng.uniform(-noise, noise, n)
    out[:, 3] = fz + rng.uniform(-noise, noise, n)
    return out


def test_converged_when_steady():
    assert is_converged(_forces(300, 100.0, 5.0, noise=0.05), WINDOW, TOL, MIN_ITERS)


def test_not_converged_when_drifting():
    f = _forces(300, 100.0, 5.0)
    f[-50:, 1] = np.linspace(100.0, 99.0, 50)  # drag falls 1% across the window
    assert not is_converged(f, WINDOW, TOL, MIN_ITERS)


def test_converged_with_near_zero_lift():
    # Lift swings +-0.1 about zero: huge relative to its own mean, tiny against drag.
    f = _forces(300, 100.0)
    f[:, 3] = np.where(np.arange(300) % 2 == 0, 0.1, -0.1)
    assert is_converged(f, WINDOW, TOL, MIN_ITERS)


def test_not_before_min_iters():
    assert not is_converged(_forces(150, 100.0), WINDOW, TOL, MIN_ITERS)


def test_not_converged_with_too_few_rows():
    assert not is_converged(_forces(30, 100.0), WINDOW, TOL, 0)


def test_nan_is_diverged():
    f = _forces(300, 100.0)
    assert not is_diverged(f)
    f[100, 1] = np.nan
    assert is_diverged(f)
    assert not is_converged(f, WINDOW, TOL, MIN_ITERS)
    f[100, 1] = np.inf
    assert is_diverged(f)


def test_parse_both_formats():
    paren = "# Time forces\n10 (1 2 3) (0 0 0) (1 2 3)\n"
    plain = "# Time total_x ...\n10 1 2 3 0 0 0 1 2 3\n"
    for text in (paren, plain):
        a = parse_force_dat(text)
        assert a.shape == (1, 4)
        assert a.tolist() == [[10, 1, 2, 3]]


def test_parse_skips_comments_and_blank_lines():
    text = "# a\n# b\n\n1 1 0 0 0 0 0 0 0 0\n2 2 0 0 0 0 0 0 0 0\n"
    assert parse_force_dat(text)[:, 0].tolist() == [1, 2]


def test_parse_empty_gives_empty_array():
    assert parse_force_dat("# header only\n").shape == (0, 4)


def test_restart_segments_dedupe():
    first = "1 1 0 0 0 0 0 0 0 0\n2 2 0 0 0 0 0 0 0 0\n3 3 0 0 0 0 0 0 0 0\n"
    second = "3 30 0 0 0 0 0 0 0 0\n4 40 0 0 0 0 0 0 0 0\n"
    f = load_forces([first, second])
    assert f[:, 0].tolist() == [1, 2, 3, 4]
    assert f[:, 1].tolist() == [1, 2, 30, 40]  # last row wins for the repeated iteration 3


def test_load_forces_empty():
    assert load_forces([]).shape == (0, 4)


def test_parse_skips_truncated_last_line():
    good = "1 1 2 3 0 0 0 0 0 0\n"
    a = parse_force_dat(good + "2 (1 2")
    assert a.tolist() == [[1, 1, 2, 3]]
    assert parse_force_dat("2 (1 2").shape == (0, 4)


def test_parse_skips_cut_off_number_token():
    good = "1 1 2 3 0 0 0 0 0 0\n"
    a = parse_force_dat(good + "2 (1 2 3.5e")
    assert a.tolist() == [[1, 1, 2, 3]]
    assert parse_force_dat("2 1 2 1.2e").shape == (0, 4)


def _seg(first, last, fx):
    return "".join(f"{i} {fx} 0 0 0 0 0 0 0 0\n" for i in range(first, last + 1))


def test_restart_truncates_stale_tail_of_old_segment():
    old = _seg(1, 187, 1)  # ran on to 187 before the interruption
    new = _seg(151, 160, 2)  # resumed from the save at 150
    f = load_forces([old, new])
    assert f[-1, 0] == 160
    assert f[:, 0].tolist() == list(range(1, 161))
    assert np.all(f[f[:, 0] <= 150, 1] == 1)
    assert np.all(f[f[:, 0] > 150, 1] == 2)
