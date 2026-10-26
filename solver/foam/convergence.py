"""Force-history parsing and the convergence rule for OpenFOAM wind-tunnel runs.

OpenFOAM's `forces` function object writes one row per iteration to
postProcessing/forces/<startTime>/force.dat. A restarted run starts a new
<restartTime>/force.dat, so a full history is several file texts joined in time order.

Maths link (IGCSE 0606): the stopping test is a "small increment" test. Over the last
`window` iterations the spread of drag (max - min, "delta F") must be smaller than a tiny
fraction of its mean value, i.e. delta F / F < 0.002 (0.2%). It is the same idea as using
small increments delta x to approximate a rate of change: when the increments stop mattering,
the quantity has settled.
"""
from __future__ import annotations

import numpy as np

_N_COLS = 4  # iteration, Fx, Fy, Fz (total force)


def parse_force_dat(text: str) -> np.ndarray:
    """Parse one force.dat into an (N, 4) array: iteration, Fx, Fy, Fz (totals).

    '#' lines and blank lines are skipped and '(' ')' are stripped, so both the
    '10 (1 2 3) (0 0 0) (1 2 3)' and the plain '10 1 2 3 0 0 0 1 2 3' layouts parse.
    Only the first four numbers matter: Time, then the total force vector.
    A truncated last line (live read during a write, or power loss) is skipped, not an error.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace("(", " ").replace(")", " ").split()
        try:
            row = [float(p) for p in parts[:_N_COLS]]
        except ValueError:  # cut-off token such as "1.2e": a write in progress or power loss
            continue
        if len(row) == _N_COLS:  # a truncated final line has too few numbers: skip it
            rows.append(row)
    if not rows:
        return np.empty((0, _N_COLS))
    return np.array(rows, dtype=float)


def load_forces(texts: list[str]) -> np.ndarray:
    """Join restart segments (texts in time order) into one history, sorted by iteration.

    A restart resumes from the last saved iteration (say 150) even though the old segment
    ran on to 187. The later segment is the truth from its first iteration onward, so it
    truncates every earlier row with iteration >= its own first iteration. Without this,
    stale rows 161-187 would survive until the new run overtook them and would pollute the
    convergence window. Within the result the last row per iteration wins.
    """
    kept: list[np.ndarray] = []
    for text in texts:
        seg = parse_force_dat(text)
        if not len(seg):
            continue
        first = seg[0, 0]
        kept = [k[k[:, 0] < first] for k in kept]
        kept.append(seg)
    if not kept:
        return np.empty((0, _N_COLS))
    latest: dict[float, np.ndarray] = {}
    for row in np.vstack(kept):  # duplicates inside one segment: last row wins
        latest[float(row[0])] = row
    return np.array([latest[k] for k in sorted(latest)], dtype=float)


def is_diverged(forces: np.ndarray) -> bool:
    """True if any force value is NaN or infinite (the solver has blown up)."""
    return bool(len(forces)) and not bool(np.all(np.isfinite(forces)))


def is_converged(forces: np.ndarray, window: int, tol: float, min_iters: int) -> bool:
    """Drag and lift each moved by less than `tol` over the last `window` iterations.

    delta F = max - min over the window (np.ptp). Drag is compared with its own mean.
    Lift is compared with max(|mean lift|, |mean drag|): a car or building can have lift
    close to zero, and 'a tiny fraction of nearly nothing' would never be met, so lift is
    judged against the (much larger) drag instead.
    """
    if len(forces) < window or forces[-1, 0] < min_iters or is_diverged(forces):
        return False
    last = forces[-window:]
    drag_scale = max(abs(last[:, 1].mean()), 1e-12)
    lift_scale = max(abs(last[:, 3].mean()), drag_scale)
    return bool(
        (np.ptp(last[:, 1]) <= tol * drag_scale) and (np.ptp(last[:, 3]) <= tol * lift_scale)
    )
