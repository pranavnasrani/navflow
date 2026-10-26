"""Resumable run queue for the OpenFOAM batch: what to run, in what order, and how far each got.

Maths link (IGCSE 0606): coarse_first_order picks each object's wind directions by repeated
midpoints (a+b)/2 of the gaps between chosen directions - interval bisection. After the
first three runs (both ends and the middle) every extra run halves a gap, so a partial
sweep always covers the whole direction range as evenly as possible.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from solver.foam.settings import ObjectCfd

STATUSES = ("waiting", "running", "stopped", "done", "flagged", "failed")


@dataclass
class RunRecord:
    run_id: str
    obj_id: str
    direction_index: int
    direction_deg: float
    status: str  # one of STATUSES
    iterations: int = 0
    converged: bool = False
    log: str = ""
    machine: str = ""


def coarse_first_order(n: int) -> list[int]:
    """Order direction indices 0..n-1 so early runs span the range: ends+middle, then bisect.

    Start with [0, n//2, n-1] (de-duplicated, in that order). Then repeat passes: for every
    adjacent pair (a, b) of chosen indices, left to right, with b - a > 1, append
    floor((a+b)/2). Each pass works from the set chosen at the start of that pass.
    n=9 -> [0, 4, 8, 2, 6, 1, 3, 5, 7].
    """
    if n <= 0:
        return []
    order: list[int] = []
    for i in (0, n // 2, n - 1):
        if i not in order:
            order.append(i)
    while len(order) < n:
        chosen = sorted(order)
        for a, b in zip(chosen, chosen[1:]):
            if b - a > 1:
                order.append((a + b) // 2)
    return order


def build_queue(
    objects: dict[str, ObjectCfd], obj_ids: list[str], directions: list[int] | None = None
) -> list[RunRecord]:
    """Index-major queue: every object's first index, then every object's next index, ...

    `directions` (e.g. [0, 4, 8] for the preview) restricts which indices are queued; the
    coarse-first order is kept. Objects interleave in the order of `obj_ids`.
    """
    n = max((len(objects[o].directions_deg) for o in obj_ids), default=0)
    wanted = None if directions is None else set(directions)
    records: list[RunRecord] = []
    for idx in coarse_first_order(n):
        if wanted is not None and idx not in wanted:
            continue
        for obj_id in obj_ids:
            degs = objects[obj_id].directions_deg
            if idx < len(degs):
                records.append(
                    RunRecord(
                        run_id=f"{obj_id}_d{idx}",
                        obj_id=obj_id,
                        direction_index=idx,
                        direction_deg=float(degs[idx]),
                        status="waiting",
                    )
                )
    return records


def load_queue(path: Path | str) -> list[RunRecord]:
    """Read the queue JSON; a missing file means an empty queue."""
    p = Path(path)
    if not p.exists():
        return []
    return [RunRecord(**d) for d in json.loads(p.read_text())]


def save_queue(path: Path | str, records: list[RunRecord]) -> None:
    """Write to <path>.tmp then os.replace, so a crash never leaves a half-written queue."""
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in records], indent=2))
    os.replace(tmp, p)


def merge_queue(existing: list[RunRecord], new: list[RunRecord]) -> list[RunRecord]:
    """Follow `new`'s order but keep the existing record (status, progress) for known run_ids.

    Records only in `existing` are appended, so a finished run is never lost.
    """
    known = {r.run_id: r for r in existing}
    merged = [known.get(r.run_id, r) for r in new]
    seen = {r.run_id for r in new}
    merged.extend(r for r in existing if r.run_id not in seen)
    return merged


def next_run(records: list[RunRecord]) -> RunRecord | None:
    """Resume an interrupted run (running/stopped) first, otherwise the first waiting one."""
    for r in records:
        if r.status in ("running", "stopped"):
            return r
    for r in records:
        if r.status == "waiting":
            return r
    return None
