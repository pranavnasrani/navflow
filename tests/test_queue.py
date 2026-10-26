"""Tests for solver.foam.queue: coarse-first ordering and the resumable run queue."""
import json
from pathlib import Path

import pytest

from solver.foam.queue import (
    RunRecord,
    build_queue,
    coarse_first_order,
    load_queue,
    merge_queue,
    next_run,
    reset_failed,
    save_queue,
)
from solver.foam.settings import load_settings

CONFIG = Path(__file__).resolve().parent.parent / "config" / "exhibition_config.json"


def _rec(run_id, status="waiting", **kw):
    return RunRecord(run_id=run_id, obj_id=run_id.split("_")[0], direction_index=0,
                     direction_deg=0.0, status=status, **kw)


def test_coarse_first_order():
    assert coarse_first_order(9) == [0, 4, 8, 2, 6, 1, 3, 5, 7]
    assert coarse_first_order(5) == [0, 2, 4, 1, 3]
    assert coarse_first_order(1) == [0]
    assert coarse_first_order(2) == [0, 1]


@pytest.mark.parametrize("n", range(1, 13))
def test_coarse_first_order_is_a_permutation(n):
    order = coarse_first_order(n)
    assert sorted(order) == list(range(n))
    assert len(order) == n


def test_build_queue_interleaves_objects():
    _, objs, _ = load_settings(CONFIG)
    ids = ["airplane", "f1_car"]
    q = build_queue(objs, ids)
    assert len(q) == 18
    assert [r.run_id for r in q[:6]] == [
        "airplane_d0", "f1_car_d0", "airplane_d4", "f1_car_d4", "airplane_d8", "f1_car_d8"]
    assert all(r.status == "waiting" and r.iterations == 0 for r in q)
    r = q[2]
    assert (r.obj_id, r.direction_index, r.direction_deg) == (
        "airplane", 4, objs["airplane"].directions_deg[4])


def test_build_queue_directions_filter_preview():
    _, objs, _ = load_settings(CONFIG)
    q = build_queue(objs, ["building", "human"], directions=[0, 4, 8])
    assert [r.run_id for r in q] == [
        "building_d0", "human_d0", "building_d4", "human_d4", "building_d8", "human_d8"]
    # A filter keeps coarse-first order, not the order the caller listed.
    q2 = build_queue(objs, ["human"], directions=[1, 8, 0])
    assert [r.direction_index for r in q2] == [0, 8, 1]


def test_next_run_prefers_interrupted_runs():
    recs = [_rec("a_d0", "waiting"), _rec("b_d0", "running"), _rec("c_d0", "done")]
    assert next_run(recs).run_id == "b_d0"
    recs2 = [_rec("a_d0", "waiting"), _rec("b_d0", "stopped"), _rec("c_d0", "running")]
    assert next_run(recs2).run_id == "b_d0"  # first interrupted record in queue order


def test_next_run_falls_back_to_waiting_then_none():
    assert next_run([_rec("a_d0", "done"), _rec("b_d0", "waiting")]).run_id == "b_d0"
    assert next_run([_rec("a_d0", "done"), _rec("b_d0", "flagged"), _rec("c_d0", "failed")]) is None
    assert next_run([]) is None


def test_save_queue_atomic_roundtrip(tmp_path):
    path = tmp_path / "queue.json"
    recs = [
        RunRecord("airplane_d0", "airplane", 0, -10.0, "done", iterations=312,
                  converged=True, log="ok", machine="mac-1"),
        _rec("f1_car_d4", "flagged", iterations=1000, log="hit cap"),
    ]
    save_queue(path, recs)
    assert not (tmp_path / "queue.json.tmp").exists()
    assert json.loads(path.read_text())  # valid JSON
    assert load_queue(path) == recs
    save_queue(path, recs[:1])  # overwrite replaces
    assert load_queue(path) == recs[:1]


def test_load_queue_missing_file(tmp_path):
    assert load_queue(tmp_path / "nope.json") == []


def test_merge_keeps_done_status():
    existing = [_rec("a_d0", "done", iterations=300, converged=True), _rec("a_d4", "running")]
    new = [_rec("a_d0", "waiting"), _rec("a_d4", "waiting"), _rec("a_d8", "waiting")]
    merged = merge_queue(existing, new)
    assert [r.run_id for r in merged] == ["a_d0", "a_d4", "a_d8"]
    assert merged[0].status == "done" and merged[0].iterations == 300 and merged[0].converged
    assert merged[1].status == "running"
    assert merged[2].status == "waiting"


def test_merge_follows_new_order_and_keeps_existing_only_records():
    existing = [_rec("a_d0", "done"), _rec("old_d0", "done")]
    new = [_rec("a_d4"), _rec("a_d0")]
    merged = merge_queue(existing, new)
    # New order first; finished runs absent from `new` are appended so no result is lost.
    assert [r.run_id for r in merged] == ["a_d4", "a_d0", "old_d0"]
    assert merged[1].status == "done"


def test_reset_failed_puts_only_failed_records_back_to_waiting():
    recs = [_rec("a_d0", "failed", iterations=120, log="boom"), _rec("a_d1", "done", iterations=300, converged=True),
            _rec("a_d2", "stopped", iterations=80), _rec("a_d3", "failed", iterations=9)]
    assert reset_failed(recs) == 2
    assert [r.status for r in recs] == ["waiting", "done", "stopped", "waiting"]
    assert recs[0].iterations == 0 and recs[0].converged is False and recs[0].log == ""
    assert recs[1].iterations == 300 and recs[1].converged  # untouched
    assert recs[2].iterations == 80
    assert next_run(recs).run_id == "a_d2"  # the interrupted one is still resumed first
    assert reset_failed(recs) == 0
