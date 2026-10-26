"""Tests for solver.foam.runner and solver/run_openfoam.py.

Most tests drive the real runner through a FakeExecutor: a scripted stand-in for `docker` that
records every command and simulates just enough of a container (files, a solver that advances
100 iterations each time it is polled, force history, exit marker). The last test needs Docker.
"""
import json
import subprocess
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import trimesh

from solver import run_openfoam
from solver.foam.geometry import prepare_model
from solver.foam.queue import RunRecord, load_queue, save_queue
from solver.foam.runner import (EXIT_MARKER, FORCE_SEP, FoamContainer, RunContext, RunnerError,
                                SubprocessExecutor, check_mesh_log, mesh_cell_count, mesh_only, run_batch, run_one, stop_flag)
from solver.foam.settings import load_settings

ROOT = Path(__file__).resolve().parent.parent
SETTINGS, OBJECTS, _RAW = load_settings(ROOT / "config" / "exhibition_config.json")
KEY = "full/airplane_d0"  # where the case lives in the volume: <mode>/<run_id>
# The relevant part of a real checkMesh log ("cells:" sits in the Mesh stats block, indented).
CHECKMESH_LOG = (
    "Mesh stats\n    points:           1300\n    faces:            4000\n    internal faces:   3500\n"
    "    cells:            {cells}\n    faces per cell:   6\n\nMesh OK.\n"
)
STEP = 100  # iterations the fake solver advances per poll


def steady(it):  # drag and lift constant: converges as soon as the minimum iteration count is reached
    return 100.0, 20.0


def noisy(it):  # drag swings +-10%: never converges
    return 100.0 + (10.0 if it % 2 else -10.0), 20.0


class FakeExecutor:
    """Pretends to be docker + a container. `scenario(it) -> (drag, lift)` shapes the force history."""

    def __init__(self, scenario=steady, bad_mesh=False, fatal_runs=(), on_poll=None, docker_ok=True, cells=1234, mesh_log=None):
        self.calls: list[list[str]] = []
        self.files: dict[tuple[str, str], str] = {}  # (run_id, rel_path) -> text
        self.scenario, self.bad_mesh, self.fatal_runs, self.on_poll = scenario, bad_mesh, set(fatal_runs), on_poll
        self.docker_ok, self.cells, self.mesh_log = docker_ok, cells, mesh_log
        self.cases: set[str] = set()
        self.segments: dict[str, list[tuple[int, list[str]]]] = {}  # run -> [(start_time, rows)]
        self.iter: dict[str, int] = {}
        self.sim: dict | None = None  # the running (detached) solver
        self.polls = 0
        self.lock_log = []
        self.failing: dict[str, int] = {}  # script substring -> how many more times it fails with exit 1

    # -- helpers -----------------------------------------------------------------------------
    def cmds(self) -> list[str]:
        """The shell script of every `docker exec` (the plain `rm -rf` exec is shown as its argv)."""
        return [" ".join(c[3:]) if c[3] == "rm" else c[-1] for c in self.calls if c[:2] == ["docker", "exec"]]

    def count(self, text: str) -> int:
        return sum(text in c for c in self.cmds())

    @staticmethod
    def _cp(rc=0, out=""):
        return subprocess.CompletedProcess([], rc, stdout=out, stderr="")

    def _advance(self):
        sim = self.sim
        run = sim["run"]
        if sim["fatal"]:
            self.files[(run, "log.simpleFoam")] = "FOAM FATAL ERROR: boom"
            self._finish(1)
            return
        new = min(self.iter[run] + STEP, sim["cap"])
        rows = self.segments[run][-1][1]
        for it in range(self.iter[run] + 1, new + 1):
            drag, lift = self.scenario(it)
            rows.append(f"{it} ({drag} 0 {lift}) (0 0 0) (0 0 0)")
        self.iter[run] = new
        if self.stop_issued or new >= sim["cap"]:
            self._finish(0)

    def _finish(self, code):
        self.files[(self.sim["run"], EXIT_MARKER)] = f"{code}\n"
        self.sim = None

    # -- docker ------------------------------------------------------------------------------
    def run(self, args, check=True):
        self.calls.append(list(args))
        if args[:2] == ["docker", "info"]:
            return self._cp(0 if self.docker_ok else 1)
        if args[:2] == ["docker", "inspect"]:
            return self._cp(1)
        if args[:2] == ["docker", "cp"]:
            return self._copy(args)
        if args[:2] == ["docker", "exec"] or args[:3] == ["docker", "exec", "-d"]:
            return self._exec(args)
        return self._cp()

    def _copy(self, args):
        src, dst = args[2], args[3]
        if ":" not in src:  # copy in: docker cp <dir>/. name:/cases/<mode>/<run>
            self.cases.add(dst.split(":/cases/", 1)[1])
            return self._cp()
        mode, name, path = src.split(":/cases/", 1)[1].split("/", 2)
        run = f"{mode}/{name}"
        if path.startswith("log.") and (run, path) not in self.files:
            return self._cp(1)
        target = Path(dst) / Path(path).name
        if path in ("VTK", "postProcessing"):
            (target / "x").mkdir(parents=True, exist_ok=True)
            if path == "VTK":
                (target / "case_100.vtm").write_text("")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.files.get((run, path), ""))
        return self._cp()

    def _exec(self, args):
        detach = args[2] == "-d"
        if len(args) > 4 and args[3] == "mkdir":
            return self._cp()
        if len(args) > 4 and args[3] == "bash" and args[4] == "-c":
            return self._raw(args[-1])
        if len(args) > 4 and args[3] == "rm":
            run = args[-1][len("/cases/"):]
            self.cases.discard(run)
            for key in [k for k in self.files if k[0] == run]:
                del self.files[key]
            self.segments.pop(run, None)
            self.iter.pop(run, None)
            return self._cp()
        cmd = args[-1]
        run = cmd.split("cd /cases/")[1].split(";")[0]
        script = cmd.split(f"cd /cases/{run}; ", 1)[1]
        files = self.files
        if self._should_fail(script):
            return self._cp(1)

        if "blockMesh" in script:
            files[(run, "log.checkMesh")] = (self.mesh_log if self.mesh_log is not None else "Failed 2 mesh checks." if self.bad_mesh
                                             else CHECKMESH_LOG.format(cells=self.cells))
            return self._cp(1 if self.bad_mesh else 0)
        if "decomposePar" in script:
            files[(run, "processor0")] = ""
            return self._cp()
        if "stopAt -set writeNow" in script:
            self.stop_issued = True
            return self._cp()
        if "stopAt -set endTime" in script and "rm -f" in script:
            self.stop_issued = False
            files.pop((run, EXIT_MARKER), None)
            return self._cp()
        if "mpirun" in script and "simpleFoam" in script:
            assert detach, "simpleFoam must be launched detached"
            self.iter.setdefault(run, 0)
            files[(run, "log.simpleFoam")] = "trapFpe: Floating point exception trapping enabled\n"  # real banner
            self.segments.setdefault(run, []).append((self.iter[run], []))
            cap = self.cap_for(run)
            self.sim = {"run": run, "cap": cap, "fatal": run in self.fatal_runs}
            return self._cp()
        if script.startswith("for d in"):
            segs = self.segments.get(run, [])
            return self._cp(0, "".join(f"{FORCE_SEP}\n" + "\n".join(rows) + "\n" for _, rows in segs))
        if script.startswith("grep"):
            lines = [ln for ln in files.get((run, "log.simpleFoam"), "").splitlines()
                     if ("FOAM FATAL" in ln or "Floating point exception" in ln) and "trapping enabled" not in ln]
            return self._cp(0 if lines else 1)
        if "reconstructPar" in script:
            return self._cp()
        if script.startswith("cat "):
            rel = script.split()[-1]
            return self._cp(0, files[(run, rel)]) if (run, rel) in files else self._cp(1)
        raise AssertionError(f"unscripted command: {script}")

    def _should_fail(self, script):
        for text, left in self.failing.items():
            if text in script and left > 0:
                self.failing[text] = left - 1
                return True
        return False

    def _raw(self, script):
        """Scripts run with plain `bash -c`: probes (exit 0 / 3), pgrep and the active-run lock."""
        if self._should_fail(script):
            return self._cp(1)  # docker itself failed
        if "pgrep" in script:
            self.polls += 1
            if self.on_poll:
                self.on_poll(self.polls)
            if self.sim:
                self._advance()
            return self._cp(0 if self.sim else 3)
        if "test -e" in script:
            path = script.split("test -e ", 1)[1].split(";")[0].removeprefix("/cases/")
            mode, name, rel = path.split("/", 2)
            return self._cp(0 if (f"{mode}/{name}", rel) in self.files else 3)
        if "/cases/.active" in script:
            if script.startswith("cat"):
                return self._cp(0, self.files.get(("", ".active"), ""))
            if script.startswith("echo"):
                self.files[("", ".active")] = script.split()[1]
                self.lock_log.append(script.split()[1])
            elif script.startswith("rm"):
                self.files.pop(("", ".active"), None)
            return self._cp()
        raise AssertionError(f"unscripted raw command: {script}")

    lock_log: list = []
    stop_issued = False
    cap_for = staticmethod(lambda run: 300)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    out = {}
    for obj_id in ("airplane", "f1_car"):
        src = tmp_path_factory.mktemp(obj_id) / "box.stl"
        trimesh.creation.box(extents=(4.0, 1.0, 1.0)).export(src)
        out[obj_id] = prepare_model(src, replace(OBJECTS[obj_id], real_size_m=4.0, size_axis="x"))
    return out


def make_ctx(tmp_path, models, ex, cap=300):
    container = FoamContainer(SETTINGS, "navflow-test", ex)
    ex.cap_for = lambda run: cap
    return RunContext(SETTINGS, OBJECTS, models, tmp_path / "full", cap, container, poll_s=0.0, say=lambda m: None)


def waiting(obj_id="airplane", idx=0, status="waiting"):
    return RunRecord(f"{obj_id}_d{idx}", obj_id, idx, OBJECTS[obj_id].directions_deg[idx], status)


def test_fresh_run_meshes_then_launches_detached(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    rec = run_one(waiting(), ctx)
    cmds = ex.cmds()
    order = [next(i for i, c in enumerate(cmds) if key in c)
             for key in ("blockMesh", "decomposePar", "potentialFoam", "mpirun -np 4 simpleFoam")]
    assert order == sorted(order)
    launch = next(c for c in ex.calls if "simpleFoam -parallel" in c[-1])
    assert launch[:3] == ["docker", "exec", "-d"]
    assert "OMPI" not in launch[-1]
    assert "snappyHexMesh -overwrite" in cmds[order[0]] and "checkMesh" in cmds[order[0]]
    assert (tmp_path / "full/airplane/airplane_d0/case/system/controlDict").is_file()
    assert rec.status == "done"


def test_resume_skips_meshing_when_processor0_exists(tmp_path, models):
    ex = FakeExecutor(scenario=noisy)
    ctx = make_ctx(tmp_path, models, ex, cap=1000)
    ex.files[(KEY, "processor0")] = ""
    ex.cases.add(KEY)
    ex.iter[KEY] = 150
    ex.segments[KEY] = [(0, [f"{i} (100 0 20) (0 0 0) (0 0 0)" for i in range(1, 151)])]
    rec = waiting(status="stopped")
    rec.iterations = 150
    ex.cap_for = lambda run: 300
    ctx = replace(ctx, cap=300)
    run_one(rec, ctx)
    assert ex.count("blockMesh") == 0 and ex.count("decomposePar") == 0 and ex.count("potentialFoam") == 0
    assert ex.count("mpirun -np 4 simpleFoam") == 1
    assert ex.count("stopAt -set endTime") == 1  # a leftover writeNow from the stop must be undone
    assert rec.iterations == 300 and rec.status == "flagged"


def test_restarts_when_case_missing_from_volume(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    rec = run_one(waiting(status="running"), ctx)  # "running" in the queue, but nothing in the volume
    assert ex.count("blockMesh") == 1 and ex.count("decomposePar") == 1
    assert ex.cmds().index(next(c for c in ex.cmds() if "rm -rf" in c)) < ex.cmds().index(
        next(c for c in ex.cmds() if "blockMesh" in c)), "stale case must be removed before the fresh copy"
    assert rec.status == "done"


def test_fresh_run_deletes_stale_local_case(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    stale = tmp_path / "full/airplane/airplane_d0/case/processor0/stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    run_one(waiting(), ctx)
    assert not stale.exists()


def test_stop_flag_sets_write_now_and_status_stopped(tmp_path, models):
    ex = FakeExecutor(scenario=noisy)
    ctx = make_ctx(tmp_path, models, ex)
    ex.on_poll = lambda n: stop_flag(ctx.root).touch() if n == 2 else None  # poll 1 is the pre-launch check
    ctx.root.mkdir(parents=True)
    rec = run_one(waiting(), ctx)
    assert rec.status == "stopped"
    assert ex.count("stopAt -set writeNow") == 1
    assert ex.count("reconstructPar") == 0  # nothing is exported, the case stays for the resume
    assert (KEY, "processor0") in ex.files


def test_stop_flag_during_meshing_keeps_prepared_case_without_launching(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.on_poll = lambda n: stop_flag(ctx.root).touch() if n == 1 else None
    ctx.root.mkdir(parents=True)
    rec = run_one(waiting(), ctx)
    assert rec.status == "stopped" and ex.count("mpirun -np 4 simpleFoam") == 0
    assert (KEY, "processor0") in ex.files  # the meshed, decomposed case is kept for the resume


def test_converged_run_is_done_instant(tmp_path, models):
    ex = FakeExecutor(scenario=steady)
    ctx = make_ctx(tmp_path, models, ex)
    rec = run_one(waiting(), ctx)
    assert rec.status == "done" and rec.converged
    assert ex.count("stopAt -set writeNow") == 1
    info = json.loads((tmp_path / "full/airplane/airplane_d0/result/run.json").read_text())
    assert info == {"run_id": "airplane_d0", "obj_id": "airplane", "direction_index": 0,
                    "direction_deg": OBJECTS["airplane"].directions_deg[0],
                    "speed_m_s": OBJECTS["airplane"].speed_m_s, "iterations": rec.iterations,
                    "converged": True, "fields": "instant", "status": "done", "cells": 1234,
                    "mesh_warnings": []}
    assert list((tmp_path / "full/airplane/airplane_d0/result/VTK").glob("*.vtm"))
    assert (tmp_path / "full/airplane/airplane_d0/result/postProcessing").is_dir()
    assert ex.count("rm -rf /cases/full/airplane_d0") == 2  # once before the fresh copy, once after export


def test_cap_run_is_flagged_mean(tmp_path, models):
    ex = FakeExecutor(scenario=noisy)
    ctx = make_ctx(tmp_path, models, ex, cap=300)
    rec = run_one(waiting(), ctx)
    assert rec.status == "flagged" and not rec.converged and rec.iterations == 300
    assert ex.count("stopAt -set writeNow") == 0
    info = json.loads((tmp_path / "full/airplane/airplane_d0/result/run.json").read_text())
    assert info["fields"] == "mean" and info["status"] == "flagged" and info["converged"] is False


def test_fatal_log_marks_failed_and_batch_continues(tmp_path, models):
    ex = FakeExecutor(scenario=steady, fatal_runs={KEY})
    ctx = make_ctx(tmp_path, models, ex)
    handled = run_batch(["airplane", "f1_car"], ctx, "laptop", directions=[0])
    assert [(r.run_id, r.status) for r in handled] == [("airplane_d0", "failed"), ("f1_car_d0", "done")]
    saved = {r.run_id: r for r in load_queue(tmp_path / "full/queue_laptop.json")}
    assert saved["airplane_d0"].status == "failed" and saved["f1_car_d0"].machine == "laptop"
    assert (tmp_path / "full/airplane/airplane_d0/result/log.simpleFoam").is_file()
    assert not (tmp_path / "full/airplane/airplane_d0/result/run.json").exists()


def test_bad_mesh_marks_failed(tmp_path, models):
    ex = FakeExecutor(bad_mesh=True)
    ctx = make_ctx(tmp_path, models, ex)
    rec = run_one(waiting(), ctx)
    assert rec.status == "failed" and "Mesh OK." in rec.log
    assert ex.count("decomposePar") == 0 and ex.count("mpirun -np 4 simpleFoam") == 0
    assert (tmp_path / "full/airplane/airplane_d0/result/log.checkMesh").read_text().startswith("Failed")


def test_batch_returns_after_stopped_and_resumes_from_queue(tmp_path, models):
    ex = FakeExecutor(scenario=noisy)
    ctx = make_ctx(tmp_path, models, ex)
    ex.on_poll = lambda n: stop_flag(ctx.root).touch() if n == 2 else None
    handled = run_batch(["airplane"], ctx, "laptop", directions=[0, 4])
    assert [(r.run_id, r.status) for r in handled] == [("airplane_d0", "stopped")]
    assert [r.status for r in load_queue(tmp_path / "full/queue_laptop.json")] == ["stopped", "waiting"]
    # Second invocation: the stale flag is removed, the stopped run resumes (no re-mesh) and the queue continues.
    ex.on_poll = None
    handled = run_batch(["airplane"], ctx, "laptop", directions=[0, 4])
    assert [(r.run_id, r.status) for r in handled] == [("airplane_d0", "flagged"), ("airplane_d4", "flagged")]
    assert ex.count("blockMesh") == 2  # d0 once (not re-meshed on resume) + d4


def test_batch_ignores_objects_not_requested(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ctx.root.mkdir(parents=True)
    save_queue(ctx.root / "queue_laptop.json", [waiting("f1_car", 0)])
    handled = run_batch(["airplane"], ctx, "laptop", directions=[0])
    assert [r.run_id for r in handled] == ["airplane_d0"]


def test_attaches_to_running_solver_instead_of_launching_second(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.files[(KEY, "processor0")] = ""
    ex.cases.add(KEY)
    ex.iter[KEY] = 100
    ex.segments[KEY] = [(0, [f"{i} (100 0 20) (0 0 0) (0 0 0)" for i in range(1, 101)])]
    ex.sim = {"run": KEY, "cap": 300, "fatal": False}  # still running in the container
    ex.files[("", ".active")] = KEY  # ... and it is OUR solver
    rec = run_one(waiting(status="running"), ctx)
    assert ex.count("mpirun -np 4 simpleFoam") == 0
    assert rec.status == "done"


def test_solver_vanishing_without_marker_raises_and_stays_resumable(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.sim = None
    original = ex._exec

    def exec_without_launch(args):
        if args[2] == "-d":
            return ex._cp()  # the launch silently does nothing
        return original(args)

    ex._exec = exec_without_launch
    rec = waiting()
    with pytest.raises(RunnerError):
        run_one(rec, ctx)
    assert rec.status == "running"


def test_container_start_creates_or_starts(tmp_path):
    ex = FakeExecutor()
    FoamContainer(SETTINGS, "navflow-laptop", ex).start()
    run_call = next(c for c in ex.calls if c[:2] == ["docker", "run"])
    assert "navflow-cases:/cases" in run_call and "OMPI_ALLOW_RUN_AS_ROOT=1" in run_call
    assert run_call[-3:] == [SETTINGS.image, "sleep", "infinity"]

    class Existing(FakeExecutor):
        def run(self, args, check=True):
            self.calls.append(list(args))
            return self._cp(0)

    ex = Existing()
    FoamContainer(SETTINGS, "navflow-laptop", ex).start()
    assert ex.calls[-1] == ["docker", "start", "navflow-laptop"]


def test_docker_not_running_message(capsys):
    for argv in (["status"], ["stop"], ["run", "--objects", "airplane"], ["test", "--object", "airplane"]):
        assert run_openfoam.main(argv, ex=FakeExecutor(docker_ok=False)) == 2
        assert "Docker Desktop isn't running — open it and try again." in capsys.readouterr().err


# --- fix round 1: namespacing, lock, lost stop, Docker errors, divergence, export failure ---

def test_modes_get_separate_case_paths(tmp_path, models):
    ex = FakeExecutor()
    for mode in ("test", "preview", "full"):
        ctx = make_ctx(tmp_path, models, ex)
        ctx = replace(ctx, root=tmp_path / mode)
        run_one(waiting(), ctx)
    copies = [c[3] for c in ex.calls if c[:2] == ["docker", "cp"] and ":" in c[3] and "/." in c[2]]
    assert copies == ["navflow-test:/cases/test/airplane_d0", "navflow-test:/cases/preview/airplane_d0",
                      "navflow-test:/cases/full/airplane_d0"]
    for mode in ("test", "preview", "full"):
        assert ex.count(f"cd /cases/{mode}/airplane_d0;") > 0
        assert ex.count(f"rm -rf /cases/{mode}/airplane_d0") == 2
    assert ex.count("rm -rf /cases/airplane_d0") == 0  # the un-namespaced path is never touched


def test_live_solver_of_another_case_is_refused_not_attached(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.sim = {"run": "test/airplane_d4", "cap": 1000, "fatal": False}  # a long test run owns the container
    ex.files[("", ".active")] = "test/airplane_d4"
    ex.iter["test/airplane_d4"], ex.segments["test/airplane_d4"] = 0, [(0, [])]
    rec = waiting(status="running")
    with pytest.raises(RunnerError, match="test/airplane_d4"):
        run_one(rec, ctx)
    assert ex.count("rm -rf") == 0 and ex.count("blockMesh") == 0 and ex.count("mpirun") == 0
    assert not any(c[:2] == ["docker", "cp"] for c in ex.calls)
    assert ex.sim is not None and ex.files[("", ".active")] == "test/airplane_d4"  # untouched
    assert rec.status == "running"


def test_active_lock_written_before_launch_and_cleared_after(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    run_one(waiting(), ctx)
    assert ex.lock_log == [KEY]
    assert ("", ".active") not in ex.files
    cmds = ex.cmds()
    assert cmds.index(f"echo {KEY} > /cases/.active") < cmds.index(next(c for c in cmds if "mpirun -np 4 simpleFoam" in c))


def test_lost_stop_request_is_retried(tmp_path, models):
    ex = FakeExecutor(scenario=noisy)
    ctx = make_ctx(tmp_path, models, ex, cap=1000)
    ex.failing = {"stopAt -set writeNow": 1}  # the first attempt fails (e.g. a Docker hiccup)
    ex.on_poll = lambda n: stop_flag(ctx.root).touch() if n == 2 else None
    ctx.root.mkdir(parents=True)
    rec = run_one(waiting(), ctx)
    assert ex.count("stopAt -set writeNow") == 2  # failed once, retried on the next poll, then not repeated
    assert rec.status == "stopped"


def test_docker_error_is_not_mistaken_for_missing_volume(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.failing = {"test -e": 1}  # the processor0 probe gets a docker failure, not "absent"
    rec = waiting(status="running")
    with pytest.raises(RunnerError, match="nothing was changed"):
        run_one(rec, ctx)
    assert ex.count("rm -rf") == 0 and ex.count("blockMesh") == 0
    assert not any(c[:2] == ["docker", "cp"] for c in ex.calls)
    assert rec.status == "running"


def test_exists_distinguishes_absent_from_docker_failure():
    ex = FakeExecutor()
    box = FoamContainer(SETTINGS, "n", ex)
    ex.files[("full/x", "processor0")] = ""
    assert box.exists("full/x", "processor0") is True
    assert box.exists("full/x", "nope") is False
    ex.failing = {"test -e": 1}
    with pytest.raises(RunnerError):
        box.exists("full/x", "processor0")


def test_nan_forces_stop_the_solver_early_and_fail(tmp_path, models):
    def blows_up(it):
        return (float("nan"), 20.0) if it > 150 else (100.0 + it % 3, 20.0)

    ex = FakeExecutor(scenario=blows_up)
    ctx = make_ctx(tmp_path, models, ex, cap=1000)
    rec = run_one(waiting(), ctx)
    assert rec.status == "failed"
    assert ex.count("stopAt -set writeNow") == 1  # told to stop as soon as the NaN showed up
    assert rec.iterations < 1000
    assert ex.count("reconstructPar") == 0


def test_export_failure_leaves_run_resumable_and_rerun_re_exports(tmp_path, models):
    ex = FakeExecutor(scenario=steady)
    ctx = make_ctx(tmp_path, models, ex)
    ex.failing = {"reconstructPar": 1}
    rec = waiting()
    with pytest.raises(RunnerError, match="export"):
        run_one(rec, ctx)
    assert rec.status == "running"  # not failed: next_run will pick it up again
    assert ex.count("rm -rf /cases/full/airplane_d0") == 1  # only the pre-meshing wipe; the solved case is kept
    launches = ex.count("mpirun -np 4 simpleFoam")
    rec = run_one(rec, ctx)
    assert ex.count("mpirun -np 4 simpleFoam") == launches  # solver is not run again
    assert ex.count("blockMesh") == 1
    assert rec.status == "done"


def test_finished_unseen_goes_straight_to_classification(tmp_path, models):
    ex = FakeExecutor()
    ctx = make_ctx(tmp_path, models, ex)
    ex.files[(KEY, "processor0")] = ""
    ex.files[(KEY, EXIT_MARKER)] = "0\n"  # the solver finished while Python was dead
    ex.files[(KEY, "log.simpleFoam")] = "trapFpe: Floating point exception trapping enabled\nEnd\n"
    ex.cases.add(KEY)
    ex.iter[KEY] = 300
    ex.segments[KEY] = [(0, [f"{i} (100 0 20) (0 0 0) (0 0 0)" for i in range(1, 301)])]
    rec = run_one(waiting(status="running"), ctx)
    assert ex.count("mpirun -np 4 simpleFoam") == 0 and ex.count("blockMesh") == 0
    assert ex.count("stopAt -set writeNow") == 0
    assert rec.status == "done" and rec.iterations == 300


# --- final fix wave: retry, mesh size, error handling, status ---

def test_mesh_cell_count_parses_checkmesh_and_ignores_other_lines():
    assert mesh_cell_count(CHECKMESH_LOG.format(cells=1234567)) == 1234567
    assert mesh_cell_count("Mesh OK.\n") is None
    assert mesh_cell_count("    cells: lots\n") is None
    assert mesh_cell_count("    boundary cells:   55\n    cells:   42\n") == 42  # only the Mesh stats line


def test_cell_count_is_reported_and_saved_in_run_json(tmp_path, models):
    ex = FakeExecutor(cells=1234)
    said = []
    ctx = replace(make_ctx(tmp_path, models, ex), say=said.append)
    rec = run_one(waiting(), ctx)
    assert rec.status == "done" and rec.cells == 1234
    assert any("1234 cells" in m and "WARNING" not in m for m in said)
    info = json.loads((tmp_path / "full/airplane/airplane_d0/result/run.json").read_text())
    assert info["cells"] == 1234


def test_cell_count_near_the_cap_warns(tmp_path, models):
    ex = FakeExecutor(cells=5_500_000)  # 92% of maxGlobalCells (6 000 000)
    said = []
    ctx = replace(make_ctx(tmp_path, models, ex), say=said.append)
    run_one(waiting(), ctx)
    warning = [m for m in said if "WARNING" in m]
    assert warning and "maxGlobalCells" in warning[0] and "5500000 cells" in warning[0]
    said.clear()
    ex2 = FakeExecutor(cells=5_300_000)  # 88%: no warning
    ctx2 = replace(make_ctx(tmp_path / "b", models, ex2), say=said.append)
    run_one(waiting(), ctx2)
    assert not [m for m in said if "WARNING" in m]


def test_retry_failed_resets_only_this_machines_failed_runs(tmp_path, models, monkeypatch, capsys):
    ex = FakeExecutor(scenario=steady)
    runs = tmp_path / "runs" / "full"
    runs.mkdir(parents=True)
    monkeypatch.setattr(run_openfoam, "_runs_root", lambda preview: runs)
    failed = waiting("airplane", 0, "failed")
    failed.iterations, failed.log = 77, "boom"
    save_queue(runs / "queue_laptop.json", [failed, waiting("airplane", 4, "done")])
    other = waiting("f1_car", 0, "failed")
    save_queue(runs / "queue_mini.json", [other])
    monkeypatch.setattr(run_openfoam, "_build_context", lambda args, ex, root, cap, obj_ids, **kw: make_ctx(tmp_path, models, ex))
    monkeypatch.setattr(run_openfoam, "run_batch", lambda *a, **k: [])

    assert run_openfoam.main(["run", "--objects", "airplane", "--retry-failed"], ex=ex) == 0
    assert "Reset 1 failed run" in capsys.readouterr().out
    laptop = {r.run_id: r for r in load_queue(runs / "queue_laptop.json")}
    assert laptop["airplane_d0"].status == "waiting" and laptop["airplane_d0"].iterations == 0
    assert laptop["airplane_d4"].status == "done"
    assert load_queue(runs / "queue_mini.json")[0].status == "failed"  # another machine's queue is untouched


def test_failed_run_is_run_again_after_reset(tmp_path, models):
    ex = FakeExecutor(scenario=steady, fatal_runs={KEY})
    ctx = make_ctx(tmp_path, models, ex)
    handled = run_batch(["airplane"], ctx, "laptop", directions=[0])
    assert handled[0].status == "failed"
    assert run_batch(["airplane"], ctx, "laptop", directions=[0]) == []  # failed runs are skipped forever ...
    qpath = ctx.root / "queue_laptop.json"
    records = load_queue(qpath)
    from solver.foam.queue import reset_failed
    assert reset_failed(records) == 1
    save_queue(qpath, records)
    ex.fatal_runs = set()
    handled = run_batch(["airplane"], ctx, "laptop", directions=[0])  # ... until they are reset
    assert [(r.run_id, r.status) for r in handled] == [("airplane_d0", "done")]


def test_docker_command_error_is_a_clean_stop(tmp_path, monkeypatch, capsys):
    class Exploding(FakeExecutor):
        def run(self, args, check=True):
            if args[:2] == ["docker", "run"]:
                raise subprocess.CalledProcessError(125, args, stderr="boom")
            return super().run(args, check)

    monkeypatch.setattr(run_openfoam, "_build_context", lambda args, ex, root, cap, obj_ids, **kw: make_ctx(tmp_path, {}, ex))
    code = run_openfoam.main(["run", "--objects", "airplane"], ex=Exploding())
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("Stopped:") and "docker" in err


def test_status_honours_machine(tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs" / "full"
    runs.mkdir(parents=True)
    monkeypatch.setattr(run_openfoam, "_runs_root", lambda preview: runs)
    save_queue(runs / "queue_laptop.json", [waiting("airplane", 0, "done")])
    save_queue(runs / "queue_mini.json", [waiting("f1_car", 0, "running")])
    assert run_openfoam.main(["status", "--machine", "mini"], ex=FakeExecutor()) == 0
    out = capsys.readouterr().out
    assert "f1_car_d0" in out and "airplane_d0" not in out
    assert run_openfoam.main(["status"], ex=FakeExecutor()) == 0  # no --machine: every machine's queue
    out = capsys.readouterr().out
    assert "f1_car_d0" in out and "airplane_d0" in out
    assert run_openfoam.main(["status", "--machine", "ghost"], ex=FakeExecutor()) == 0
    assert "No queue found" in capsys.readouterr().out


# --- Docker integration ---------------------------------------------------------------------

def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


class RecordingExecutor(SubprocessExecutor):
    def __init__(self):
        self.commands: list[str] = []

    def run(self, args, check=True):
        self.commands.append(" ".join(args))
        return super().run(args, check)


@pytest.mark.docker
def test_short_real_run(tmp_path):
    """Coarse airplane run in real OpenFOAM: stop it part-way, then resume it to the cap."""
    if not docker_available():
        pytest.skip("Docker is not running")
    cap = 300  # the coarse solver does ~15 iterations/s, so 120 would race the stop request
    settings = replace(SETTINGS, base_cells_per_size=4, surface_refinement=(2, 3),
                       volume=f"navflow-test-{uuid.uuid4().hex[:8]}")
    cfd = OBJECTS["airplane"]
    model0 = prepare_model(ROOT / "assets/models/airplane/model.stl", cfd)
    ex = RecordingExecutor()
    name = f"navflow-test-{uuid.uuid4().hex[:8]}"
    container = FoamContainer(settings, name, ex)
    root = tmp_path / "full"
    ctx = RunContext(settings, OBJECTS, {"airplane": model0}, root, cap, container, poll_s=1.0, say=print)
    rec = RunRecord("airplane_d0", "airplane", 0, cfd.directions_deg[0], "waiting")
    result = root / "airplane/airplane_d0/result"

    def stop_when_60():  # create the stop flag once force.dat shows >= 60 iterations (a save at 50 exists)
        while not stop_flag(root).exists():
            try:
                text = FoamContainer(settings, name, SubprocessExecutor()).read_forces("full/airplane_d0")
            except RunnerError:  # container not up yet
                time.sleep(1)
                continue
            rows = [ln for t in text for ln in t.splitlines() if ln.strip() and not ln.startswith("#")]
            if len(rows) >= 60:
                root.mkdir(parents=True, exist_ok=True)
                stop_flag(root).touch()
            time.sleep(1)

    try:
        container.start()
        watcher = threading.Thread(target=stop_when_60, daemon=True)
        watcher.start()
        rec = run_one(rec, ctx)
        watcher.join(timeout=30)
        assert rec.status == "stopped", rec
        stopped_at = rec.iterations
        assert 60 <= stopped_at < cap
        assert container.exists(KEY, "processor0")
        assert not result.exists()

        stop_flag(root).unlink()
        before = len(ex.commands)
        rec = run_one(rec, ctx)
        resumed = " ".join(ex.commands[before:])
        assert "blockMesh" not in resumed and "decomposePar" not in resumed  # processor0 reused
        assert rec.status in ("done", "flagged"), rec
        assert rec.iterations > stopped_at
        info = json.loads((result / "run.json").read_text())
        assert info["status"] in ("done", "flagged") and info["iterations"] == rec.iterations
        assert list((result / "VTK").glob("*.vtm"))
        assert not container.exists(KEY, "processor0")  # case freed after export
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        subprocess.run(["docker", "volume", "rm", settings.volume], capture_output=True)


# --- checkMesh pass rule: quality-only warnings are accepted, hard errors fail ---

F1_CHECKMESH = (Path(__file__).parent / "f1_checkMesh_sample.log").read_text()  # a real F1 run (direction 4)


def test_check_mesh_log_real_f1_sample_passes_with_one_skewness_warning():
    ok, warnings = check_mesh_log(F1_CHECKMESH)
    assert ok and len(warnings) == 1
    assert warnings[0].startswith("Max skewness = 4.23446") and not warnings[0].startswith("*")


def test_check_mesh_log_mesh_ok_has_no_warnings():
    assert check_mesh_log(CHECKMESH_LOG.format(cells=10)) == (True, [])


def test_check_mesh_log_hard_errors_fail():
    ok, errs = check_mesh_log("    Cell volumes\n ***Zero or negative cell volume detected\n\nFailed 1 mesh checks.\n")
    assert not ok and errs == ["Zero or negative cell volume detected"]
    mixed = F1_CHECKMESH.replace("Failed 1 mesh checks.", " ***Error in face pyramids: 3 faces pyramids are incorrectly oriented\n\nFailed 2 mesh checks.")
    ok, errs = check_mesh_log(mixed)
    assert not ok and errs == ["Error in face pyramids: 3 faces pyramids are incorrectly oriented"]


def test_check_mesh_log_quality_pass_needs_a_complete_log():
    skew_line = " ***Max skewness = 4.23446, 1 highly skew faces detected which may impair the quality of the results\n"
    unexplained = ("Checking topology...\n    Face vertices FAILED\n" + skew_line + "\nFailed 2 mesh checks.\n\nEnd\n")
    assert check_mesh_log(unexplained)[0] is False  # 2 failed checks, only 1 *** line
    fatal = skew_line + "--> FOAM FATAL ERROR:\nsomething broke\n"
    assert check_mesh_log(fatal)[0] is False  # crashed after the skew line
    assert check_mesh_log(F1_CHECKMESH.rsplit("Failed 1 mesh checks.", 1)[0])[0] is False  # truncated after it
    assert check_mesh_log(F1_CHECKMESH.rsplit("End", 1)[0])[0] is False  # no final End line
    assert check_mesh_log(F1_CHECKMESH + "Floating point exception (core dumped)\n")[0] is False
    assert check_mesh_log(F1_CHECKMESH)[0] is True


def test_old_queue_record_without_mesh_warnings_loads_with_empty_list(tmp_path):
    p = tmp_path / "queue.json"
    p.write_text(json.dumps([{"run_id": "airplane_d0", "obj_id": "airplane", "direction_index": 0,
                              "direction_deg": 0.0, "status": "waiting", "iterations": 0, "converged": False,
                              "log": "", "machine": "", "cells": 0}]))
    assert load_queue(p)[0].mesh_warnings == []


def test_mesh_only_log_mentions_accepted_quality_warnings(tmp_path, models):
    ex = FakeExecutor(mesh_log=F1_CHECKMESH)
    ctx = replace(make_ctx(tmp_path, models, ex), say=lambda m: None)
    rec = mesh_only(waiting(), ctx)
    assert rec.status != "failed" and "1 quality warning" in rec.log


def test_check_mesh_log_without_ok_or_warnings_fails():
    assert check_mesh_log("") == (False, [])
    assert check_mesh_log("Failed 2 mesh checks.") == (False, [])


def test_runner_continues_past_quality_warnings_and_records_them(tmp_path, models):
    ex = FakeExecutor(mesh_log=F1_CHECKMESH)
    said = []
    ctx = replace(make_ctx(tmp_path, models, ex), say=said.append)
    rec = run_one(waiting(), ctx)
    assert rec.status == "done" and ex.count("decomposePar") == 1
    assert any("WARNING" in m and "skewness" in m for m in said)
    info = json.loads((tmp_path / "full/airplane/airplane_d0/result/run.json").read_text())
    assert len(info["mesh_warnings"]) == 1 and "skewness" in info["mesh_warnings"][0]


def test_runner_still_fails_on_a_hard_mesh_error(tmp_path, models):
    ex = FakeExecutor(mesh_log=" ***Zero or negative cell volume detected\n\nFailed 1 mesh checks.\n")
    ctx = make_ctx(tmp_path, models, ex)
    rec = run_one(waiting(), ctx)
    assert rec.status == "failed" and "negative cell volume" in rec.log
    assert ex.count("decomposePar") == 0
