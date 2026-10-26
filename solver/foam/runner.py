"""Resumable Docker batch runner: drives OpenFOAM v2512 for every (object, wind direction) run.

Everything OpenFOAM runs inside one long-lived container (`navflow-<machine>`) whose Docker
volume `navflow-cases` holds the cases at /cases/<run_id>. The Python side only writes the case,
copies it in, issues shell commands, polls the force history and copies the results out.

Resuming is the key requirement (about 36 multi-hour runs; laptops sleep, Docker restarts):
  * the queue (`<root>/queue_<machine>.json`) records every run's status, saved atomically;
  * a run whose `processor0` directory is still in the volume is NOT re-meshed - simpleFoam
    continues with `startFrom latestTime` from the last save (every 50 iterations, or at once
    when we stop it with `stopAt writeNow`);
  * a fresh run first deletes any old copy of its case, so stale processor*/polyMesh/time
    directories can never leak into a resume;
  * cases live at /cases/<mode>/<run_id> (mode = test / preview / full) and /cases/.active names the
    case whose solver is running: one solver per container, and a run never attaches to or wipes
    another run's case;
  * if the Python process died while simpleFoam kept running in the container, the next start
    simply attaches to it instead of launching a second solver on the same case.

Maths link (IGCSE 0606): the run is stopped by a "small increment" test on drag and lift
(see solver/foam/convergence.py: delta F / F < 0.002 over the last 50 iterations), and the
iteration count is the x-axis of the force history we differentiate numerically.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import trimesh

from solver.foam.case_writer import RunSpec, write_case
from solver.foam.convergence import is_converged, is_diverged, load_forces
from solver.foam.geometry import run_geometry
from solver.foam.queue import RunRecord, build_queue, load_queue, merge_queue, next_run, save_queue
from solver.foam.settings import FoamSettings, ObjectCfd

# The OpenFOAM environment script inside opencfd/openfoam-default:2512. `bash -lc` already has the
# environment, but sourcing it explicitly makes every command independent of the login shell.
FOAM_BASHRC = "/usr/lib/openfoam/openfoam2512/etc/bashrc"

# Written by the detached solver wrapper when simpleFoam exits; holds its exit code. It is the
# authoritative "solver finished" signal (pgrep alone cannot tell "not started yet" from "ended").
EXIT_MARKER = ".simpleFoam.exit"
FORCE_SEP = "@@FORCE@@"
DEAD_POLLS = 3  # polls with no simpleFoam process and no exit marker before we call it a crash

MESH_SCRIPT = (
    "blockMesh > log.blockMesh 2>&1 "
    "&& surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1 "
    "&& snappyHexMesh -overwrite > log.snappyHexMesh 2>&1 "
    "&& checkMesh > log.checkMesh 2>&1"
)
LOGS = ("log.blockMesh", "log.surfaceFeatureExtract", "log.snappyHexMesh", "log.checkMesh",
        "log.decomposePar", "log.potentialFoam", "log.simpleFoam")
VTK_FIELDS = "(U p k omega nut UMean pMean kMean)"
# Every solver log starts with the banner "trapFpe: Floating point exception trapping enabled", which
# is not an error, so lines containing "trapping enabled" are ignored.
FATAL_SCRIPT = "grep -E 'FOAM FATAL|Floating point exception' log.simpleFoam | grep -qv 'trapping enabled'"


class RunnerError(RuntimeError):
    """The solver vanished without finishing, or Docker misbehaved; the run stays resumable."""


class Executor(Protocol):
    def run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess: ...


class SubprocessExecutor:
    """Runs argv lists for real. Children get their own session so a terminal Ctrl-C reaches only
    this Python process (which turns it into a graceful stop) and never kills a `docker exec`."""

    def run(self, args: list[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, check=check, start_new_session=True)


class FoamContainer:
    """One long-lived OpenFOAM container that mounts the cases volume at /cases.

    A "case" is addressed by its key `<mode>/<run_id>` (mode = test / preview / full), so the three
    kinds of run can never see or wipe each other's directories. /cases/.active holds the key of
    the case whose solver is running, so only one solver runs per container at a time.
    """

    def __init__(self, s: FoamSettings, name: str, ex: Executor):
        self.s, self.name, self.ex = s, name, ex

    def _docker(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return self.ex.run(["docker", *args], check=check)

    def _raw(self, script: str, check: bool = False) -> subprocess.CompletedProcess:
        """Run a plain shell script in the container (no OpenFOAM environment, no cd)."""
        return self._docker("exec", self.name, "bash", "-c", script, check=check)

    def _probe(self, test: str, what: str) -> bool:
        """Evaluate a shell test: exit 0 -> True, exit 3 -> False. Any other exit code means docker
        itself failed (daemon down, container stopped), which must never be mistaken for "absent"."""
        r = self._raw(f"if {test}; then exit 0; else exit 3; fi")
        if r.returncode in (0, 3):
            return r.returncode == 0
        raise RunnerError(f"docker exec failed (exit {r.returncode}) while checking {what}; "
                          "nothing was changed - fix Docker and run again")

    def start(self) -> None:
        """Create the container, or start it if it already exists (e.g. after a Docker restart)."""
        if self._docker("inspect", self.name, check=False).returncode == 0:
            self._docker("start", self.name)
            return
        self._docker(
            "run", "-d", "--name", self.name, "-v", f"{self.s.volume}:/cases",
            "-e", "OMPI_ALLOW_RUN_AS_ROOT=1", "-e", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
            self.s.image, "sleep", "infinity",
        )

    def copy_in(self, local_case: Path, case: str) -> None:
        self._docker("exec", self.name, "mkdir", "-p", f"/cases/{case}")
        self._docker("cp", f"{local_case}/.", f"{self.name}:/cases/{case}")

    def sh(self, case: str, script: str, detach: bool = False, check: bool = True) -> subprocess.CompletedProcess:
        """Run `script` in /cases/<case> with the OpenFOAM environment loaded."""
        command = f"source {FOAM_BASHRC}; cd /cases/{case}; {script}"
        flags = ["-d"] if detach else []
        return self._docker("exec", *flags, self.name, "bash", "-lc", command, check=check)

    def read(self, case: str, rel_path: str) -> str:
        """Text of a file in the case, or "" if it does not exist."""
        r = self.sh(case, f"cat {rel_path}", check=False)
        return r.stdout if r.returncode == 0 else ""

    def exists(self, case: str, rel_path: str) -> bool:
        """True/False if the path is there/absent; RunnerError if docker could not answer."""
        return self._probe(f"test -e /cases/{case}/{rel_path}", f"{case}/{rel_path}")

    def solver_alive(self) -> bool:
        """Is any simpleFoam process running in this container?"""
        return self._probe("pgrep -x simpleFoam > /dev/null", "simpleFoam")

    def active(self) -> str:
        """Key of the case that owns the solver lock ("" if none)."""
        r = self._raw("cat /cases/.active 2>/dev/null; exit 0")
        if r.returncode != 0:
            raise RunnerError(f"docker exec failed (exit {r.returncode}) while reading the active-run lock")
        return r.stdout.strip()

    def set_active(self, case: str) -> None:
        self._raw(f"echo {case} > /cases/.active", check=True)

    def clear_active(self) -> None:
        self._raw("rm -f /cases/.active", check=True)

    def read_forces(self, case: str) -> list[str]:
        """Every postProcessing/forces/<time>/force.dat, oldest segment first (restarts add dirs)."""
        script = (
            f"for d in $(ls postProcessing/forces 2>/dev/null | sort -n); do "
            f"echo {FORCE_SEP}; cat postProcessing/forces/$d/force.dat; done; exit 0"
        )
        r = self.sh(case, script, check=False)
        if r.returncode != 0:  # the script itself always exits 0, so this is a docker failure
            raise RunnerError(f"docker exec failed (exit {r.returncode}) while reading the force history")
        return [t for t in r.stdout.split(FORCE_SEP) if t.strip()]

    def copy_out(self, case: str, rel_path: str, local_dir: Path, check: bool = True) -> None:
        """Copy /cases/<case>/<rel_path> into local_dir (it lands as local_dir/<basename>)."""
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        self._docker("cp", f"{self.name}:/cases/{case}/{rel_path}", f"{local_dir}/", check=check)

    def remove_case(self, case: str) -> None:
        self._docker("exec", self.name, "rm", "-rf", f"/cases/{case}")


def _say(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class RunContext:
    settings: FoamSettings
    objects: dict[str, ObjectCfd]
    models: dict[str, trimesh.Trimesh]  # model0 per object (prepare_model output)
    root: Path  # runs/full, runs/preview or runs/test
    cap: int
    container: FoamContainer
    poll_s: float = 20.0
    on_update: Callable[[RunRecord], None] | None = None  # called whenever a record changes
    say: Callable[[str], None] = field(default=_say)


def stop_flag(root: Path) -> Path:
    """Creating this file asks the running batch to save and stop at the next poll."""
    return Path(root) / "STOP"


def _paths(rec: RunRecord, ctx: RunContext) -> tuple[Path, Path]:
    run_dir = Path(ctx.root) / rec.obj_id / rec.run_id
    return run_dir / "case", run_dir / "result"


def case_key(rec: RunRecord, ctx: RunContext) -> str:
    """Where the case lives in the volume: /cases/<mode>/<run_id>, mode = the root folder's name
    (test / preview / full), so a test run can never wipe or hijack a full run of the same id."""
    return f"{Path(ctx.root).name}/{rec.run_id}"


def _notify(rec: RunRecord, ctx: RunContext) -> None:
    if ctx.on_update:
        ctx.on_update(rec)


def _fail(rec: RunRecord, ctx: RunContext, reason: str) -> RunRecord:
    """Mark failed and keep only the logs (the case stays in the volume for inspection)."""
    _, result = _paths(rec, ctx)
    rec.status, rec.log, rec.converged = "failed", reason, False
    for log in LOGS:
        ctx.container.copy_out(case_key(rec, ctx), log, result, check=False)
    ctx.say(f"[{rec.run_id}] FAILED: {reason}")
    _notify(rec, ctx)
    return rec


def _write_and_mesh(rec: RunRecord, ctx: RunContext) -> bool:
    """Fresh start: wipe old copies, write + copy the case, mesh it. False (and failed) if the mesh is bad."""
    c, key = ctx.container, case_key(rec, ctx)
    case_dir, _ = _paths(rec, ctx)
    shutil.rmtree(case_dir, ignore_errors=True)  # no stale local files ...
    c.remove_case(key)  # ... and no stale processor*/polyMesh/time dirs in the volume
    cfd = ctx.objects[rec.obj_id]
    model = run_geometry(ctx.models[rec.obj_id], cfd, rec.direction_deg)
    spec = RunSpec(rec.run_id, rec.obj_id, rec.direction_index, rec.direction_deg, ctx.cap)
    write_case(case_dir, model, cfd, ctx.settings, spec)
    c.copy_in(case_dir, key)
    ctx.say(f"[{rec.run_id}] meshing")
    c.sh(key, MESH_SCRIPT, check=False)
    if "Mesh OK." not in c.read(key, "log.checkMesh"):
        _fail(rec, ctx, "checkMesh did not report 'Mesh OK.' (see log.checkMesh)")
        return False
    return True


def mesh_only(rec: RunRecord, ctx: RunContext) -> RunRecord:
    """Write, copy in and mesh a case, then stop. The case stays in the volume to inspect."""
    if _write_and_mesh(rec, ctx):
        cells = ""
        for line in ctx.container.read(case_key(rec, ctx), "log.checkMesh").splitlines():
            if line.strip().startswith("cells:"):
                cells = line.strip()
        rec.log = f"Mesh OK. {cells}".strip()
        ctx.say(f"[{rec.run_id}] {rec.log}")
    return rec


def _decompose_and_init(rec: RunRecord, ctx: RunContext) -> bool:
    """Split into N processor meshes and start from a potential-flow field (a good first guess).

    `-writep` is deliberately not used: in v2512 it needs a div(div(phi,U)) scheme that our
    fvSchemes does not define, and potentialFoam aborts with a FOAM FATAL IO ERROR.
    """
    n = ctx.settings.n_procs
    r = ctx.container.sh(
        case_key(rec, ctx),
        "decomposePar -force > log.decomposePar 2>&1 "
        f"&& mpirun -np {n} potentialFoam -parallel -initialiseUBCs > log.potentialFoam 2>&1",
        check=False,
    )
    if r.returncode != 0:
        _fail(rec, ctx, "decomposePar/potentialFoam failed (see log.decomposePar, log.potentialFoam)")
        return False
    return True


def _poll(rec: RunRecord, ctx: RunContext) -> bool:
    """Watch the solver until it ends. Returns True if we saw convergence.

    Each poll: read the force history; if it has converged, diverged (NaN/inf) or the stop flag
    appeared, tell the solver to write the current time and stop (`stopAt writeNow` in the
    case-root controlDict; runTimeModifiable picks it up at the next iteration). A request only
    counts as sent when the command succeeded; otherwise it is retried on the next poll.
    Ends when simpleFoam is gone and has left its exit marker.
    """
    c, s, key = ctx.container, ctx.settings, case_key(rec, ctx)
    converged_seen = stop_sent = False
    dead = 0
    while True:
        time.sleep(ctx.poll_s)
        forces = load_forces(c.read_forces(key))
        if len(forces) and int(forces[-1, 0]) != rec.iterations:
            rec.iterations = int(forces[-1, 0])
            _notify(rec, ctx)
        if not stop_sent:
            diverged = is_diverged(forces)  # a NaN that did not trip trapFpe must not burn on to the cap
            converged = is_converged(forces, s.convergence_window, s.convergence_tolerance, s.min_iterations)
            if diverged or converged or stop_flag(ctx.root).exists():
                r = c.sh(key, "foamDictionary -entry stopAt -set writeNow system/controlDict", check=False)
                if r.returncode == 0:
                    stop_sent = True
                    converged_seen = converged_seen or converged
                    reason = "diverged" if diverged else "converged" if converged else "stop requested"
                    ctx.say(f"[{rec.run_id}] {reason} at iteration {rec.iterations}; saving")
        if c.solver_alive():
            dead = 0
            continue
        if c.exists(key, EXIT_MARKER):
            return converged_seen
        dead += 1
        if dead >= DEAD_POLLS:
            raise RunnerError(f"{rec.run_id}: simpleFoam is gone without finishing; run again to resume")


def _export(rec: RunRecord, ctx: RunContext, fields: str) -> RunRecord:
    """Reconstruct, write VTK, copy VTK + forces out with run.json, then free the volume.

    If the export itself fails the solved case is NOT marked failed: the run stays `running`
    and a RunnerError is raised, so running again finds the finished solver and re-exports.
    """
    c, key = ctx.container, case_key(rec, ctx)
    _, result = _paths(rec, ctx)
    ctx.say(f"[{rec.run_id}] exporting VTK")
    r = c.sh(key,
             "reconstructPar -latestTime > log.reconstructPar 2>&1 "
             f"&& foamToVTK -latestTime -fields '{VTK_FIELDS}' > log.foamToVTK 2>&1", check=False)
    if r.returncode != 0:
        for log in ("log.reconstructPar", "log.foamToVTK"):
            c.copy_out(key, log, result, check=False)
        raise RunnerError(f"{rec.run_id}: reconstructPar/foamToVTK failed (logs in {result}); "
                          "the solved case is kept - run again to retry the export")
    shutil.rmtree(result, ignore_errors=True)
    result.mkdir(parents=True)
    c.copy_out(key, "VTK", result)
    c.copy_out(key, "postProcessing", result)
    rec.status = "done" if fields == "instant" else "flagged"
    (result / "run.json").write_text(json.dumps({
        "run_id": rec.run_id, "obj_id": rec.obj_id, "direction_index": rec.direction_index,
        "direction_deg": rec.direction_deg, "speed_m_s": ctx.objects[rec.obj_id].speed_m_s,
        "iterations": rec.iterations, "converged": rec.converged, "fields": fields, "status": rec.status,
    }, indent=2))
    ctx.say(f"[{rec.run_id}] {rec.status} after {rec.iterations} iterations (fields: {fields})")
    _notify(rec, ctx)  # record the final status BEFORE freeing the volume: a crash in between is harmless
    c.remove_case(key)
    return rec


def run_one(rec: RunRecord, ctx: RunContext) -> RunRecord:
    """Run (or resume) one run to done / flagged / stopped / failed. Mutates and returns `rec`.

    Raises RunnerError (leaving the run resumable and the volume untouched) if Docker cannot
    answer, if the solver vanished, if the export failed, or if a solver for ANOTHER case is
    already running in this container (one solver per container; never attach to someone else's).
    """
    if rec.status in ("done", "flagged", "failed"):
        return rec
    c, key = ctx.container, case_key(rec, ctx)

    # Guard first, before anything destructive: who owns a live solver in this container?
    alive = c.solver_alive()
    if alive:
        owner = c.active()
        if owner != key:
            raise RunnerError(f"{key}: simpleFoam is already running for {owner or 'an unknown case'} in "
                              f"container {c.name}; wait for it to finish (or stop it) first")
    resume = alive or (rec.status in ("running", "stopped") and c.exists(key, "processor0"))

    if not resume:
        if not _write_and_mesh(rec, ctx) or not _decompose_and_init(rec, ctx):
            return rec
    else:
        ctx.say(f"[{rec.run_id}] resuming from the last save (iteration {rec.iterations})")

    # Launch, unless our solver is already running (we crashed, it did not) or it finished unseen.
    finished_unseen = False
    if not alive:
        finished_unseen = rec.status == "running" and c.exists(key, EXIT_MARKER)
        if not finished_unseen:
            if stop_flag(ctx.root).exists():  # asked to stop while meshing: keep the prepared case
                rec.status = "stopped"
                _notify(rec, ctx)
                return rec
            c.sh(key, f"rm -f {EXIT_MARKER}; "
                 "foamDictionary -entry stopAt -set endTime system/controlDict")  # undo an earlier writeNow
            c.set_active(key)
            n = ctx.settings.n_procs
            c.sh(key, f"mpirun -np {n} simpleFoam -parallel > log.simpleFoam 2>&1; echo $? > {EXIT_MARKER}",
                 detach=True)
    rec.status = "running"
    _notify(rec, ctx)

    converged_seen = False if finished_unseen else _poll(rec, ctx)

    forces = load_forces(c.read_forces(key))
    rec.iterations = int(forces[-1, 0]) if len(forces) else 0
    fatal = c.sh(key, FATAL_SCRIPT, check=False).returncode == 0
    try:
        exit_code = int(c.read(key, EXIT_MARKER).strip() or 0)
    except ValueError:
        exit_code = 1
    c.clear_active()  # the solver has ended and is classified: release the one-solver-per-container lock
    if fatal or exit_code != 0 or is_diverged(forces):
        return _fail(rec, ctx, "simpleFoam crashed or diverged (see log.simpleFoam)")

    s = ctx.settings
    rec.converged = converged_seen or is_converged(forces, s.convergence_window, s.convergence_tolerance, s.min_iterations)
    if rec.converged:
        return _export(rec, ctx, "instant")
    if rec.iterations >= ctx.cap:
        return _export(rec, ctx, "mean")  # hit the cap: not settled, so export the averaged fields
    rec.status = "stopped"
    ctx.say(f"[{rec.run_id}] stopped at iteration {rec.iterations}; run again to resume")
    _notify(rec, ctx)
    return rec


def run_batch(obj_ids: list[str], ctx: RunContext, machine: str,
              directions: list[int] | None = None) -> list[RunRecord]:
    """Work through this machine's queue until it is empty or a stop is requested.

    Returns the records of the runs handled by this call (in order). `directions` restricts which
    direction indices are queued (the preview uses [0, 4, 8]). Ctrl-C asks for a graceful stop.
    """
    root = Path(ctx.root)
    root.mkdir(parents=True, exist_ok=True)
    qpath = root / f"queue_{machine}.json"
    records = merge_queue(load_queue(qpath), build_queue(ctx.objects, obj_ids, directions))
    save_queue(qpath, records)
    flag = stop_flag(root)
    flag.unlink(missing_ok=True)  # a flag left over from last time must not stop this batch

    def save(_rec: RunRecord | None = None) -> None:
        save_queue(qpath, records)

    batch_ctx = dataclasses.replace(ctx, on_update=save)

    def on_sigint(signum, frame):  # Ctrl-C: finish the current save, then stop
        flag.touch()
        ctx.say("Stop requested: saving the current run, then stopping (it resumes next time).")

    try:
        previous = signal.signal(signal.SIGINT, on_sigint)
    except ValueError:  # not the main thread: no handler, the STOP file still works
        previous = None

    handled: list[RunRecord] = []
    try:
        ctx.container.start()
        while not flag.exists():
            rec = next_run([r for r in records if r.obj_id in obj_ids])
            if rec is None:
                break
            rec.machine = machine
            save()
            run_one(rec, batch_ctx)
            save()
            handled.append(rec)
            if rec.status == "stopped":
                break
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    return handled
