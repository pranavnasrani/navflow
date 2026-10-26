"""Command-line front end for the OpenFOAM batch runner.

    ./.venv/bin/python solver/run_openfoam.py run --objects airplane f1_car [--preview] [--machine laptop]
    ./.venv/bin/python solver/run_openfoam.py stop [--preview]
    ./.venv/bin/python solver/run_openfoam.py status [--preview]
    ./.venv/bin/python solver/run_openfoam.py test --object airplane --direction-index 0 --iterations 60 \
        [--mesh-only] [--coarse]

`run` works through this machine's queue (runs/full or runs/preview, file queue_<machine>.json)
and can be interrupted with Ctrl-C or `stop`; running the same command again resumes where it
stopped. Preview = wind directions 0, 4 and 8 only, cap 500 iterations. `test` runs one case in
runs/test, mainly to check meshing and the pipeline before committing to hours of compute.
All commands need Docker Desktop to be running.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.foam.geometry import prepare_model
from solver.foam.queue import build_queue, load_queue
from solver.foam.runner import (Executor, FoamContainer, RunContext, RunnerError, SubprocessExecutor,
                                mesh_only, run_batch, run_one, stop_flag)
from solver.foam.settings import load_settings

DOCKER_DOWN = "Docker Desktop isn't running — open it and try again."
PREVIEW_DIRECTIONS = [0, 4, 8]
COARSE_CELLS = 4  # --coarse: a quick, rough mesh for pipeline checks
COARSE_REFINEMENT = (2, 3)


def docker_running(ex: Executor) -> bool:
    try:
        return ex.run(["docker", "info"], check=False).returncode == 0
    except OSError:  # the docker program itself is missing
        return False


def _runs_root(preview: bool) -> Path:
    return PROJECT_ROOT / "runs" / ("preview" if preview else "full")


def _build_context(args, ex: Executor, root: Path, cap: int, obj_ids: list[str], coarse: bool = False,
                   poll_s: float = 20.0) -> RunContext:
    settings, objects, _ = load_settings(PROJECT_ROOT / "config" / "exhibition_config.json")
    if coarse:  # a per-object refinement would override the coarse setting, so drop it too
        settings = replace(settings, base_cells_per_size=COARSE_CELLS, surface_refinement=COARSE_REFINEMENT)
        objects = {k: replace(v, surface_refinement=None) for k, v in objects.items()}
    unknown = [o for o in obj_ids if o not in objects]
    if unknown:
        sys.exit(f"Unknown object(s) {unknown}; choose from {sorted(objects)}")
    models = {o: prepare_model(PROJECT_ROOT / "assets" / "models" / o / "model.stl", objects[o]) for o in obj_ids}
    container = FoamContainer(settings, f"navflow-{args.machine}", ex)
    return RunContext(settings, objects, models, root, cap, container, poll_s=poll_s)


def cmd_run(args, ex: Executor) -> int:
    settings, _, _ = load_settings(PROJECT_ROOT / "config" / "exhibition_config.json")
    cap = settings.preview_iterations if args.preview else settings.max_iterations
    ctx = _build_context(args, ex, _runs_root(args.preview), cap, args.objects)
    done = run_batch(args.objects, ctx, args.machine, PREVIEW_DIRECTIONS if args.preview else None)
    for r in done:
        print(f"{r.run_id}: {r.status} ({r.iterations} iterations)")
    return 0


def cmd_stop(args, ex: Executor) -> int:
    flag = stop_flag(_runs_root(args.preview))
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.touch()
    print(f"Stop requested ({flag}). The running batch saves its current run and stops at its next check.")
    return 0


def cmd_status(args, ex: Executor) -> int:
    root = _runs_root(args.preview)
    files = sorted(root.glob("queue_*.json"))
    if not files:
        print(f"No queue found in {root}")
        return 0
    print(f"{'run':<16}{'status':<10}{'iterations':>11}  {'converged':<10}{'machine'}")
    for f in files:
        for r in load_queue(f):
            print(f"{r.run_id:<16}{r.status:<10}{r.iterations:>11}  {str(r.converged):<10}{r.machine}")
    return 0


def cmd_test(args, ex: Executor) -> int:
    root = PROJECT_ROOT / "runs" / "test"
    ctx = _build_context(args, ex, root, args.iterations, [args.object], coarse=args.coarse, poll_s=args.poll_s)
    root.mkdir(parents=True, exist_ok=True)
    stop_flag(root).unlink(missing_ok=True)
    rec = build_queue(ctx.objects, [args.object], [args.direction_index])[0]
    ctx.container.start()
    rec = mesh_only(rec, ctx) if args.mesh_only else run_one(rec, ctx)
    print(f"{rec.run_id}: {rec.status} {rec.log}".rstrip())
    return 1 if rec.status == "failed" else 0


def make_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--machine", default="laptop", help="queue/container name suffix (default laptop)")
    p = argparse.ArgumentParser(description="Run the OpenFOAM wind-tunnel batch in Docker.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="run or resume this machine's queue")
    run.add_argument("--objects", nargs="+", required=True)
    run.add_argument("--preview", action="store_true", help="directions 0, 4, 8; cap 500; runs/preview")
    run.set_defaults(func=cmd_run)

    stop = sub.add_parser("stop", parents=[common], help="ask the running batch to save and stop")
    stop.add_argument("--preview", action="store_true")
    stop.set_defaults(func=cmd_stop)

    status = sub.add_parser("status", parents=[common], help="one row per run")
    status.add_argument("--preview", action="store_true")
    status.set_defaults(func=cmd_status)

    test = sub.add_parser("test", parents=[common], help="run a single case in runs/test")
    test.add_argument("--object", required=True)
    test.add_argument("--direction-index", type=int, default=0)
    test.add_argument("--iterations", type=int, default=60)
    test.add_argument("--mesh-only", action="store_true", help="mesh and checkMesh only")
    test.add_argument("--coarse", action="store_true", help="coarse mesh: 4 cells per size, refinement (2, 3)")
    test.add_argument("--poll-s", type=float, default=20.0, help="seconds between solver checks")
    test.set_defaults(func=cmd_test)
    return p


def main(argv: list[str] | None = None, ex: Executor | None = None) -> int:
    args = make_parser().parse_args(argv)
    ex = ex or SubprocessExecutor()
    if not docker_running(ex):
        print(DOCKER_DOWN, file=sys.stderr)
        return 2
    try:
        return args.func(args, ex)
    except RunnerError as e:  # the run is left resumable; say why and exit non-zero
        print(f"Stopped: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
