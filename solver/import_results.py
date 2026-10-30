"""Import finished OpenFOAM runs into the display caches.

    ./.venv/bin/python solver/import_results.py --root runs/full [--objects airplane f1_car] [--out-dir assets/cache]

For every object this reads runs/<mode>/<object>/<run_id>/result/, builds the model-frame surface
pressure, Navier-Stokes terms, smoke-wand streamlines and sustainability metrics, and writes
<out-dir>/<object>_states.npz (format version 2). Directions that have not been run yet are
filled from the nearest finished direction; the report says which. Safe to re-run as more runs finish. If one object fails, the others are still imported and the exit
code is 1.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solver.foam.geometry import load_prepared
from solver.foam.importer import import_object
from solver.foam.settings import load_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "runs" / "full", help="runs folder (default runs/full)")
    ap.add_argument("--objects", nargs="+", help="object ids (default: all in the config)")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "assets" / "cache", help="default assets/cache")
    args = ap.parse_args(argv)

    settings, objects, raw = load_settings(PROJECT_ROOT / "config" / "exhibition_config.json")
    configs = {o["id"]: o for o in raw["objects"]}
    obj_ids = args.objects or list(objects)
    unknown = [o for o in obj_ids if o not in objects]
    if unknown:
        sys.exit(f"Unknown object(s) {unknown}; choose from {sorted(objects)}")

    failed = 0
    for obj_id in obj_ids:
        start = time.perf_counter()
        try:
            model0 = load_prepared(PROJECT_ROOT / "assets" / "models" / obj_id / "model.stl", objects[obj_id])
            report = import_object(obj_id, settings, objects[obj_id], model0, configs[obj_id], args.root,
                                   args.out_dir / f"{obj_id}_states.npz")
        except Exception as exc:  # no finished runs, a bad model, a missing file, a corrupt VTK: report it, do the rest
            print(f"{obj_id}: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue
        filled = ", ".join(f"{i}<-{j}" for i, j in report["filled"].items()) or "none"
        print(f"{obj_id}: imported {report['imported']}, filled {len(report['filled'])} ({filled}), "
              f"flagged {report['flagged']}  [{time.perf_counter() - start:.1f} s]")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
