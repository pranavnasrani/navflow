"""
Master Launcher for the 4-Monitor Interactive Navier-Stokes CFD Exhibition.

Modes:
- Production Mode (--monitor-mode): Spans 4 fullscreen windows across 4 physical monitors.
- Development Mode (--dev-mode, default): Opens 4 synchronized windows arranged on the current desktop.
- Single Object Mode (--object <id>): Runs a single window for focused testing.

The windows show OpenFOAM results from assets/cache/<object>_states.npz (written by
solver/import_results.py). There is no live solving at run time.
"""

import argparse
import sys
import zipfile
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from app.cache_format import read_cache
from app.input_controller import ExhibitionInputController
from app.monitor_window import MonitorWindow
from solver.foam.settings import load_settings

CACHE_HELP = (
    "Run solver/import_results.py (or git checkout v1-fallback for the old exhibit)"
)


def _check_caches(cache_dir: Path, obj_ids: list[str]) -> None:
    """Exit 1 with a clear message if any needed cache is missing or is not a v2 cache."""
    for obj_id in obj_ids:
        path = cache_dir / f"{obj_id}_states.npz"
        problem = None
        if not path.exists():
            problem = f"{path} not found."
        else:
            try:
                read_cache(path)
            except (ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile) as exc:  # v1 caches raise ValueError
                problem = str(exc)
        if problem:
            print(f"\n[!] {problem}\n[!] {CACHE_HELP}", file=sys.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Navier-Stokes Interactive 4-Monitor CFD Exhibition App")
    parser.add_argument("--monitor-mode", action="store_true", help="Assign each of the 4 windows to a separate physical monitor")
    parser.add_argument("--dev-mode", action="store_true", help="Arrange windows on the primary desktop for development")
    parser.add_argument("--object", type=str, default=None, help="Run only one specific object (airplane, f1_car, building, human)")
    parser.add_argument("--config", type=str, default="config/exhibition_config.json", help="Path to config file")
    parser.add_argument("--cache-dir", type=str, default="assets/cache", help="Folder holding <object>_states.npz")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found at {cfg_path}")
        sys.exit(1)

    settings, _, config = load_settings(cfg_path)

    objects_to_launch = config["objects"]
    if args.object:
        objects_to_launch = [obj for obj in objects_to_launch if obj["id"] == args.object]
        if not objects_to_launch:
            print(f"Error: Unknown object '{args.object}'. Choose from: {[o['id'] for o in config['objects']]}")
            sys.exit(1)

    cache_dir = Path(args.cache_dir)
    _check_caches(cache_dir, [o["id"] for o in objects_to_launch])

    app = QApplication(sys.argv)
    app.setApplicationName("Navier-Stokes STEM Exhibition")

    # Global input controller: 10 smoke-wand buttons, one direction knob, Tab for the view.
    input_ctrl = ExhibitionInputController(
        wand_count=settings.wand_count,
        knob_step=float(config["controls"]["knob_step"]),
    )

    screens = QApplication.screens()
    print(f"Detected {len(screens)} physical display(s).")

    windows = []
    for i, obj_cfg in enumerate(objects_to_launch):
        win = MonitorWindow(
            obj_id=obj_cfg["id"],
            input_controller=input_ctrl,
            cache_dir=str(cache_dir),
        )
        windows.append(win)

        if args.monitor_mode and len(screens) >= len(objects_to_launch):
            # Move to corresponding physical monitor and go fullscreen
            target_screen = screens[i]
            geom = target_screen.geometry()
            win.setGeometry(geom)
            win.showFullScreen()
            print(f"Placed {obj_cfg['display_name']} on Screen {i+1} ({geom.width()}x{geom.height()})")
        else:
            # Development Mode: Arrange windows nicely on desktop
            primary_geom = screens[0].availableGeometry()
            pw = primary_geom.width()
            ph = primary_geom.height()

            if len(objects_to_launch) == 1:
                win.setGeometry(primary_geom.x() + 50, primary_geom.y() + 50, pw - 100, ph - 100)
            elif len(objects_to_launch) == 4:
                # 2x2 grid
                half_w = pw // 2
                half_h = ph // 2
                positions = [
                    (0, 0),             # Top-Left: Airplane
                    (half_w, 0),        # Top-Right: F1 Car
                    (0, half_h),        # Bottom-Left: Building
                    (half_w, half_h)    # Bottom-Right: Human
                ]
                x_off, y_off = positions[i]
                win.setGeometry(primary_geom.x() + x_off, primary_geom.y() + y_off, half_w, half_h)

            win.show()

    print("\n========================================================")
    print("NAVIER-STOKES EXHIBITION APPLICATION RUNNING")
    print("Controls:")
    print("  - Keys '1' to '0' : Move the smoke wand to height 1 to 10")
    print("  - Keys Up / Down  : Step the smoke wand up/down")
    print("  - Keys Right / Left or '+' / '-' : Turn the wind-direction knob")
    print("  - Mouse Wheel     : Smoothly turn the knob")
    print("  - Tab             : Change view (pressure / dominant N-S term / turbulence k)")
    print("========================================================\n")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
