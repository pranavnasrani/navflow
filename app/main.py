"""
Master Launcher for the 4-Monitor Interactive Navier-Stokes CFD Exhibition.

Modes:
- Production Mode (--monitor-mode): Spans 4 fullscreen windows across 4 physical monitors.
- Development Mode (--dev-mode, default): Opens 4 synchronized windows arranged on the current desktop.
- Single Object Mode (--object <id>): Runs a single window for focused testing.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.input_controller import ExhibitionInputController
from app.monitor_window import MonitorWindow


def main():
    parser = argparse.ArgumentParser(description="Navier-Stokes Interactive 4-Monitor CFD Exhibition App")
    parser.add_argument("--monitor-mode", action="store_true", help="Assign each of the 4 windows to a separate physical monitor")
    parser.add_argument("--dev-mode", action="store_true", help="Arrange windows on the primary desktop for development")
    parser.add_argument("--object", type=str, default=None, help="Run only one specific object (airplane, f1_car, building, human)")
    parser.add_argument("--config", type=str, default="config/exhibition_config.json", help="Path to config file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found at {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r") as f:
        config = json.load(f)

    # Verify that cache files exist
    cache_dir = Path("assets/cache")
    if not cache_dir.exists() or len(list(cache_dir.glob("*.npz"))) == 0:
        print("\n[!] Pre-computed CFD cache files not found in assets/cache/.")
        print("[!] Generating starter models and pre-computed CFD states now...")
        from assets.generate_starter_models import generate_all_models
        from solver.batch_precompute import run_batch_precompute

        generate_all_models()
        run_batch_precompute(quick=True)

    app = QApplication(sys.argv)
    app.setApplicationName("Navier-Stokes STEM Exhibition")

    # Global input controller (listens for 10 Z-step buttons + velocity rotary knob)
    input_ctrl = ExhibitionInputController(
        total_z_steps=len(config["wind_tunnel"]["z_steps"]),
        min_speed=config["wind_tunnel"]["wind_speeds"][0],
        max_speed=config["wind_tunnel"]["wind_speeds"][-1],
        speed_step=2.5,
        controls_cfg=config.get("controls", {})
    )

    screens = QApplication.screens()
    print(f"Detected {len(screens)} physical display(s).")

    windows = []
    objects_to_launch = config["objects"]

    if args.object:
        objects_to_launch = [obj for obj in objects_to_launch if obj["id"] == args.object]
        if not objects_to_launch:
            print(f"Error: Unknown object '{args.object}'. Choose from: {[o['id'] for o in config['objects']]}")
            sys.exit(1)

    for i, obj_cfg in enumerate(objects_to_launch):
        win = MonitorWindow(
            obj_id=obj_cfg["id"],
            input_controller=input_ctrl,
            cache_dir=str(cache_dir)
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
    print("  - Keys '1' to '0' : Jump to Z-height Step 1 to 10")
    print("  - Keys Up / Down  : Step Z-height up/down")
    print("  - Keys Right / Left or '+' / '-' : Adjust Wind Velocity")
    print("  - Mouse Wheel     : Smoothly turn Velocity Knob")
    print("========================================================\n")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
