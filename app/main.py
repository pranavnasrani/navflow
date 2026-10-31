"""
Master Launcher for the Multi-Monitor Interactive Navier-Stokes CFD Exhibition.

Modes:
- Production Mode (--monitor-mode): one fullscreen window per physical monitor.
- Development Mode (--dev-mode, default): the windows tiled on the current desktop.

Which windows this computer opens: --objects airplane f1_car building (default: every object the exhibit shows;
--object <id> is the same with one name). Which monitor each one goes to in --monitor-mode: --screens 1 2 (1 = first monitor).

Two computers: the HOST (no --follow) opens its windows and also owns the macropad, the iPad page and the idle
show. A second computer runs `--follow <host-ip>` with the other objects and mirrors the host; its own keys go
to the host. The host prints the exact command to type on the other computer when it starts.

Look: --theme light|dark picks the starting theme (T on the keyboard or the moon button on the iPad switches every
screen); --orbit SECONDS sets how long the camera takes to circle each model (default 60, 0 holds it still).

The iPad remote (a small HTTP page on controls.remote_port) starts automatically on the host; --no-remote
skips it (and so the link to a second computer too, which uses the same server). --ngrok [DOMAIN] also puts it
on the internet through ngrok (app/tunnel.py), so an iPad on any network can open it and add it to its Home Screen.

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
from app.follower import FollowerLink, parse_follow
from app.input_controller import ExhibitionInputController
from app.monitor_window import MonitorWindow
from app.remote_server import RemoteServer, lan_ip
from app.tunnel import NgrokTunnel
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


def tile(count: int, index: int, area: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Window rectangle (x, y, w, h) for window `index` of `count`, tiled inside `area` = (x, y, w, h):
    one window fills the area with a margin, 2 sit side by side, 3 or 4 make a 2 x 2 grid (the last cell
    stays empty for 3). Row-major, so the order is top-left, top-right, bottom-left, bottom-right.
    """
    x, y, w, h = area
    if count <= 1:
        return x + 50, y + 50, w - 100, h - 100
    cols = 1 if count > 4 else 2  # more than 4 windows is not an exhibit layout; stack them
    rows = -(-count // cols)
    cell_w, cell_h = w // cols, h // rows
    return x + (index % cols) * cell_w, y + (index // cols) * cell_h, cell_w, cell_h


def main():
    parser = argparse.ArgumentParser(description="Navier-Stokes Interactive Multi-Monitor CFD Exhibition App")
    parser.add_argument("--monitor-mode", action="store_true", help="One fullscreen window per physical monitor")
    parser.add_argument("--dev-mode", action="store_true", help="Arrange windows on the primary desktop for development")
    parser.add_argument("--objects", nargs="+", default=None, metavar="ID",
                        help="Open windows only for these objects, in this order (airplane f1_car building; default: the exhibit's objects)")
    parser.add_argument("--object", type=str, default=None, help="Same as --objects with one name")
    parser.add_argument("--screens", nargs="+", type=int, default=None, metavar="N",
                        help="With --monitor-mode: the monitor (1 = first) for each object in --objects, in the same order")
    parser.add_argument("--theme", choices=("light", "dark"), default="light",
                        help="Colour theme to start in (the T key or the iPad remote switches it on every screen)")
    parser.add_argument("--orbit", type=float, default=None, metavar="SECONDS",
                        help="Seconds the camera takes to circle each model (default 60; 0 = hold still)")
    parser.add_argument("--ngrok", nargs="?", const="", default=None, metavar="DOMAIN",
                        help="Also put the iPad remote on the internet through ngrok; DOMAIN is your reserved ngrok domain, if any")
    parser.add_argument("--follow", type=str, default=None, metavar="HOST[:PORT]|https://URL",
                        help="Run as the second computer: mirror the host Mac at this address (or its https ngrok address, which works on any network) instead of owning the controls")
    parser.add_argument("--config", type=str, default="config/exhibition_config.json", help="Path to config file")
    parser.add_argument("--no-remote", action="store_true", help="Do not start the iPad remote web server")
    parser.add_argument("--cache-dir", type=str, default="assets/cache", help="Folder holding <object>_states.npz")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found at {cfg_path}")
        sys.exit(1)

    settings, _, config = load_settings(cfg_path)

    # An object with "display": false stays in the solver and the caches but is not part of the exhibit
    # (the human). Naming it with --objects still opens it, for a quick look.
    configured = config["objects"]
    all_objects = [o for o in configured if o.get("display", True)]
    wanted = list(args.objects or [])
    if args.object:
        wanted.append(args.object)
    objects_to_launch = all_objects
    if wanted:
        by_id = {o["id"]: o for o in configured}
        unknown = [w for w in wanted if w not in by_id]
        if unknown:
            print(f"Error: Unknown object(s) {unknown}. Choose from: {list(by_id)}")
            sys.exit(1)
        if len(set(wanted)) != len(wanted):
            print(f"Error: an object is listed twice in {wanted}.")
            sys.exit(1)
        objects_to_launch = [by_id[w] for w in wanted]

    if args.screens is not None:
        if len(args.screens) != len(objects_to_launch):
            print(f"Error: --screens gave {len(args.screens)} monitor(s) for {len(objects_to_launch)} window(s); give one each.")
            sys.exit(1)
        if min(args.screens) < 1 or len(set(args.screens)) != len(args.screens):
            print("Error: --screens numbers start at 1 and must all be different.")
            sys.exit(1)

    follow = None
    if args.follow:
        try:
            follow = parse_follow(args.follow, int(config["controls"]["remote_port"]))
        except ValueError as exc:
            print(f"Error: --follow {args.follow!r}: {exc}")
            sys.exit(1)

    cache_dir = Path(args.cache_dir)
    _check_caches(cache_dir, [o["id"] for o in objects_to_launch])

    app = QApplication(sys.argv)
    app.setApplicationName("Navier-Stokes STEM Exhibition")

    orbit_seconds = 60.0 if args.orbit is None else float(args.orbit)

    # Global input controller: 10 smoke-wand buttons, one direction knob, Tab for the view.
    input_ctrl = ExhibitionInputController(
        wand_count=settings.wand_count,
        knob_step=float(config["controls"]["knob_step"]),
        idle_seconds=float(config["controls"].get("idle_seconds", 25)),  # 0 turns the idle show off
    )
    input_ctrl.theme = args.theme  # before the windows exist, so they start in it

    # Host: a tiny HTTP server so a tablet on the same Wi-Fi / hotspot drives the same controller as the
    # macropad, and so a second computer can follow. The display itself stays this native app.
    remote_url = None
    follower = None
    if follow:
        # Follower: no controls of its own. Its controller mirrors the host and sends its own input there.
        follower = FollowerLink(input_ctrl, *follow)
        follower.start()
        app.aboutToQuit.connect(follower.stop)
        remote_url = follower.url  # the iPad page is served by the host
    elif not args.no_remote:
        try:
            remote = RemoteServer(input_ctrl, port=int(config["controls"]["remote_port"]))
            remote_url = remote.start()
            app.aboutToQuit.connect(remote.stop)
            if args.ngrok is not None:
                tunnel = NgrokTunnel(int(config["controls"]["remote_port"]), args.ngrok or None)
                if tunnel.start():
                    app.aboutToQuit.connect(tunnel.stop)
        except (OSError, OverflowError, ValueError, TypeError, KeyError) as exc:  # busy or invalid port: never block the launch
            print(f"\n[!] {exc}\n[!] Continuing without the iPad remote (use the keyboard or macropad).", file=sys.stderr)

    screens = QApplication.screens()
    print(f"Detected {len(screens)} physical display(s).")

    windows = []
    for i, obj_cfg in enumerate(objects_to_launch):
        win = MonitorWindow(
            obj_id=obj_cfg["id"],
            input_controller=input_ctrl,
            cache_dir=str(cache_dir),
            remote_url=remote_url,
            orbit_seconds=orbit_seconds,
        )
        windows.append(win)

        if args.monitor_mode and max(args.screens or [len(objects_to_launch)]) <= len(screens):
            # Move to the monitor chosen with --screens (default: the i-th monitor) and go fullscreen
            screen_no = args.screens[i] if args.screens else i + 1
            geom = screens[screen_no - 1].geometry()
            win.setGeometry(geom)
            win.showFullScreen()
            print(f"Placed {obj_cfg['display_name']} on Screen {screen_no} ({geom.width()}x{geom.height()})")
        else:
            if args.monitor_mode:
                print(f"[!] --monitor-mode needs monitor(s) {args.screens or list(range(1, len(objects_to_launch) + 1))} "
                      f"but only {len(screens)} found; tiling on the main screen instead.", file=sys.stderr)
            # Development Mode: tile the windows on the desktop
            avail = screens[0].availableGeometry()
            x, y, w, h = tile(len(objects_to_launch), i, (avail.x(), avail.y(), avail.width(), avail.height()))
            win.setGeometry(x, y, w, h)
            win.show()

    print("\n========================================================")
    print("NAVIER-STOKES EXHIBITION APPLICATION RUNNING")
    print("Controls:")
    print("  - Keys '1' to '0' : Move the smoke wand to height 1 to 10")
    print("  - Keys Up / Down  : Step the smoke wand up/down")
    print("  - Keys Right / Left or '+' / '-' : Turn the wind-direction knob")
    print("  - Mouse Wheel     : Smoothly turn the knob")
    print("  - Tab             : Change view (pressure / dominant N-S term / turbulence k)")
    if follower:
        print(f"  - Following       : {follower.url} (this Mac mirrors the host; its keys go to the host)")
    elif remote_url:
        print(f"  - iPad remote     : open {remote_url} in Safari on the same network")
    else:
        print("  - iPad remote     : off")
    others = [o["id"] for o in all_objects if o not in objects_to_launch]
    if remote_url and not follower and others:
        port = remote_url.rstrip("/").rsplit(":", 1)[-1]
        print(f"  - Other Mac       : ./run_exhibition.sh --monitor-mode --follow {lan_ip()}:{port} --objects {' '.join(others)}")
    print("========================================================\n")

    input_ctrl.start_clock()  # the idle show starts now and comes back whenever the exhibit is left alone

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
