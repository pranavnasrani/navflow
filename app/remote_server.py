"""
iPad remote control over local HTTP.

The display stays a native PySide6 + PyVista app; this module only lets a tablet on the same
Wi-Fi / phone hotspot drive the same three inputs as the macropad:

    GET  /       the single-file touch page (app/remote/index.html, no external assets)
    GET  /manifest.webmanifest, /icon-180.png, /icon-192.png, /icon-512.png   install-as-app files (PWA)
    GET  /state  {"wand", "knob", "view", "wand_count"}   (wand is 0-based, like the controller)
    GET  /sync   the controller's full snapshot (adds detailed, show_active, show_step, reset_count);
                 a second Mac started with --follow polls this to mirror the host
    POST /cmd    {"kind": "wand"|"knob"|"view"|"detail"|"reset"|"touch"|"theme", "value": number}  -> 204, or 400
                 (detail: 1 = detailed strip, 0 = simple; theme: 1 = dark, 0 = light; reset and touch ignore the value)

Thread safety: HTTP requests are served on worker threads, but Qt objects belong to the GUI
thread. So a handler never touches the controller's methods; it emits `RemoteServer.command`,
which is connected with Qt.QueuedConnection to a slot that runs on the GUI thread. (GET /state
only reads plain int/float attributes, so a benign race is harmless.)

Robustness: bodies over 1 KB are refused, every handler error becomes a 4xx/5xx reply, slow
clients time out, and a failing controller call is swallowed in the slot, so no visitor input
can crash or freeze the exhibit.

Maths link (IGCSE 0606): the page turns the dial angle into the knob with a straight line,
    knob = (angle + 135) / 270,
the same linear interpolation the state manager uses to map knob to wind direction.
"""

import ipaddress
import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal

REMOTE_DIR = Path(__file__).resolve().parent / "remote"
INDEX_PATH = REMOTE_DIR / "index.html"
# What makes the page installable on an iPad's Home Screen: a manifest and icons, served next to the page.
ASSETS = {
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/icon-180.png": ("icon-180.png", "image/png"),   # <link rel="apple-touch-icon">
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
}
MAX_BODY_BYTES = 1024
COMMAND_KINDS = ("wand", "knob", "view", "detail", "reset", "touch", "theme")


def parse_command(body: bytes) -> tuple[str, float]:
    """
    Decode a POST /cmd body into (kind, value).
    Raises ValueError for bad JSON, a non-object, an unknown kind, or a missing / non-numeric /
    non-finite value (NaN, +-inf, numbers too big for a float, strings and booleans).
    """
    try:
        data = json.loads(body)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise ValueError(f"invalid JSON: {exc.__class__.__name__}") from None
    if not isinstance(data, dict):
        raise ValueError("command must be a JSON object")
    kind = data.get("kind")
    if not isinstance(kind, str) or kind not in COMMAND_KINDS:
        raise ValueError(f"unknown kind {kind!r}; expected one of {COMMAND_KINDS}")
    raw = data.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError("value must be a number")
    try:
        value = float(raw)
    except OverflowError:
        raise ValueError("value is too large") from None
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return kind, value


def _is_private_lan(ip: str) -> bool:
    """True for a usable private IPv4 (10.x, 172.16-31.x, 192.168.x), never loopback or link-local."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return addr.version == 4 and addr.is_private and not addr.is_loopback and not addr.is_link_local


def _probe_route(target: str) -> str | None:
    """Address of the interface the OS would use to reach `target` (UDP connect sends no packet)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((target, 9))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return ip if ip and not ip.startswith("0.") else None


def _hostname_addresses() -> list[str]:
    """IPv4 addresses the machine's own hostname resolves to (works with no default route)."""
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return [info[4][0] for info in infos]


def lan_ip() -> str:
    """
    This machine's LAN address, for the URL typed on the iPad.

    1. Default-route probes: a UDP "connect" to a public address makes the OS pick the outgoing
       interface, whose address we read back (8.8.8.8, then 10.255.255.255).
    2. No default route (an isolated router or phone hotspot): probe the common private gateways
       192.168.255.255 and 172.20.10.1 (iPhone hotspots use 172.20.10.x), then the addresses the
       hostname resolves to; the first private IPv4 that is not 127.x wins.
    3. Otherwise 127.0.0.1.
    """
    for probe in ("8.8.8.8", "10.255.255.255"):
        ip = _probe_route(probe)
        if ip:
            return ip
    for probe in ("192.168.255.255", "172.20.10.1"):
        ip = _probe_route(probe)
        if ip and _is_private_lan(ip):
            return ip
    for ip in _hostname_addresses():
        if _is_private_lan(ip):
            return ip
    return "127.0.0.1"


class _Handler(BaseHTTPRequestHandler):
    """One short-lived handler per request. `self.server.remote` is the owning RemoteServer."""

    protocol_version = "HTTP/1.1"
    timeout = 5  # seconds: a stalled client must not hold a worker thread forever

    def log_message(self, format, *args):  # keep the exhibit terminal quiet
        pass

    # -- helpers --
    def _reply(self, status: int, body: bytes = b"", content_type: str = "text/plain; charset=utf-8"):
        self.send_response(status)
        if status != 204:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _safely(self, work):
        """Run a route; any unexpected error becomes a 500 instead of an unhandled exception."""
        try:
            work()
        except (BrokenPipeError, ConnectionError, TimeoutError):
            self.close_connection = True  # the client went away; nothing to answer
        except Exception:
            self.close_connection = True
            try:
                self._reply(500, b"internal error")
            except Exception:
                pass

    # -- routes --
    def do_GET(self):
        self._safely(self._get)

    def do_POST(self):
        self._safely(self._post)

    def _get(self):
        path = self.path.split("?", 1)[0]
        remote = self.server.remote
        if path == "/":
            self._reply(200, remote.index_html, "text/html; charset=utf-8")
        elif path == "/state":
            c = remote.controller
            state = {
                "wand": int(c.current_wand_index),
                "knob": float(c.current_knob),
                "view": int(c.current_view),
                "dark": c.theme == "dark",
                "wand_count": int(c.wand_count),
            }
            self._reply(200, json.dumps(state).encode(), "application/json")
        elif path in ASSETS:
            self._reply(200, remote.assets[path], ASSETS[path][1])
        elif path == "/sync":
            self._reply(200, json.dumps(remote.controller.snapshot()).encode(), "application/json")
        else:
            self._reply(404, b"not found")

    def _post(self):
        if self.path.split("?", 1)[0] != "/cmd":
            self.close_connection = True  # the unread body would otherwise corrupt keep-alive
            self._reply(404, b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY_BYTES:
            self.close_connection = True  # we refuse to read the body, so drop the connection
            self._reply(400, b"bad or oversized body")
            return
        try:
            kind, value = parse_command(self.rfile.read(length))
        except ValueError as exc:
            self._reply(400, str(exc).encode())
            return
        self.server.remote.command.emit(kind, value)  # hand off to the GUI thread
        self._reply(204)


class RemoteServer(QObject):
    """Serves the iPad page and turns its commands into GUI-thread controller calls."""

    command = Signal(str, float)

    def __init__(self, controller, port: int = 8765):
        super().__init__()
        self.controller = controller
        self.port = port
        self.index_html = b""
        self.assets: dict[str, bytes] = {}
        self._httpd = None
        self._thread = None
        # QueuedConnection: the emit happens on an HTTP worker thread, the slot runs on this
        # object's (GUI) thread.
        self.command.connect(self._apply, Qt.QueuedConnection)

    def _apply(self, kind: str, value: float):
        """GUI-thread slot. The controller clamps the values (wand and knob) or wraps them (view)."""
        try:
            if kind == "wand":
                self.controller.set_wand(int(value))
            elif kind == "knob":
                self.controller.set_knob(value)
            elif kind == "view":
                self.controller.set_view(int(value))
            elif kind == "detail":
                self.controller.set_detail(value >= 0.5)
            elif kind == "reset":
                self.controller.request_camera_reset()
            elif kind == "touch":
                self.controller.touch()
            elif kind == "theme":
                self.controller.set_theme("dark" if value >= 0.5 else "light")
        except Exception as exc:  # never let a remote command take the exhibit down
            print(f"[remote] ignored {kind} command: {exc}")

    def start(self) -> str:
        """Bind 0.0.0.0:<port> and serve in a daemon thread. Returns the URL to open on the iPad."""
        if self._httpd is not None:
            return self._url()
        self.index_html = INDEX_PATH.read_bytes()
        self.assets = {route: (REMOTE_DIR / name).read_bytes() for route, (name, _) in ASSETS.items()}
        try:
            httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _Handler)
        except OSError as exc:
            raise OSError(
                f"iPad remote could not listen on port {self.port} ({exc.strerror or exc}); "
                f"is another copy of the exhibit already running?"
            ) from exc
        httpd.daemon_threads = True
        httpd.remote = self
        self._httpd = httpd
        self._thread = threading.Thread(
            target=lambda: httpd.serve_forever(poll_interval=0.05), name="remote-http", daemon=True
        )
        self._thread.start()
        return self._url()

    def _url(self) -> str:
        return f"http://{lan_ip()}:{self._httpd.server_address[1]}/"

    def stop(self):
        """Stop serving and join the thread. Safe to call twice or before start()."""
        httpd, thread = self._httpd, self._thread
        self._httpd = self._thread = None
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)
