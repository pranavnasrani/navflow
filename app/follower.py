"""
Second-Mac link: lets one exhibition run on two computers.

The HOST Mac owns the macropad, the iPad page and the idle show, and serves its controller over the same
small HTTP server as the iPad remote (app/remote_server.py). A FOLLOWER Mac (`app/main.py --follow HOST`)
shows other windows and mirrors the host:

    host -> follower   GET  /sync   polled about 25 times a second; the reply goes to controller.apply_sync()
    follower -> host   POST /cmd    the follower's own keys, wheel and drags (macropad on the follower still works)

Both directions run on worker threads; Qt objects are only touched on the GUI thread (the poll result
reaches the controller through a queued signal), so a slow or dead network can never freeze a window.
If the host goes away the windows simply keep their last picture and the link keeps retrying.
"""

import http.client
import json
import queue
import threading
import time

from PySide6.QtCore import QObject, Qt, Signal

POLL_SECONDS = 0.04        # 25 snapshots a second: the knob glide in the idle show moves at 15 a second
TIMEOUT_SECONDS = 1.0
RETRY_SECONDS = 1.0
TOUCH_GAP_SECONDS = 0.5    # a visitor's "I'm here" only needs sending now and then


def parse_host(text: str, default_port: int) -> tuple[str, int]:
    """'192.168.1.5', '192.168.1.5:9000', 'mini.local' or 'http://192.168.1.5:8765/' -> (host, port)."""
    text = text.strip()
    for prefix in ("http://", "https://"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.split("/", 1)[0]
    if not text:
        raise ValueError("empty host")
    host, sep, port = text.rpartition(":")
    if not sep:
        return text, default_port
    if not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError(f"cannot read a host and port from {text!r}")
    return host, int(port)


class FollowerLink(QObject):
    """Mirrors a host controller into `controller` and sends the follower's own input to the host."""

    synced = Signal(dict)  # worker thread -> GUI thread

    def __init__(self, controller, host: str, port: int):
        super().__init__()
        self.controller = controller
        self.host, self.port = host, port
        self.connected = False
        self._stop = threading.Event()
        self._outbox: "queue.Queue[tuple[str, float]]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._last_touch = 0.0
        self.synced.connect(self._apply, Qt.QueuedConnection)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self):
        """Route this controller's input to the host and start mirroring it."""
        self.controller.forward = self.send
        for target, name in ((self._poll_loop, "follow-poll"), (self._send_loop, "follow-send")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)

    # -- GUI thread ------------------------------------------------------------------------------
    def send(self, kind: str, value: float):
        """Called by the controller for every input on this Mac; never blocks."""
        if kind == "touch":
            now = time.monotonic()
            if now - self._last_touch < TOUCH_GAP_SECONDS:
                return
            self._last_touch = now
        self._outbox.put((kind, float(value)))

    def _apply(self, snap: dict):
        try:
            self.controller.apply_sync(snap)
        except (KeyError, TypeError, ValueError) as exc:  # a host running other code: ignore the snapshot
            print(f"[follow] ignored a snapshot from the host: {exc!r}")

    # -- worker threads --------------------------------------------------------------------------
    def _connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(self.host, self.port, timeout=TIMEOUT_SECONDS)

    def _set_connected(self, up: bool):
        if up != self.connected:
            self.connected = up
            print(f"[follow] {'connected to' if up else 'lost'} the host at {self.url}"
                  + ("" if up else " (will keep trying)"))

    def _poll_loop(self):
        conn = self._connection()
        while not self._stop.is_set():
            try:
                conn.request("GET", "/sync")
                reply = conn.getresponse()
                body = reply.read()
                if reply.status != 200:
                    raise OSError(f"host answered {reply.status}")
                snap = json.loads(body)
                self._set_connected(True)
                self.synced.emit(snap)
                self._stop.wait(POLL_SECONDS)
            except (OSError, http.client.HTTPException, ValueError):
                self._set_connected(False)
                conn.close()
                conn = self._connection()
                self._stop.wait(RETRY_SECONDS)
        conn.close()

    def _send_loop(self):
        conn = self._connection()
        while not self._stop.is_set():
            try:
                kind, value = self._outbox.get(timeout=0.2)
            except queue.Empty:
                continue
            body = json.dumps({"kind": kind, "value": value}).encode()
            try:
                conn.request("POST", "/cmd", body=body, headers={"Content-Type": "application/json"})
                conn.getresponse().read()
            except (OSError, http.client.HTTPException):
                conn.close()
                conn = self._connection()  # this input is lost; the next one retries
        conn.close()
