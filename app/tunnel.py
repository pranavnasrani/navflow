"""
ngrok tunnel for the iPad remote (`app/main.py --ngrok`).

The remote is served by the host app on one local port (controls.remote_port). ngrok forwards a public
https address to that port, so an iPad can reach it from any network: a school Wi-Fi that blocks devices from
seeing each other, or a phone hotspot, with no address to type on the Mac's side. https is also what lets the
page be added to the iPad's Home Screen as an app (app/remote/manifest.webmanifest).

    ngrok http <port> [--url=<your-reserved-domain>]

ngrok needs a free account once: `ngrok config add-authtoken <token>` (the token is on the ngrok dashboard,
Your Authtoken). A reserved domain (also free, one per account) keeps the same address every day, so the
Home Screen icon keeps working; without one ngrok picks a new random address each run.

Anyone who has the address can use the controls (wand, knob, view, theme), so treat it like a password.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
import urllib.request

API_URL = "http://127.0.0.1:4040/api/tunnels"  # the agent's local status page
TOKEN_HINT = ("ngrok needs your account's token once:  ngrok config add-authtoken <token>\n"
              "    (copy it from https://dashboard.ngrok.com/get-started/your-authtoken)")


def ngrok_command(port: int, domain: str | None = None) -> list[str]:
    """The command line that forwards a public address to localhost:<port>."""
    cmd = ["ngrok", "http", str(int(port)), "--log=stdout", "--log-format=json"]
    if domain:
        cmd.append(f"--url={domain.removeprefix('https://').rstrip('/')}")
    return cmd


def parse_public_url(api_json: dict) -> str | None:
    """The https address from the agent's /api/tunnels reply, or None while the tunnel is still starting."""
    tunnels = api_json.get("tunnels") or []
    urls = [t.get("public_url", "") for t in tunnels if isinstance(t, dict)]
    https = [u for u in urls if u.startswith("https://")]
    return (https or urls or [None])[0] or None


class NgrokTunnel:
    """Runs ngrok as a child process and reports the public address (or why it could not start)."""

    def __init__(self, port: int, domain: str | None = None, say=print):
        self.port, self.domain, self.say = port, domain, say
        self.url: str | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._tail: list[str] = []

    def start(self) -> bool:
        """Start ngrok. Returns False (after saying why) when it is not installed."""
        if shutil.which("ngrok") is None:
            self.say("[ngrok] not installed: run  brew install ngrok  and  ngrok config add-authtoken <token>")
            return False
        self._proc = subprocess.Popen(
            ngrok_command(self.port, self.domain), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        threading.Thread(target=self._read_log, name="ngrok-log", daemon=True).start()
        threading.Thread(target=self._wait_for_url, name="ngrok-url", daemon=True).start()
        return True

    def stop(self) -> None:
        self._stop.set()
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    # -- worker threads --------------------------------------------------------------------------
    def _read_log(self) -> None:
        """Keep the last log lines; if ngrok exits on its own, say why."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self._tail = (self._tail + [line.rstrip()])[-12:]
        code = proc.wait()
        if not self._stop.is_set():
            self.say(f"[ngrok] stopped (exit {code}). " + self._reason())

    def _reason(self) -> str:
        text = "\n".join(self._tail)
        if "authtoken" in text.lower() or "ERR_NGROK_4018" in text:
            return TOKEN_HINT
        messages = []
        for line in self._tail:
            try:
                msg = json.loads(line).get("err") or json.loads(line).get("msg")
            except ValueError:
                msg = line
            if msg and msg not in messages:
                messages.append(str(msg))
        return "; ".join(messages[-3:])

    def _wait_for_url(self) -> None:
        for _ in range(60):  # up to about 30 seconds
            if self._stop.wait(0.5) or self._proc is None or self._proc.poll() is not None:
                return
            try:
                with urllib.request.urlopen(API_URL, timeout=1) as reply:
                    url = parse_public_url(json.load(reply))
            except (OSError, ValueError):
                continue
            if url:
                self.url = url
                self.say(f"[ngrok] iPad remote on the internet: {url}")
                return
        self.say("[ngrok] no address after 30 s. " + self._reason())
