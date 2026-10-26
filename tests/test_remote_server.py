"""Tests for app.remote_server: command parsing, the HTTP routes and the GUI-thread hand-off."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import http.client
import json
import re
import socket
import time
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.input_controller import ExhibitionInputController
import app.remote_server as remote_server
from app.remote_server import RemoteServer, parse_command

INDEX_HTML = Path(__file__).resolve().parent.parent / "app" / "remote" / "index.html"


@pytest.fixture(scope="module")
def qapp():
    # QApplication is a QCoreApplication; reusing the same kind the other GUI tests create
    # avoids poisoning them when the whole suite runs in one process.
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qapp):
    return ExhibitionInputController()


@pytest.fixture
def server(qapp, controller):
    srv = RemoteServer(controller, port=0)
    url = srv.start()
    srv.test_url = url
    yield srv
    srv.stop()


def pump(predicate=lambda: False, timeout=2.0):
    """Process Qt events until predicate() is true (or the timeout passes) so queued signals deliver."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


def conn(server):
    port = int(re.search(r":(\d+)/$", server.test_url).group(1))
    return http.client.HTTPConnection("127.0.0.1", port, timeout=5)


def request(server, method, path, body=None, headers=None):
    c = conn(server)
    try:
        c.request(method, path, body=body, headers=headers or {})
        r = c.getresponse()
        return r.status, r.read(), dict(r.getheaders())
    finally:
        c.close()


def post_cmd(server, payload):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return request(server, "POST", "/cmd", body=body, headers={"Content-Type": "application/json"})


# ---- parse_command -------------------------------------------------------------------------

def test_parse_command_accepts_valid():
    assert parse_command(b'{"kind": "knob", "value": 0.25}') == ("knob", 0.25)
    assert parse_command(b'{"kind": "wand", "value": 3}') == ("wand", 3.0)
    kind, value = parse_command(b'{"kind":"view","value":2}')
    assert kind == "view" and value == 2.0 and isinstance(value, float)


@pytest.mark.parametrize("body", [
    b"{",                                    # not JSON
    b"",                                     # empty
    b"\xff\xfe",                             # not UTF-8
    b"[1, 2]",                               # JSON but not an object
    b'{"kind":"x"}',                         # unknown kind, no value
    b'{"kind":"x","value":1}',               # unknown kind
    b'{"kind":"knob"}',                      # missing value
    b'{"kind":"knob","value":"nan"}',        # string, not a number
    b'{"kind":"knob","value":"0.5"}',        # numeric string is still a string
    b'{"kind":"knob","value":true}',         # bool is not a number here
    b'{"kind":"knob","value":null}',
    b'{"kind":"knob","value":1e999}',        # overflows to infinity
    b'{"kind":"knob","value":NaN}',          # JSON-extension NaN
    b'{"kind":"wand","value":-Infinity}',
    b'{"kind":"wand","value":1' + b"0" * 400 + b"}",   # int too big for a float
    b"[" * 100000,                           # nesting bomb
    b'{"kind":["knob"],"value":1}',          # unhashable kind
])
def test_parse_command_rejects_bad_input(body):
    with pytest.raises(ValueError):
        parse_command(body)


# ---- HTTP routes ---------------------------------------------------------------------------

def test_post_cmd_moves_controller(server, controller):
    status, _, _ = post_cmd(server, {"kind": "knob", "value": 0.5})
    assert status == 204
    assert pump(lambda: controller.current_knob == 0.5)
    assert controller.current_knob == 0.5


def test_wand_is_zero_based_and_view_moves(server, controller):
    assert post_cmd(server, {"kind": "wand", "value": 0})[0] == 204
    assert post_cmd(server, {"kind": "view", "value": 2})[0] == 204
    assert pump(lambda: controller.current_wand_index == 0 and controller.current_view == 2)
    assert post_cmd(server, {"kind": "wand", "value": 9})[0] == 204
    assert pump(lambda: controller.current_wand_index == 9)


def test_out_of_range_values_are_clamped_by_controller(server, controller):
    assert post_cmd(server, {"kind": "knob", "value": 7.5})[0] == 204
    assert post_cmd(server, {"kind": "wand", "value": 1e300})[0] == 204
    assert pump(lambda: controller.current_knob == 1.0 and controller.current_wand_index == controller.wand_count - 1)


def test_bad_post_is_400_and_state_unchanged(server, controller):
    before = (controller.current_wand_index, controller.current_knob, controller.current_view)
    for bad in (b"{", b'{"kind":"x"}', b'{"kind":"knob","value":"nan"}', b'{"kind":"knob","value":1e999}'):
        status, _, _ = post_cmd(server, bad)
        assert status == 400
    pump(timeout=0.2)
    assert (controller.current_wand_index, controller.current_knob, controller.current_view) == before


def test_get_state_json(server, controller):
    controller.set_wand(2)
    controller.set_knob(0.75)
    controller.set_view(1)
    status, body, headers = request(server, "GET", "/state")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body) == {"wand": 2, "knob": 0.75, "view": 1, "wand_count": 10}


def test_get_index_is_self_contained_html(server):
    status, body, headers = request(server, "GET", "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    html = body.decode("utf-8")
    assert "<svg" in html and 'type="range"' in html
    assert body == INDEX_HTML.read_bytes()
    # No external assets: nothing may be loaded from another host.
    assert not re.findall(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", html)
    assert "@import" not in html and "url(http" not in html


def test_oversized_body_is_400_and_server_survives(server, controller):
    big = b'{"kind":"knob","value":0.5,"pad":"' + b"x" * 2000 + b'"}'
    assert post_cmd(server, big)[0] == 400
    pump(timeout=0.1)
    assert controller.current_knob == 0.0
    assert request(server, "GET", "/state")[0] == 200


def test_bad_content_length_is_400(server):
    for value in ("abc", "-5"):
        c = conn(server)
        try:
            c.putrequest("POST", "/cmd")
            c.putheader("Content-Length", value)
            c.endheaders()
            assert c.getresponse().status == 400
        finally:
            c.close()


def test_unknown_routes_and_methods_do_not_crash(server):
    assert request(server, "GET", "/nope")[0] == 404
    assert request(server, "POST", "/state", body=b"{}")[0] in (404, 405)
    assert request(server, "PUT", "/cmd", body=b"{}")[0] in (404, 405, 501)
    assert request(server, "GET", "/state")[0] == 200


def test_handler_error_is_a_500_not_a_crash(server, controller):
    controller_attr = "current_wand_index"
    # Make GET /state blow up inside the handler: a non-numeric attribute cannot be JSON-ised
    # into the contract, so the handler must answer 500 and keep serving.
    class Boom:
        def __float__(self):
            raise RuntimeError("boom")
    setattr(controller, "current_knob", Boom())
    assert request(server, "GET", "/state")[0] == 500
    controller.current_knob = 0.0
    assert request(server, "GET", "/state")[0] == 200
    assert controller_attr


def test_failing_controller_slot_does_not_crash_gui_thread(server, controller):
    def explode(_):
        raise RuntimeError("controller broke")
    controller.set_knob = explode
    assert post_cmd(server, {"kind": "knob", "value": 0.5})[0] == 204
    pump(timeout=0.2)  # delivering the queued command must swallow the exception
    assert request(server, "GET", "/state")[0] == 200


# ---- lifecycle -----------------------------------------------------------------------------

def test_start_reports_actual_port_and_lan_url(qapp, controller):
    srv = RemoteServer(controller, port=0)
    try:
        url = srv.start()
        m = re.fullmatch(r"http://(\d+\.\d+\.\d+\.\d+):(\d+)/", url)
        assert m, url
        assert int(m.group(2)) != 0
    finally:
        srv.stop()


def test_stop_frees_port_and_is_idempotent(qapp, controller):
    srv = RemoteServer(controller, port=0)
    url = srv.start()
    port = int(re.search(r":(\d+)/$", url).group(1))
    thread = srv._thread
    srv.stop()
    assert not thread.is_alive()
    srv.stop()  # a second stop must be harmless
    RemoteServer(controller, port=0).stop()  # stop before start too
    with socket.socket() as s:  # the port can be bound again
        s.bind(("0.0.0.0", port))


def test_busy_port_raises_oserror_with_clear_message(qapp, controller):
    with socket.socket() as blocker:
        blocker.bind(("0.0.0.0", 0))
        blocker.listen(1)
        busy = blocker.getsockname()[1]
        srv = RemoteServer(controller, port=busy)
        with pytest.raises(OSError, match=str(busy)):
            srv.start()
        srv.stop()  # must still be safe after a failed start


# ---- lan_ip fallbacks (isolated router / phone hotspot: no default route) ------------------

def test_lan_ip_default_route_probe_wins(monkeypatch):
    monkeypatch.setattr(remote_server, "_probe_route", lambda target: "192.168.1.7")
    monkeypatch.setattr(remote_server, "_hostname_addresses", lambda: ["10.9.9.9"])
    assert remote_server.lan_ip() == "192.168.1.7"


def test_lan_ip_without_default_route_uses_hotspot_probe(monkeypatch):
    # only the iPhone-hotspot gateway probe finds an interface
    monkeypatch.setattr(remote_server, "_probe_route", lambda t: "172.20.10.3" if t == "172.20.10.1" else None)
    monkeypatch.setattr(remote_server, "_hostname_addresses", lambda: [])
    assert remote_server.lan_ip() == "172.20.10.3"


def test_lan_ip_without_any_route_picks_private_hostname_address(monkeypatch):
    monkeypatch.setattr(remote_server, "_probe_route", lambda target: None)
    monkeypatch.setattr(remote_server, "_hostname_addresses",
                        lambda: ["127.0.0.1", "127.0.1.1", "169.254.3.4", "8.8.4.4", "192.168.4.20", "10.0.0.2"])
    ip = remote_server.lan_ip()
    assert ip == "192.168.4.20"  # first private, non-loopback, non-link-local IPv4


def test_lan_ip_ignores_public_address_from_gateway_probe(monkeypatch):
    monkeypatch.setattr(remote_server, "_probe_route", lambda t: "93.184.216.34" if t == "192.168.255.255" else None)
    monkeypatch.setattr(remote_server, "_hostname_addresses", lambda: ["10.1.2.3"])
    assert remote_server.lan_ip() == "10.1.2.3"


def test_lan_ip_loopback_when_nothing_works(monkeypatch):
    monkeypatch.setattr(remote_server, "_probe_route", lambda target: None)
    monkeypatch.setattr(remote_server, "_hostname_addresses", lambda: ["127.0.1.1"])
    assert remote_server.lan_ip() == "127.0.0.1"
