"""Tests for the two-computer link: --follow address parsing, window tiling, snapshots and a live host + follower."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import re
import time

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from app.follower import FollowerLink, parse_follow, parse_host
from app.input_controller import ExhibitionInputController
from app.main import tile
from app.remote_server import RemoteServer


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def pump(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    QCoreApplication.processEvents()
    return predicate()


# ---- --follow address ----------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("192.168.1.5", ("192.168.1.5", 8765)),
    ("192.168.1.5:9000", ("192.168.1.5", 9000)),
    ("mini.local", ("mini.local", 8765)),
    ("http://100.114.120.115:8765/", ("100.114.120.115", 8765)),
    ("  10.0.0.2  ", ("10.0.0.2", 8765)),
])
def test_parse_host(text, expected):
    assert parse_host(text, 8765) == expected


@pytest.mark.parametrize("text,expected", [
    ("192.168.1.5", ("192.168.1.5", 8765, False)),
    ("http://192.168.1.5:9000/", ("192.168.1.5", 9000, False)),
    ("https://abc-def.ngrok-free.dev", ("abc-def.ngrok-free.dev", 443, True)),
    ("HTTPS://abc-def.ngrok-free.dev/", ("abc-def.ngrok-free.dev", 443, True)),
    ("https://abc-def.ngrok-free.dev:8443", ("abc-def.ngrok-free.dev", 8443, True)),
])
def test_parse_follow(text, expected):
    assert parse_follow(text, 8765) == expected


def test_follower_url_for_https_and_http():
    assert FollowerLink(None, "x.ngrok-free.dev", 443, secure=True).url == "https://x.ngrok-free.dev/"
    assert FollowerLink(None, "10.0.0.2", 8765).url == "http://10.0.0.2:8765/"


@pytest.mark.parametrize("text", ["", "http://", ":8765", "host:", "host:abc", "host:0", "host:70000"])
def test_parse_host_rejects_bad_input(text):
    with pytest.raises(ValueError):
        parse_host(text, 8765)


# ---- window tiling -------------------------------------------------------------------------

AREA = (0, 0, 1000, 800)


def test_tile_one_window_fills_the_area_with_a_margin():
    assert tile(1, 0, AREA) == (50, 50, 900, 700)


def test_tile_two_windows_sit_side_by_side():
    assert tile(2, 0, AREA) == (0, 0, 500, 800)
    assert tile(2, 1, AREA) == (500, 0, 500, 800)


def test_tile_four_windows_make_a_2x2_grid_in_reading_order():
    assert [tile(4, i, AREA) for i in range(4)] == [
        (0, 0, 500, 400), (500, 0, 500, 400), (0, 400, 500, 400), (500, 400, 500, 400)]


def test_tile_three_windows_use_the_first_three_cells():
    assert [tile(3, i, AREA)[:2] for i in range(3)] == [(0, 0), (500, 0), (0, 400)]
    assert tile(3, 0, AREA)[3] == 400


# ---- snapshot and apply --------------------------------------------------------------------

def test_apply_sync_copies_the_host_and_emits_one_signal_per_change(qapp):
    c = ExhibitionInputController()
    seen = []
    c.state_changed.connect(lambda w, k: seen.append(("state", w, k)))
    c.view_changed.connect(lambda v: seen.append(("view", v)))
    c.detail_changed.connect(lambda d: seen.append(("detail", d)))
    c.show_changed.connect(lambda s: seen.append(("show", s)))
    c.show_step_changed.connect(lambda n: seen.append(("step", n)))
    c.camera_reset.connect(lambda: seen.append(("reset",)))
    snap = {"wand": 7, "knob": 0.25, "view": 2, "detailed": True, "show_active": True,
            "show_step": 3, "reset_count": 1, "wand_count": 10}
    c.apply_sync(snap)
    assert seen == [("state", 7, 0.25), ("view", 2), ("detail", True), ("show", True), ("step", 3), ("reset",)]
    seen.clear()
    c.apply_sync(snap)  # nothing changed: nothing emitted
    assert seen == []


def test_show_slot_sees_the_new_step_when_the_show_starts(qapp):
    c = ExhibitionInputController()
    steps = []
    c.show_changed.connect(lambda on: steps.append(c.show_step))  # like MonitorWindow.on_show_changed
    c.apply_sync({"wand": 4, "knob": 0.0, "view": 0, "detailed": False, "show_active": True,
                  "show_step": 4, "reset_count": 0, "wand_count": 10})
    assert steps == [4]


def test_apply_sync_clamps_out_of_range_values(qapp):
    c = ExhibitionInputController()
    c.apply_sync({"wand": 99, "knob": 7.0, "view": 5, "detailed": False, "show_active": False,
                  "show_step": 999, "reset_count": 0, "wand_count": 10})
    assert (c.current_wand_index, c.current_knob, c.current_view) == (9, 1.0, 2)
    assert 0 <= c.show_step


def test_snapshot_round_trips(qapp):
    host, other = ExhibitionInputController(), ExhibitionInputController()
    host.set_wand(1)
    host.set_knob(0.6)
    host.set_view(2)
    host.set_detail(True)
    host.request_camera_reset()
    other.apply_sync(host.snapshot())
    assert other.snapshot() == host.snapshot()


def test_follower_controller_forwards_instead_of_changing_itself(qapp):
    c = ExhibitionInputController()
    sent = []
    c.forward = lambda kind, value: sent.append((kind, value))
    c.set_wand(3)
    c.set_knob(0.5)
    c.set_view(1)
    c.toggle_detail()
    c.request_camera_reset()
    c.touch()
    assert sent == [("wand", 3), ("knob", 0.5), ("view", 1), ("detail", 1), ("reset", 0), ("touch", 0)]
    assert (c.current_wand_index, c.current_knob, c.current_view, c.detailed) == (4, 0.0, 0, False)


def test_follower_never_starts_its_own_show(qapp):
    c = ExhibitionInputController(idle_seconds=0.01)
    c.forward = lambda kind, value: None
    c.start_clock()
    assert not c.show_active


# ---- a real host and follower over localhost -----------------------------------------------

@pytest.fixture
def pair(qapp):
    host = ExhibitionInputController(idle_seconds=0)
    server = RemoteServer(host, port=0)
    port = int(re.search(r":(\d+)/$", server.start()).group(1))
    follower_ctrl = ExhibitionInputController()
    link = FollowerLink(follower_ctrl, "127.0.0.1", port)
    link.start()
    yield host, follower_ctrl, link
    link.stop()
    server.stop()


def test_follower_mirrors_the_host(pair):
    host, follower, _ = pair
    host.set_wand(8)
    host.set_knob(0.75)
    host.set_view(2)
    host.set_detail(True)
    assert pump(lambda: follower.snapshot() == host.snapshot())


def test_follower_camera_reset_follows_the_host(pair):
    host, follower, _ = pair
    resets = []
    follower.camera_reset.connect(lambda: resets.append(1))
    host.request_camera_reset()
    assert pump(lambda: len(resets) == 1)


def test_input_on_the_follower_reaches_the_host_and_comes_back(pair):
    host, follower, _ = pair
    follower.set_wand(1)
    follower.set_knob(0.5)
    follower.toggle_detail()
    assert pump(lambda: host.current_wand_index == 1 and host.current_knob == 0.5 and host.detailed)
    assert pump(lambda: follower.snapshot() == host.snapshot())


def test_a_touch_on_the_follower_stops_the_hosts_show(pair):
    host, follower, _ = pair
    host.start_show()
    assert pump(lambda: follower.show_active)
    follower.touch()
    assert pump(lambda: not host.show_active)
    assert pump(lambda: not follower.show_active)


def test_follower_survives_a_dead_host(qapp):
    follower_ctrl = ExhibitionInputController()
    link = FollowerLink(follower_ctrl, "127.0.0.1", 1)  # nothing listens on port 1
    link.start()
    follower_ctrl.set_knob(0.5)  # must not block or raise
    pump(lambda: False, timeout=0.3)
    assert not link.connected
    link.stop()


# ---- theme is shared like everything else ---------------------------------------------------

def test_theme_switches_with_a_signal_and_a_key(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent

    c = ExhibitionInputController()
    seen = []
    c.theme_changed.connect(seen.append)
    c.set_theme("dark")
    c.set_theme("dark")        # no change, no signal
    c.set_theme("blue")        # unknown names are ignored
    c.handle_key_press(QKeyEvent(QEvent.KeyPress, Qt.Key_T, Qt.NoModifier))
    assert seen == ["dark", "light"] and c.theme == "light"


def test_theme_follows_the_host(pair):
    host, follower, _ = pair
    host.set_theme("dark")
    assert pump(lambda: follower.theme == "dark")
    follower.set_theme("light")  # a tap on the follower goes to the host and comes back
    assert pump(lambda: host.theme == "light" and follower.theme == "light")
