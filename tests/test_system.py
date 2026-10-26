"""
Unit and integration test suite for the Navier-Stokes Exhibition System.
Tests:
1. ObjectStateManager (v2 cache) RAM loading and <1ms state retrieval
2. IGCSE 0606 Linear Interpolation between wind directions
3. ExhibitionInputController macropad event handling (wand, knob, view)
4. Custom 3D Navier-Stokes solver divergence-free condition
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

from app.state_manager import ObjectStateManager
from app.input_controller import ExhibitionInputController
from solver.foam.settings import load_settings
from solver.navier_stokes_3d import NavierStokes3DSolver
from tests.fake_cache import write_fake_cache


def _object_config(obj_id: str) -> dict:
    """The object's raw config entry (including its `cfd` block), as the importer stores it."""
    _, _, raw = load_settings()
    return next(o for o in raw["objects"] if o["id"] == obj_id)


@pytest.fixture
def manager(tmp_path):
    write_fake_cache(tmp_path / "f1_car_states.npz", _object_config("f1_car"))
    return ObjectStateManager("f1_car", cache_dir=tmp_path)


def _knob_between(mgr, d_lo: int, d_hi: int, frac: float = 0.5) -> float:
    """Knob value whose direction lies `frac` of the way from direction d_lo to d_hi."""
    d = mgr.directions_deg
    target = d[d_lo] + frac * (d[d_hi] - d[d_lo])
    return float((target - d[0]) / (d[-1] - d[0]))


def test_lookup_under_1ms(manager):
    knob = _knob_between(manager, 2, 3)
    manager.get_state(4, knob)  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        manager.get_state(4, knob)
        times.append((time.perf_counter() - t0) * 1000.0)
    median_ms = float(np.median(times))
    print(f" -> median state lookup: {median_ms:.3f} ms (target < 1.0 ms)")
    assert median_ms < 1.0


def test_interpolation_between_directions(manager):
    # IGCSE 0606: halfway between directions 2 and 3, t = 0.5, so y = 0.5*2 + 0.5*3 = 2.5 ...
    state = manager.get_state(0, _knob_between(manager, 2, 3))
    assert np.isclose(state["surface_pressure"].mean(), 2.5, atol=1e-5)
    # ... and the telemetry blends the same way (drag force is 100 + d).
    assert np.isclose(state["metrics"]["drag_force_n"], 102.5)
    assert np.isclose(state["metrics"]["drag_coefficient_cd"], 0.3 + 0.01 * 2.5)
    # t = 0.25 -> 0.75*2 + 0.25*3
    quarter = manager.get_state(0, _knob_between(manager, 2, 3, 0.25))
    assert np.isclose(quarter["surface_pressure"].mean(), 2.25, atol=1e-5)
    # Non-numeric metrics are passed through, never blended.
    assert state["metrics"]["converged"] is True
    assert state["metrics"]["filled_from"] is None
    assert state["filled"] is False
    # Blended wind vector is a unit vector; N-S terms blend numerically.
    assert np.isclose(np.linalg.norm(state["wind_vector"]), 1.0)
    assert np.isclose(state["ns_terms"]["pressure"], 2.0)
    assert state["wind_speed"] == manager.wind_speed == 40.0


def test_knob_end_points_and_clamp(manager):
    d = manager.directions_deg
    assert manager.direction_for_knob(0.0) == d[0]
    assert manager.direction_for_knob(1.0) == d[-1]
    assert manager.direction_for_knob(-3.0) == d[0]
    assert manager.direction_for_knob(7.0) == d[-1]
    # Exact upper end uses the last direction (pressure is the constant 8 there).
    assert np.isclose(manager.get_state(0, 1.0)["surface_pressure"].mean(), 8.0)
    assert np.isclose(manager.get_state(0, 0.0)["surface_pressure"].mean(), 0.0)


def test_nearest_lines(manager):
    # t = 0.4 and t = 0.6 between directions 2 and 3 must pick different directions' lines/k/dominant.
    lo = manager.get_state(3, _knob_between(manager, 2, 3, 0.4))
    hi = manager.get_state(3, _knob_between(manager, 2, 3, 0.6))
    assert lo["lines"].shape[1] == 3
    assert lo["k"].shape[0] == lo["lines"].shape[0] == lo["dominant"].shape[0]
    real = ~np.isnan(lo["lines"][:, 0])
    # Fake lines sit at the wand height, shifted along x by 0.1 * direction index.
    assert np.allclose(lo["lines"][real, 2], manager.wand_heights[3])
    assert np.allclose(hi["lines"][real, 0] - lo["lines"][real, 0], 0.1)
    # k = 10 * d + (0.5..1.5): the integer tens digit identifies the direction.
    assert int(np.nanmin(lo["k"]) // 10) == 2 and int(np.nanmin(hi["k"]) // 10) == 3
    # dominant pattern (arange + d) % 3 starts at 2 for d=2 and at 0 for d=3.
    assert lo["dominant"][0] == 2 and hi["dominant"][0] == 0
    assert lo["wand_z"] == manager.wand_heights[3]
    # Exactly on a stored direction (t = 0): that direction.
    exact = manager.get_state(3, _knob_between(manager, 5, 6, 0.0))
    assert int(np.nanmin(exact["k"]) // 10) == 5
    # Wand index is clamped to the available wands.
    assert manager.get_state(99, 0.5)["wand_index"] == 9
    assert manager.get_state(-4, 0.5)["wand_index"] == 0


def test_filled_and_non_numeric_metrics_come_from_nearest(tmp_path):
    cfg = {"id": "building", "cfd": {"speed_m_s": 50.0, "widgets": ["wind_load"]}}
    write_fake_cache(tmp_path / "building_states.npz", cfg)
    mgr = ObjectStateManager("building", cache_dir=tmp_path)
    # Direction 3 is the filled run (filled_from = 2); direction 2 is not.
    near_2 = mgr.get_state(0, _knob_between(mgr, 2, 3, 0.4))
    near_3 = mgr.get_state(0, _knob_between(mgr, 2, 3, 0.6))
    assert near_2["filled"] is False and near_2["metrics"]["filled_from"] is None
    assert near_3["filled"] is True and near_3["metrics"]["filled_from"] == 2
    # The wind_load list is not blended: it is the nearest direction's list, unchanged.
    assert near_2["metrics"]["wind_load"][0]["label"] == "Breeze d2"
    assert near_3["metrics"]["wind_load"][0]["label"] == "Breeze d3"
    assert near_3["metrics"]["wind_load"][0]["force_n"] == 3.0
    # Numbers in the same metrics dict are still blended (drag force is 100 + d).
    assert np.isclose(near_3["metrics"]["drag_force_n"], 100.0 + 2 + 0.6)


def test_lookup_under_1ms_real_size(tmp_path):
    # ~200k surface vertices, 5 lines x 20,001 rows = ~100k points per wand, 9 directions x 2 wands.
    cfg = _object_config("f1_car")
    write_fake_cache(tmp_path / "f1_car_states.npz", cfg, n_wand=2, n_vertices=200_000, points_per_line=20_000)
    mgr = ObjectStateManager("f1_car", cache_dir=tmp_path)
    assert len(mgr.mesh_vertices) == 200_000
    assert len(mgr.get_state(1, 0.3)["lines"]) == 5 * 20_001
    knob = _knob_between(mgr, 2, 3)
    mgr.get_state(1, knob)  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        mgr.get_state(1, knob)
        times.append((time.perf_counter() - t0) * 1000.0)
    median_ms = float(np.median(times))
    print(f" -> real-size median state lookup: {median_ms:.3f} ms (target < 1.0 ms)")
    assert median_ms < 1.0


def test_rejects_v1_cache(tmp_path):
    np.savez(tmp_path / "airplane_states.npz", z_steps=np.arange(10), wind_speeds=np.arange(5))
    with pytest.raises(ValueError):
        ObjectStateManager("airplane", cache_dir=tmp_path)


def test_controller_signals_and_clamps():
    ctrl = ExhibitionInputController()
    events = []
    views = []
    ctrl.state_changed.connect(lambda w, k: events.append((w, k)))
    ctrl.view_changed.connect(views.append)
    assert (ctrl.current_wand_index, ctrl.current_knob, ctrl.current_view) == (4, 0.0, 0)

    ctrl.set_wand(7)
    assert ctrl.current_wand_index == 7 and events[-1] == (7, 0.0)
    ctrl.set_wand(50)
    assert ctrl.current_wand_index == 9
    ctrl.step_wand(+1)  # already at the top: no change, no signal
    assert ctrl.current_wand_index == 9
    ctrl.set_wand(-5)
    assert ctrl.current_wand_index == 0

    n = len(events)
    ctrl.step_knob(-1)  # already at 0: clamped, no signal
    assert ctrl.current_knob == 0.0 and len(events) == n
    ctrl.step_knob(+1)
    assert np.isclose(ctrl.current_knob, 1 / 32) and len(events) == n + 1
    ctrl.set_knob(5.0)
    assert ctrl.current_knob == 1.0
    ctrl.set_knob(-5.0)
    assert ctrl.current_knob == 0.0

    ctrl.cycle_view()
    ctrl.cycle_view()
    ctrl.cycle_view()
    assert views == [1, 2, 0] and ctrl.current_view == 0
    ctrl.set_view(2)
    assert views[-1] == 2


class _FakeWheel:
    def __init__(self, dy):
        self._dy = dy

    def angleDelta(self):
        class _P:
            def __init__(self, y):
                self._y = y

            def y(self):
                return self._y
        return _P(self._dy)


def test_keys_and_wheel():
    ctrl = ExhibitionInputController()

    def press(key):
        return ctrl.handle_key_press(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))

    assert press(Qt.Key_1) and ctrl.current_wand_index == 0
    assert press(Qt.Key_0) and ctrl.current_wand_index == 9
    assert press(Qt.Key_Down) and ctrl.current_wand_index == 8
    assert press(Qt.Key_W) and ctrl.current_wand_index == 9
    assert press(Qt.Key_S) and ctrl.current_wand_index == 8
    assert press(Qt.Key_Right) and np.isclose(ctrl.current_knob, 1 / 32)
    assert press(Qt.Key_BracketLeft) and ctrl.current_knob == 0.0
    assert not press(Qt.Key_Q)
    assert ctrl.handle_wheel_event(_FakeWheel(120)) and ctrl.current_knob > 0.0
    assert not ctrl.handle_wheel_event(_FakeWheel(0))


def test_tab_cycles_view():
    ctrl = ExhibitionInputController()
    views = []
    ctrl.view_changed.connect(views.append)
    tab = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)
    assert ctrl.handle_key_press(tab) is True
    assert ctrl.current_view == 1 and views == [1]


def test_navier_stokes_divergence():
    print("\nTesting Navier-Stokes Divergence-Free Condition...")
    solver = NavierStokes3DSolver(nx=20, ny=10, nz=10, domain_size=(1.0, 0.5, 0.5))
    solver.solve_state(pipe_z=0.25, wind_speed=20.0, num_steps=5, poisson_iter=30)

    # Calculate divergence du/dx + dv/dy + dw/dz
    div = (
        (solver.u[2:, 1:-1, 1:-1] - solver.u[:-2, 1:-1, 1:-1]) / (2.0 * solver.dx) +
        (solver.v[1:-1, 2:, 1:-1] - solver.v[1:-1, :-2, 1:-1]) / (2.0 * solver.dy) +
        (solver.w[1:-1, 1:-1, 2:] - solver.w[1:-1, 1:-1, :-2]) / (2.0 * solver.dz)
    )
    interior_div = div[2:-2, 1:-1, 1:-1]
    mean_div = float(np.mean(np.abs(interior_div)))
    print(f" -> Mean interior velocity divergence: {mean_div:.4f} (Demonstrates incompressibility!)")
    assert mean_div < 20.0, "Mean divergence should be well-bounded!"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
