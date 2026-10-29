"""Tests for app.hud_widgets: streamline polylines, log bars, the N-S term panel and the SDG overlay."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from app.hud_widgets import (
    EQUATION,
    NsTermsPanel,
    SustainabilityOverlay,
    compact_si,
    line_runs,
    log_bar_value,
    particles_on_lines,
    polyline_from_nan_joined,
    wand_rod_endpoints,
    wind_arrow_geometry,
)

NAN3 = [np.nan] * 3


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _two_lines():
    return np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], NAN3, [0, 1, 0], [1, 1, 0]], dtype=float)


def test_polyline_from_nan_joined():
    pts = _two_lines()
    scal = {"k": np.array([1, 2, 3, np.nan, 4, 5], dtype=float), "dominant": np.array([0, 1, 2, -1, 0, 1], dtype=np.int8)}
    poly = polyline_from_nan_joined(pts, scal)
    assert poly.n_lines == 2
    assert poly.n_points == 5
    assert len(poly.point_data["k"]) == 5
    assert list(poly.point_data["k"]) == [1, 2, 3, 4, 5]
    assert list(poly.point_data["dominant"]) == [0, 1, 2, 0, 1]
    starts, lengths = line_runs(poly)
    assert list(starts) == [0, 3] and list(lengths) == [3, 2]


def test_polyline_clips_to_bounds_and_drops_single_points():
    pts = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0], [1.5, 0, 0], [1.8, 0, 0], NAN3, [0.2, 0, 0]], dtype=float)
    scal = {"k": np.arange(7, dtype=float)}
    # x > 2 is outside: it splits the first run into (0,1) and (1.5,1.8); the lone last point is dropped.
    poly = polyline_from_nan_joined(pts, scal, bounds=np.array([-1, 2, -1, 1, -1, 1], dtype=float))
    assert poly.n_lines == 2 and poly.n_points == 4
    assert list(poly.point_data["k"]) == [0, 1, 3, 4]
    assert float(np.max(poly.points[:, 0])) <= 2.0


def test_polyline_empty_input():
    poly = polyline_from_nan_joined(np.empty((0, 3)), {"k": np.empty(0)})
    assert poly.n_points == 0 and poly.n_lines == 0
    all_nan = polyline_from_nan_joined(np.array([NAN3]), {"k": np.array([np.nan])})
    assert all_nan.n_points == 0


def test_particles_stay_on_their_lines():
    poly = polyline_from_nan_joined(_two_lines())
    starts, lengths = line_runs(poly)
    for phase in (0.0, 0.3, 0.77):
        p = particles_on_lines(np.asarray(poly.points), starts, lengths, phase, per_line=3)
        assert p.shape == (6, 3)
        assert np.all(p[:3, 1] == 0.0) and np.all(p[3:, 1] == 1.0)  # first line at y = 0, second at y = 1
        assert np.all((p[:, 0] >= 0.0) & (p[:, 0] <= 2.0))
    assert particles_on_lines(np.empty((0, 3)), np.empty(0, int), np.empty(0, int), 0.5).shape == (0, 3)


def test_log_bar_value():
    assert log_bar_value(1e-3) == 0
    assert log_bar_value(1e5) == 100
    assert log_bar_value(10) == 50
    assert log_bar_value(0.0) == 0 and log_bar_value(-4.0) == 0 and log_bar_value(float("nan")) == 0
    assert log_bar_value(1e-9) == 0 and log_bar_value(1e9) == 100


def test_ns_panel_marks_dominant_term(qapp):
    panel = NsTermsPanel()
    panel.set_terms({"convective": 5.0, "pressure": 4000.0, "viscous": 0.2})
    assert panel.dominant == "pressure"
    assert panel.equation_label.text() == EQUATION == "ρ(v·∇)v = −∇p + μ∇²v"
    assert "◀" in panel.value_labels["pressure"].text()
    assert "◀" not in panel.value_labels["convective"].text()
    assert panel.bars["pressure"].value() > panel.bars["convective"].value() > panel.bars["viscous"].value()
    assert "turbulent mixing" in panel.name_labels["viscous"].text()
    panel.set_terms({"convective": 9000.0, "pressure": 4000.0, "viscous": 0.2})
    assert panel.dominant == "convective"
    panel.set_terms({"convective": 0.0, "pressure": 0.0, "viscous": 0.0})
    assert panel.dominant is None


def _wind_load(scale=1.0):
    return [
        {"label": "Breeze", "v_m_s": 3.33, "force_n": 1.0e3 * scale, "moment_nm": 2.0e5 * scale},
        {"label": "Category 5 hurricane", "v_m_s": 70, "force_n": 4.4e6 * scale, "moment_nm": 9.0e8 * scale},
    ]


def test_overlay_text(qapp):
    fuel = SustainabilityOverlay(("fuel", "speed"), None)
    fuel.set_metrics({"fuel_burn_l_per_hr": 900.0, "fuel_saved_l_per_hr": 123.4, "co2_saved_kg_per_hr": 311.0,
                      "speed_gain_pct": 8.2})
    text = fuel.all_text()
    assert "L/h" in text and "SDG 9" in text and "SDG 13" not in text and "kg/h" in text and "+8.2 %" in text
    assert "Fuel burned: 900.0 L/h" in text and "Top-speed gain vs worst angle: +8.2 %" in text

    tower = SustainabilityOverlay(("wind_load",), None)
    tower.set_metrics({"wind_load": _wind_load(), "direction_deg": 15.0}, 37.5)
    text = tower.all_text()
    assert "Category 5" in text and "SDG 11" in text and "Worst wind direction: 37.5°" in text


def _fuel_metrics(direction):
    return {
        "direction_deg": direction, "reference_direction_deg": 4.0, "wind_speed_m_s": 102.0, "eval_speed_m_s": 250.0, "eval_altitude_km": 11.0,
        "fuel_burn_l_per_hr": 900.0, "fuel_saved_l_per_hr": 123.4, "co2_saved_kg_per_hr": 311.0, "speed_gain_pct": 8.2,
    }


def test_overlay_worst_angle_text(qapp):
    fuel = SustainabilityOverlay(("fuel", "speed"), None)
    fuel.set_metrics(_fuel_metrics(4.0))  # at the reference angle itself
    text = fuel.all_text()
    assert "worst angle tested" in text and "Saved vs worst angle" not in text and "CO₂ saved" not in text
    fuel.set_metrics(_fuel_metrics(6.0))
    text = fuel.all_text()
    assert "Saved vs worst angle (4°): 123.4 L/h" in text and "CO₂ saved: 311.0 kg/h" in text
    assert "worst angle tested" not in text
    assert "at cruise (11 km, 250 m/s)" in text
    fuel.set_metrics(_fuel_metrics(5.0))  # an interpolated knob position between two angles: the normal text
    assert "Saved vs worst angle" in fuel.all_text()
    m = _fuel_metrics(6.0)
    m["eval_altitude_km"] = 9.0  # the altitude comes from the metrics (config), not a fixed "11 km"
    fuel.set_metrics(m)
    assert "at cruise (9 km, 250 m/s)" in fuel.all_text() and "11 km" not in fuel.all_text()
    m["eval_altitude_km"] = 0.0
    fuel.set_metrics(m)
    assert "at cruise (250 m/s)" in fuel.all_text()


def test_overlay_best_angle_text_for_aircraft(qapp):
    plane = SustainabilityOverlay(("fuel",), None)
    m = dict(_fuel_metrics(4.0), reference_is_best=True, lift_to_drag=17.24)
    plane.set_metrics(m)
    text = plane.all_text()
    assert "best angle tested" in text and "worst" not in text and "L/h" not in text
    assert "Lift ÷ drag = 17.2" in text and "drag = weight ÷ (L/D)" in text
    plane.set_metrics(dict(m, direction_deg=8.0))
    text = plane.all_text()
    # extra / (burn - extra) = 123.4 / 776.6 = 15.9 %; L/D at the best angle = 17.24 x 900 / 776.6 = 20.0
    assert "Fuel vs best angle (4°): +16 %" in text and "20.0 at the best angle" in text
    assert "L/h" not in text and "kg/h" not in text and "Saved" not in text


def test_overlay_cruise_note_only_when_evaluated_off_the_simulated_speed(qapp):
    car = SustainabilityOverlay(("fuel", "speed"), None)
    m = _fuel_metrics(6.0)
    m["eval_speed_m_s"] = m["wind_speed_m_s"]
    car.set_metrics(m)
    assert "cruise" not in car.all_text()


def test_overlay_worst_direction_is_the_given_value_not_a_running_maximum(qapp):
    tower = SustainabilityOverlay(("wind_load",), None)
    # The strongest load is seen first, then weaker ones, at other angles: the text only follows the given worst angle.
    for deg, scale in ((0.0, 1.0), (30.0, 3.0), (60.0, 2.0)):
        tower.set_metrics({"wind_load": _wind_load(scale), "direction_deg": deg}, 22.5)
        assert tower.worst_label.text() == "Worst wind direction: 22.5°"
    assert "so far" not in tower.all_text()
    n_cells = len(tower._table_cells)  # a repeated call does not grow the table
    tower.set_metrics({"wind_load": _wind_load(1.0), "direction_deg": 10.0}, 22.5)
    assert len(tower._table_cells) == n_cells == 2
    tower.set_metrics({"wind_load": _wind_load(1.0), "direction_deg": 10.0})  # no worst direction known
    assert tower.worst_label.text() == ""


def test_overlay_without_widgets_is_hidden(qapp):
    empty = SustainabilityOverlay((), None)
    empty.set_metrics({"fuel_saved_l_per_hr": 1.0})
    assert empty.isHidden()


def test_colorbar_ticks():
    from app.hud_widgets import colorbar_ticks

    lin = colorbar_ticks(-1.0, 1.0, log=False)
    assert [round(f, 2) for f, _ in lin] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert lin[2][1] == "0"
    dec = colorbar_ticks(0.01, 100.0, log=True)  # four decades, evenly spaced on a log scale
    assert [t for _, t in dec] == ["10⁻²", "10⁻¹", "10⁰", "10¹", "10²"]
    assert [round(f, 2) for f, _ in dec] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(colorbar_ticks(2.0, 9.0, log=True)) == 3  # under two decades: low, geometric middle, high
    assert colorbar_ticks(1.0, 1.0, log=False) == []


BOX = np.array([-6.0, 14.0, -4.0, 4.0, -2.0, 2.0])
CENTRE = np.array([0.0, 0.0, 0.0])


def _rot(axis, deg):
    from solver.foam.geometry import rotation_matrix

    return rotation_matrix(axis, deg)


def _check_arrow(start, d, length, wind):
    """The arrow is inside the box, points the way the wind blows and sits upstream of the model centre."""
    lo, hi = BOX[0::2], BOX[1::2]
    assert np.all(start >= lo - 1e-9) and np.all(start <= hi + 1e-9)
    tip = start + length * d
    assert np.all(tip >= lo - 1e-9) and np.all(tip <= hi + 1e-9)
    assert np.allclose(d, wind / np.linalg.norm(wind))
    assert float(np.dot(CENTRE - start, d)) > 0.0  # walking along the wind from the start reaches the model
    assert length > 0.05 * float((hi - lo).max())  # not collapsed


def test_wind_arrow_z_rotation_has_negative_y_component():
    # importer rule: wind in the model frame = R^T [1, 0, 0]; for a z-rotation of 30 deg that is (cos, -sin, 0)
    wind = _rot("z", 30.0).T @ np.array([1.0, 0.0, 0.0])
    assert wind[1] < 0
    start, d, length = wind_arrow_geometry(wind, CENTRE, BOX)
    _check_arrow(start, d, length, wind)
    assert start[0] < 0 and start[1] > 0  # upstream is -x and, because the wind blows toward -y, +y


def test_wind_arrow_y_rotation_has_vertical_component():
    wind = _rot("y", 10.0).T @ np.array([1.0, 0.0, 0.0])
    start, d, length = wind_arrow_geometry(wind, CENTRE, BOX)
    _check_arrow(start, d, length, wind)
    assert wind[2] != 0.0


def test_wind_arrow_all_sign_combinations_and_edge_cases():
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                wind = np.array([sx * 1.0, sy * 0.4, sz * 0.2])
                start, d, length = wind_arrow_geometry(wind, np.array([1.0, 0.5, -0.3]), BOX)
                assert length > 0.1 and np.allclose(np.linalg.norm(d), 1.0)
                assert np.all(start >= BOX[0::2] - 1e-9) and np.all(start <= BOX[1::2] + 1e-9)
    # a zero vector falls back to +x instead of dividing by zero
    _, d, _ = wind_arrow_geometry(np.zeros(3), CENTRE, BOX)
    assert np.allclose(d, [1, 0, 0])


def test_wand_rod_endpoints():
    lines = np.array([[0, -1, 5], [1, -1, 5], NAN3, [0, 1, 5], [1, 1, 5], NAN3, [0, 0, 5], [1, 0, 5]], dtype=float)
    a, e = wand_rod_endpoints(lines, 5.0, BOX)
    assert np.allclose(a, [0, -1, 5]) and np.allclose(e, [0, 1, 5])
    a, e = wand_rod_endpoints(np.empty((0, 3)), 1.5, BOX)  # no streamlines: across y at the wand height
    assert a[1] == BOX[2] and e[1] == BOX[3] and a[2] == e[2] == 1.5


def test_compact_si_and_narrow_ticks():
    from app.hud_widgets import colorbar_ticks

    assert compact_si(-3540.0) == "-3.5k" and compact_si(1.85e6) == "1.9M" and compact_si(52.4) == "52.4"
    assert compact_si(0.0) == "0" and compact_si(2000.0) == "2k"
    assert len(colorbar_ticks(-3540.0, 3640.0, False, n_linear=3)) == 3
    labels = [t for _, t in colorbar_ticks(-3540.0, 3640.0, False, n_linear=3)]
    assert all(len(t) <= 6 for t in labels)


def test_overlay_fuel_note_is_wrapped(qapp):
    fuel = SustainabilityOverlay(("fuel", "speed"), None)
    assert fuel.fuel_note.wordWrap()
    fuel.fit(340)
    assert fuel.height() >= fuel.sizeHint().height() - 1

