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
    line_runs,
    log_bar_value,
    particles_on_lines,
    polyline_from_nan_joined,
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
    fuel.set_metrics({"fuel_saved_l_per_hr": 123.4, "co2_saved_kg_per_hr": 311.0, "speed_gain_pct": 8.2})
    text = fuel.all_text()
    assert "L/h" in text and "SDG 13" in text and "kg/h" in text and "+8.2 %" in text

    tower = SustainabilityOverlay(("wind_load",), None)
    tower.set_metrics({"wind_load": _wind_load(), "direction_deg": 15.0})
    text = tower.all_text()
    assert "Category 5" in text and "SDG 9" in text and "Worst direction so far" in text


def test_overlay_worst_direction_tracks_the_peak(qapp):
    tower = SustainabilityOverlay(("wind_load",), None)
    for deg, scale in ((0.0, 1.0), (30.0, 3.0), (60.0, 2.0)):
        tower.set_metrics({"wind_load": _wind_load(scale), "direction_deg": deg})
    assert "30.0°" in tower.worst_label.text()
    # a second call with the same data does not grow the table
    n_cells = len(tower._table_cells)
    tower.set_metrics({"wind_load": _wind_load(1.0), "direction_deg": 10.0})
    assert len(tower._table_cells) == n_cells == 2
    assert "30.0°" in tower.worst_label.text()


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
