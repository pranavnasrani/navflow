"""Tests for solver.foam.sustainability: drag power, fuel and CO2 saved, wind-load table."""
import pytest

from solver.foam.sustainability import (
    G, drag_power_kw, fuel_and_co2, fuel_burn_l_per_hr, level_flight_drag_n, speed_gain_pct, wind_load_table,
)


def test_drag_power_cubic_in_speed():
    assert drag_power_kw(1.0, 2.0, 1.225, 40) == pytest.approx(78.4)
    assert drag_power_kw(1.0, 2.0, 1.225, 80) == pytest.approx(8 * 78.4)  # V doubles -> P x 8


def test_speed_gain_uses_cda():
    """Equal power: P ~ (Cd A) v^3, so halving Cd A lets v rise by 2^(1/3) - 1 = 25.99 %."""
    assert speed_gain_pct(0.5, 1.0) == pytest.approx(25.99, abs=0.01)
    assert speed_gain_pct(1.0, 1.0) == pytest.approx(0.0)
    assert speed_gain_pct(cda=2.0, ref_cda=16.0) == pytest.approx(100.0)  # (16/2)^(1/3) = 2


def test_fuel_burn_is_power_times_factor():
    assert fuel_burn_l_per_hr(100.0, 0.28) == pytest.approx(28.0)


def test_fuel_and_co2():
    fuel, co2 = fuel_and_co2(power_kw=300.0, ref_power_kw=400.0, fuel_l_per_kwh=0.28, co2_kg_per_l=2.5)
    assert fuel == pytest.approx(28.0)
    assert co2 == pytest.approx(70.0)


def test_fuel_clamped_at_zero_when_power_exceeds_reference():
    assert fuel_and_co2(500.0, 400.0, 0.28, 2.5) == (0.0, 0.0)


def test_wind_load_table_scales_with_v_squared():
    rows = wind_load_table(1000, 5000, 50)
    assert [r["label"] for r in rows][-1].startswith("Category 5")
    assert [r["v_m_s"] for r in rows] == [3.33, 33, 50, 70]
    cat5 = rows[-1]
    assert cat5["force_n"] == pytest.approx(1960)
    assert cat5["moment_nm"] == pytest.approx(9800)
    assert rows[2]["force_n"] == pytest.approx(1000)
    assert set(cat5) == {"label", "v_m_s", "force_n", "moment_nm"}


def test_speed_gain_is_zero_for_non_positive_cd():
    assert speed_gain_pct(0.0, 1.0) == 0.0
    assert speed_gain_pct(-0.2, 1.0) == 0.0
    assert speed_gain_pct(0.5, 0.0) == 0.0


def test_level_flight_drag_is_weight_over_lift_to_drag():
    # 500 t at L/D 20: D = m g / (L/D) = 500000 x 9.81 / 20 = 245 250 N
    assert level_flight_drag_n(500_000, 20.0) == pytest.approx(500_000 * G / 20.0)
    assert level_flight_drag_n(500_000, 10.0) == pytest.approx(2 * level_flight_drag_n(500_000, 20.0))  # half the ratio, twice the drag
    assert level_flight_drag_n(500_000, 0.0) is None and level_flight_drag_n(500_000, -1.0) is None
