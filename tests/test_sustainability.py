"""Tests for solver.foam.sustainability: drag power, fuel and CO2 saved, wind-load table."""
import pytest

from solver.foam.sustainability import drag_power_kw, fuel_and_co2, speed_gain_pct, wind_load_table


def test_drag_power_cubic_in_speed():
    assert drag_power_kw(1.0, 2.0, 1.225, 40) == pytest.approx(78.4)
    assert drag_power_kw(1.0, 2.0, 1.225, 80) == pytest.approx(8 * 78.4)  # V doubles -> P x 8


def test_speed_gain_is_cube_root_of_cd_ratio():
    assert speed_gain_pct(0.5, 1.0) == pytest.approx(25.99, abs=0.01)
    assert speed_gain_pct(1.0, 1.0) == pytest.approx(0.0)


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
