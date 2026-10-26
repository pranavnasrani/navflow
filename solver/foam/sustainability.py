"""Sustainability numbers derived from the drag of a shape: power, fuel, CO2, speed, wind load.

Maths link (IGCSE 0606):
  * Drag power P = F v with F = 1/2 rho Cd A v^2, so P = 1/2 rho Cd A v^3: power grows with the
    CUBE of speed (double the speed, eight times the power).
  * Holding power fixed, a shape with a smaller Cd can go faster: v2/v1 = (Cd1/Cd2)^(1/3).
    That is a cube root, i.e. the power 1/3 - an index (surd) calculation.
  * Wind load scales with v^2, so a wind of speed v loads a structure (v / v_sim)^2 times as
    much as the simulated wind: a direct-proportion rule with a squared ratio.
"""
from __future__ import annotations

# Reference winds for the building's wind-load table: (label, speed in m/s).
WIND_LOAD_WINDS = (
    ("Breeze", 3.33),
    ("Category 1 hurricane", 33),
    ("Category 3 hurricane", 50),
    ("Category 5 hurricane", 70),
)


def drag_power_kw(cd: float, area: float, rho: float, v: float) -> float:
    """P = 1/2 rho Cd A v^3 in kilowatts (inputs in kg/m^3, m^2, m/s)."""
    return 0.5 * rho * cd * area * v**3 / 1000.0


def fuel_and_co2(
    power_kw: float, ref_power_kw: float, fuel_l_per_kwh: float, co2_kg_per_l: float
) -> tuple[float, float]:
    """(fuel saved in litres/hour, CO2 saved in kg/hour) against a reference drag power.

    Saved power (ref - actual) x fuel burned per kWh = litres per hour; x kg CO2 per litre = kg per hour.
    Never negative: a shape worse than the reference saves nothing.
    """
    fuel = max(0.0, (ref_power_kw - power_kw) * fuel_l_per_kwh)
    return fuel, fuel * co2_kg_per_l


def speed_gain_pct(cd: float, ref_cd: float) -> float:
    """Extra top speed at equal power, in percent: ((ref_cd / cd)^(1/3) - 1) x 100.

    A drag coefficient that is zero or negative (a noisy or unconverged force) has no meaningful cube
    root ratio, so it reports 0.0 instead of failing.
    """
    if cd <= 0.0 or ref_cd <= 0.0:
        return 0.0
    return ((ref_cd / cd) ** (1.0 / 3.0) - 1.0) * 100.0


def wind_load_table(force_n: float, moment_nm: float, v_sim: float) -> list[dict]:
    """Force and base moment at standard wind speeds, scaled from the simulated wind by (v / v_sim)^2."""
    rows = []
    for label, v in WIND_LOAD_WINDS:
        scale = (v / v_sim) ** 2
        rows.append({"label": label, "v_m_s": v, "force_n": force_n * scale, "moment_nm": moment_nm * scale})
    return rows
