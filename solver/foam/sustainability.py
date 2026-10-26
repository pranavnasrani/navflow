"""Sustainability numbers derived from the drag of a shape: power, fuel, CO2, speed, wind load.

Maths link (IGCSE 0606):
  * Drag power P = F v with F = 1/2 rho Cd A v^2, so P = 1/2 rho Cd A v^3: power grows with the
    CUBE of speed (double the speed, eight times the power).
  * Holding power fixed, a shape with a smaller Cd*A can go faster. P is proportional to (Cd A) v^3, so
    equal P means (Cd A)_1 v_1^3 = (Cd A)_2 v_2^3, giving v_2/v_1 = ((Cd A)_1 / (Cd A)_2)^(1/3).
    That is a cube root, i.e. the power 1/3 - an index (surd) calculation.
  * The comparison shape is the object's own WORST simulated wind direction (its largest drag power),
    so "saved" is a difference of two powers, (P_worst - P) x fuel per kWh, and is zero at the worst angle.
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


def fuel_burn_l_per_hr(power_kw: float, fuel_l_per_kwh: float) -> float:
    """Fuel burned just to overcome drag: power (kW) x fuel per kWh = litres per hour."""
    return power_kw * fuel_l_per_kwh


def speed_gain_pct(cda: float, ref_cda: float) -> float:
    """Extra top speed at equal power, in percent: ((ref_cda / cda)^(1/3) - 1) x 100.

    `cda` is the drag coefficient times the frontal area (m^2). At equal power P = 1/2 rho (Cd A) v^3,
    so v is proportional to (Cd A)^(-1/3): the speed ratio is the cube root of the (Cd A) ratio.

    A Cd A that is zero or negative (a noisy or unconverged force) has no meaningful cube root ratio,
    so it reports 0.0 instead of failing.
    """
    if cda <= 0.0 or ref_cda <= 0.0:
        return 0.0
    return ((ref_cda / cda) ** (1.0 / 3.0) - 1.0) * 100.0


def wind_load_table(force_n: float, moment_nm: float, v_sim: float) -> list[dict]:
    """Force and base moment at standard wind speeds, scaled from the simulated wind by (v / v_sim)^2."""
    rows = []
    for label, v in WIND_LOAD_WINDS:
        scale = (v / v_sim) ** 2
        rows.append({"label": label, "v_m_s": v, "force_n": force_n * scale, "moment_nm": moment_nm * scale})
    return rows
