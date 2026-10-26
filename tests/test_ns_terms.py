"""Tests for solver.foam.ns_terms: the three terms of the momentum equation from a flow field."""
import numpy as np
import pyvista as pv
import pytest

from solver.foam.ns_terms import ns_term_fields, ns_term_summary


def analytic_grid(n=21):
    """Structured grid on [1,2]^3 with v = (x^2, x*y, 0) and p = x + 2y + 3z."""
    g = pv.ImageData(dimensions=(n, n, n), spacing=(1 / (n - 1),) * 3, origin=(1.0, 1.0, 1.0))
    x, y, z = g.points.T
    g.point_data["U"] = np.column_stack([x**2, x * y, np.zeros_like(x)])
    g.point_data["p"] = x + 2 * y + 3 * z
    return g


def interior(g, margin=0.15):
    pts = g.points
    return np.all((pts > 1 + margin) & (pts < 2 - margin), axis=1)


@pytest.mark.parametrize("unstructured", [False, True], ids=["image_data", "unstructured_grid"])
def test_terms_on_analytic_field(unstructured):
    """OpenFOAM results are unstructured grids, so the same field is also checked cast to one."""
    nu = 0.1
    grid = analytic_grid().cast_to_unstructured_grid() if unstructured else analytic_grid()
    g = ns_term_fields(grid, nu, "U", "p")
    x, y, _ = g.points.T
    mask = interior(g)
    # (v . grad)v = (x^2 * 2x + xy * 0, x^2 * y + xy * x, 0) = (2x^3, 2x^2 y, 0)
    conv = np.hypot(2 * x**3, 2 * x**2 * y)
    np.testing.assert_allclose(g.point_data["ns_convective"][mask], conv[mask], rtol=0.05)
    # grad p = (1, 2, 3), |grad p| = sqrt(14)
    np.testing.assert_allclose(g.point_data["ns_pressure"][mask], np.sqrt(14), rtol=0.05)
    # nu * laplacian(v) = 0.1 * (2, 0, 0)
    np.testing.assert_allclose(g.point_data["ns_viscous"][mask], 0.1 * 2.0, rtol=0.05)


def test_dominant_index():
    g = ns_term_fields(analytic_grid(), 0.1, "U", "p")
    mask = interior(g)
    stack = np.column_stack([g.point_data[k] for k in ("ns_convective", "ns_pressure", "ns_viscous")])
    np.testing.assert_array_equal(g.point_data["ns_dominant"], stack.argmax(axis=1))
    assert g.point_data["ns_dominant"].dtype == np.int8
    # at x ~ 1.5 the convective term (2x^3 ~ 6.75) beats sqrt(14) ~ 3.74 and nu*2 = 0.2
    assert set(np.unique(g.point_data["ns_dominant"][mask])) <= {0, 1}
    # a huge viscosity makes the viscous term win everywhere in the interior
    gv = ns_term_fields(analytic_grid(), 1000.0, "U", "p")
    assert np.all(gv.point_data["ns_dominant"][mask] == 2)


def test_summary_is_mean_inside_bounds():
    g = ns_term_fields(analytic_grid(), 0.1, "U", "p")
    s = ns_term_summary(g, (1.0, 1.5, 1.0, 2.0, 1.0, 2.0))
    assert set(s) == {"convective", "pressure", "viscous"}
    inside = np.all((g.points >= [1.0, 1.0, 1.0]) & (g.points <= [1.5, 2.0, 2.0]), axis=1)
    assert s["pressure"] == pytest.approx(g.point_data["ns_pressure"][inside].mean())
    assert all(isinstance(v, float) for v in s.values())


@pytest.mark.parametrize("cell_only", [False, True], ids=["point_nut", "cell_nut"])
def test_viscous_term_uses_effective_viscosity(cell_only):
    """nu_eff = nu + nut: a constant nut scales the viscous term by (nu + nut) / nu."""
    nu, nut = 0.1, 0.4
    base = ns_term_fields(analytic_grid(), nu, "U", "p")
    g = analytic_grid()
    if cell_only:
        g.cell_data["nut"] = np.full(g.n_cells, nut)
    else:
        g.point_data["nut"] = np.full(g.n_points, nut)
    eff = ns_term_fields(g, nu, "U", "p")
    mask = interior(g)
    np.testing.assert_allclose(
        eff.point_data["ns_viscous"][mask], base.point_data["ns_viscous"][mask] * (nu + nut) / nu, rtol=1e-6
    )
    # the other two terms do not depend on viscosity
    np.testing.assert_allclose(eff.point_data["ns_convective"], base.point_data["ns_convective"])
    np.testing.assert_allclose(eff.point_data["ns_pressure"], base.point_data["ns_pressure"])
    # naming a missing array (or None) falls back to the molecular nu
    for name in ("absent", None):
        same = ns_term_fields(analytic_grid(), nu, "U", "p", nut_name=name)
        np.testing.assert_allclose(same.point_data["ns_viscous"], base.point_data["ns_viscous"])
