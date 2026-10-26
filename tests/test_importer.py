"""Tests for solver.foam.importer and app.cache_format: OpenFOAM results -> v2 cache."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyvista as pv
import pytest
import trimesh

from app.cache_format import FORMAT_VERSION, read_cache, state_key, wand_key, write_cache
from solver.foam.geometry import frontal_area, rotation_matrix, run_geometry, wand_heights
from solver.foam.importer import import_object
from solver.foam.settings import ObjectCfd, load_settings
from solver.foam.sustainability import drag_power_kw, fuel_and_co2, speed_gain_pct
from tests.fake_cache import write_fake_cache

ROOT = Path(__file__).resolve().parent.parent
RHO = 1.225
SPEED = 20.0
DIRS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0)  # idx 0 = 0 deg, idx 4 = 40 deg

OBJ_CONFIG = {
    "id": "box", "display_name": "Test Box",
    "fuel_burn_factor_l_per_kwh": 0.28, "co2_kg_per_l_fuel": 2.52,
    "cfd": {"widgets": ["fuel", "speed"]},
}


def make_settings():
    settings, _, _ = load_settings(ROOT / "config" / "exhibition_config.json")
    return replace(settings, wand_count=3, seeds_per_wand=4)


def make_cfd(**over):
    base = dict(
        obj_id="box", speed_m_s=SPEED, real_size_m=4.0, size_axis="x",
        base_rotation_deg=(0.0, 0.0, 0.0), rotation_axis="y", directions_deg=DIRS,
        direction_label="angle", floor="none", density_kg_m3=RHO, kinematic_viscosity_m2_s=1.5e-5,
        reference_cd=1.0, cruise=None, abl_z0_m=None, widgets=("fuel", "speed"), surface_refinement=None,
    )
    base.update(over)
    return ObjectCfd(**base)


def make_model0():
    """A 4 x 2 x 2 box cut into small triangles, so nearest-cell lookups are accurate to ~0.2 m."""
    return trimesh.creation.box(extents=(4.0, 2.0, 2.0)).subdivide_to_size(0.3)


def force_text(rows):
    head = "# Force\n# CofR : (0 0 0)\n#\n# Time total_x total_y total_z pressure_x pressure_y pressure_z viscous_x viscous_y viscous_z\n"
    return head + "".join(f"{i} {fx} {fy} {fz} 0 0 0 0 0 0\n" for i, fx, fy, fz in rows)


def write_fake_result(
    root, cfd, model0, idx, *, status="done", fields="instant", forces=None, moments=None,
    p_inst=1.0, p_mean=3.0, cell_only=False,
):
    """Write runs/<root>/<obj>/<run>/result/ in the layout the real pipeline produces."""
    deg = cfd.directions_deg[idx]
    run_id = f"{cfd.obj_id}_d{idx}"
    result = Path(root) / cfd.obj_id / run_id / "result"
    result.mkdir(parents=True)
    (result / "run.json").write_text(json.dumps({
        "run_id": run_id, "obj_id": cfd.obj_id, "direction_index": idx, "direction_deg": deg,
        "speed_m_s": SPEED, "iterations": 300, "converged": status == "done", "fields": fields, "status": status,
    }))
    forces = forces if forces is not None else [(i, 100.0, 5.0, 20.0) for i in range(1, 301)]
    moments = moments if moments is not None else [(i, 30.0, 40.0, 1.0) for i in range(1, 301)]
    pp = result / "postProcessing" / "forces" / "0"
    pp.mkdir(parents=True)
    (pp / "force.dat").write_text(force_text(forces))
    (pp / "moment.dat").write_text(force_text(moments))

    def write_time(time, p_scale, k_offset):
        vdir = result / "VTK" / f"{run_id}_{time}"
        (vdir / "boundary").mkdir(parents=True)
        grid = pv.ImageData(dimensions=(41, 25, 25), spacing=(1, 1, 1), origin=(-12, -12, -12)).cast_to_unstructured_grid()
        for suffix, p, ko in (("", p_inst * p_scale, k_offset), ("Mean", p_mean * p_scale, k_offset + 100)):
            x, y, _ = grid.points.T
            grid.point_data["U" + suffix] = np.column_stack([SPEED + 0.1 * y, 0 * x, 0 * x])
            grid.point_data["p" + suffix] = np.full(len(x), p)
            grid.point_data["k" + suffix] = x + ko
        grid.point_data["nut"] = np.full(grid.n_points, 0.5)  # a mean-field run still writes plain nut
        if cell_only:  # an export with cell data only: same linear fields, sampled at the cell centres
            centres = grid.cell_centers().points
            cx, cy = centres[:, 0], centres[:, 1]
            grid.point_data.clear()
            for suffix, p, ko in (("", p_inst * p_scale, k_offset), ("Mean", p_mean * p_scale, k_offset + 100)):
                grid.cell_data["U" + suffix] = np.column_stack([SPEED + 0.1 * cy, 0 * cx, 0 * cx])
                grid.cell_data["p" + suffix] = np.full(len(cx), p)
                grid.cell_data["k" + suffix] = cx + ko
            grid.cell_data["nut"] = np.full(grid.n_cells, 0.5)
        grid.save(vdir / "internal.vtu")
        run_model = run_geometry(model0, cfd, deg)
        patch = pv.PolyData.from_regular_faces(np.asarray(run_model.vertices), np.asarray(run_model.faces))
        r = rotation_matrix(cfd.rotation_axis, deg)
        centres_model = patch.cell_centers().points @ r  # back to the model frame
        for suffix, scale in (("", p_scale * p_inst), ("Mean", p_scale * p_mean)):
            # cell pressure = model-frame x (times scale): lets a test check the rotation bookkeeping
            patch.cell_data["p" + suffix] = centres_model[:, 0] * scale
        patch.save(vdir / "boundary" / "model.vtp")

    write_time(50, 999.0, 0.0)  # an older save: must be ignored
    write_time(300, 1.0, 100.0)
    return result


@pytest.fixture
def root(tmp_path):
    return tmp_path / "runs"


def run_import(root, tmp_path, cfd=None, obj_config=OBJ_CONFIG, model0=None):
    cfd = cfd or make_cfd()
    model0 = model0 or make_model0()
    out = tmp_path / "cache" / "box_states.npz"
    report = import_object("box", make_settings(), cfd, model0, obj_config, root, out)
    return report, out, model0, cfd


def test_import_fake_result(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 0)
    write_fake_result(root, cfd, model0, 4)
    report, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)

    assert report["imported"] == [0, 4]
    assert report["flagged"] == []
    assert report["filled"] == {1: 0, 2: 0, 3: 4, 5: 4, 6: 4, 7: 4, 8: 4}  # nearest direction; ties -> lower

    c = read_cache(out)
    assert c["format_version"] == FORMAT_VERSION == 2
    assert c["mesh_vertices"].shape == (len(model0.vertices), 3) and c["mesh_vertices"].dtype == np.float32
    assert c["mesh_faces"].dtype == np.int32
    assert c["directions_deg"].tolist() == list(DIRS)
    assert c["wand_heights"].shape == (3,)
    assert c["view_bounds"].shape == (6,)
    assert json.loads(c["config_json"])["id"] == "box"
    np.testing.assert_allclose(c["wind_vectors"][0], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(c["wind_vectors"][4], rotation_matrix("y", 40.0).T @ [1, 0, 0], atol=1e-9)

    for d in range(9):
        s = state_key(d)
        assert c[f"{s}_pressure"].shape == (len(model0.vertices),) and c[f"{s}_pressure"].dtype == np.float32
        terms = json.loads(c[f"{s}_ns_terms"])
        assert set(terms) == {"convective", "pressure", "viscous"}
        for w in range(3):
            key = wand_key(d, w)
            lines, k, dom = c[f"{key}_lines"], c[f"{key}_k"], c[f"{key}_dominant"]
            assert lines.shape[1] == 3 and lines.dtype == np.float32
            assert len(k) == len(dom) == len(lines) and k.dtype == np.float32 and dom.dtype == np.int8
            sep = np.isnan(lines).any(axis=1)
            assert sep.any() and (~sep).sum() >= 8  # NaN separators and real points
            assert np.array_equal(sep, np.isnan(k)) and np.array_equal(sep, dom == -1)
            assert set(dom[~sep].tolist()) <= {0, 1, 2}
    # filled directions copy the source arrays and say where from
    np.testing.assert_array_equal(c["d2_pressure"], c["d0_pressure"])
    np.testing.assert_array_equal(c["d3_w0_lines"], c["d4_w0_lines"])
    m0, m2, m3 = (json.loads(c[f"d{i}_metrics"]) for i in (0, 2, 3))
    assert m0["filled_from"] is None and m2["filled_from"] == 0 and m3["filled_from"] == 4
    # a filled slot reports its own angle, while the numbers come from the source run
    assert (m0["direction_deg"], m2["direction_deg"], m3["direction_deg"]) == (0.0, 20.0, 30.0)
    assert m2["drag_force_n"] == m0["drag_force_n"]


def test_metrics_values(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 4)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    m = json.loads(read_cache(out)["d4_metrics"])
    area = frontal_area(run_geometry(model0, cfd, 40.0))
    q_a = 0.5 * RHO * SPEED**2 * area
    assert m["direction_deg"] == 40.0 and m["wind_speed_m_s"] == SPEED
    assert m["reynolds_number"] == pytest.approx(SPEED * 4.0 / 1.5e-5)
    assert (m["drag_force_n"], m["side_force_n"], m["lift_force_n"]) == (100.0, 5.0, 20.0)
    assert m["frontal_area_m2"] == pytest.approx(area)
    assert m["drag_coefficient_cd"] == pytest.approx(100.0 / q_a)
    assert m["lift_coefficient_cl"] == pytest.approx(20.0 / q_a)
    assert m["drag_power_kw"] == pytest.approx(drag_power_kw(m["drag_coefficient_cd"], area, RHO, SPEED))
    assert m["drag_power_kw"] == pytest.approx(100.0 * SPEED / 1000)  # P = F v
    assert m["base_moment_nm"] == pytest.approx(50.0)  # hypot(30, 40)
    assert (m["iterations"], m["converged"], m["filled_from"]) == (300, True, None)
    ref = drag_power_kw(1.0, area, RHO, SPEED)
    fuel, co2 = fuel_and_co2(m["drag_power_kw"], ref, 0.28, 2.52)
    assert m["fuel_saved_l_per_hr"] == pytest.approx(fuel) and m["co2_saved_kg_per_hr"] == pytest.approx(co2)
    assert m["speed_gain_pct"] == pytest.approx(speed_gain_pct(m["drag_coefficient_cd"], 1.0))
    assert "wind_load" not in m
    assert all(isinstance(v, (int, float, bool, type(None))) for k, v in m.items())


def test_cruise_conditions_used_for_fuel(root, tmp_path):
    cfd = make_cfd(cruise={"density_kg_m3": 0.364, "speed_m_s": 250.0})
    model0 = make_model0()
    write_fake_result(root, cfd, model0, 4)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    m = json.loads(read_cache(out)["d4_metrics"])
    cd, area = m["drag_coefficient_cd"], m["frontal_area_m2"]
    power = drag_power_kw(cd, area, 0.364, 250.0)
    fuel, co2 = fuel_and_co2(power, drag_power_kw(1.0, area, 0.364, 250.0), 0.28, 2.52)
    assert m["fuel_saved_l_per_hr"] == pytest.approx(fuel)
    assert m["co2_saved_kg_per_hr"] == pytest.approx(co2)


def test_wind_load_table_for_wind_load_objects(root, tmp_path):
    cfd = make_cfd(widgets=("wind_load",))
    model0 = make_model0()
    write_fake_result(root, cfd, model0, 4)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    m = json.loads(read_cache(out)["d4_metrics"])
    cat5 = m["wind_load"][-1]
    assert cat5["v_m_s"] == 70
    assert cat5["force_n"] == pytest.approx(100.0 * (70 / SPEED) ** 2)
    assert cat5["moment_nm"] == pytest.approx(50.0 * (70 / SPEED) ** 2)


def test_pressure_follows_rotation(root, tmp_path):
    """Patch cell pressure = model-frame x, so after the R(theta) bookkeeping each vertex sees its own x."""
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 4, p_inst=1.0)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    c = read_cache(out)
    p_kinematic = c["d4_pressure"] / RHO
    assert np.abs(p_kinematic - model0.vertices[:, 0]).max() < 0.5


def test_lines_are_back_in_model_frame_and_k_is_sampled(root, tmp_path):
    """k = x_run + 100 in the file; a streamline point in the model frame maps back through R."""
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 4)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    c = read_cache(out)
    lines, k = c["d4_w1_lines"], c["d4_w1_k"]
    ok = ~np.isnan(k)
    assert ok.sum() > 8
    x_run = (lines[ok].astype(float) @ rotation_matrix("y", 40.0).T)[:, 0]
    np.testing.assert_allclose(k[ok], x_run + 100.0, atol=0.05)
    # the middle wand rake sits at the model's mid-height in the model frame, so seeds start near z0
    z0 = wand_heights(model0, 3)[1]
    first = lines[~np.isnan(lines).any(axis=1)][0]
    assert abs(first[2] - z0) < 3.0  # the rake is tilted by the 40 degree rotation, so allow slack


def test_import_with_no_results_raises(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    out = tmp_path / "cache" / "box_states.npz"
    with pytest.raises(ValueError, match="No finished runs for box"):
        import_object("box", make_settings(), cfd, model0, OBJ_CONFIG, root, out)
    assert not out.exists()


def test_cell_only_internal_mesh_is_accepted(root, tmp_path):
    """U, p, k and nut that exist only as cell data are averaged onto the points (no KeyError)."""
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 4, cell_only=True)
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    c = read_cache(out)
    lines, k = c["d4_w1_lines"], c["d4_w1_k"]
    ok = ~np.isnan(k)
    assert ok.sum() > 8
    x_run = (lines[ok].astype(float) @ rotation_matrix("y", 40.0).T)[:, 0]
    np.testing.assert_allclose(k[ok], x_run + 100.0, atol=1.0)  # cell-to-point averaging blurs by < 1 cell
    assert set(json.loads(c["d4_ns_terms"])) == {"convective", "pressure", "viscous"}


def test_failed_and_unfinished_runs_are_not_finished(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    write_fake_result(root, cfd, model0, 2, status="failed")
    stub = root / "box" / "box_d6" / "result"  # a result folder with no run.json at all
    stub.mkdir(parents=True)
    out = tmp_path / "cache" / "box_states.npz"
    with pytest.raises(ValueError, match="No finished runs for box"):
        import_object("box", make_settings(), cfd, model0, OBJ_CONFIG, root, out)
    assert not out.exists()


def test_flagged_run_uses_mean_fields(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    forces = [(i, 100.0 + i, 5.0, 20.0) for i in range(1, 151)]
    write_fake_result(root, cfd, model0, 4, status="flagged", fields="mean", forces=forces)
    report, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    assert report["flagged"] == [4] and report["imported"] == [4]
    c = read_cache(out)
    # pMean patch = model x * p_mean (3) -> vertex pressure ~ 3 * x * rho, not 1 * x * rho
    ratio = (c["d4_pressure"] / RHO) / np.where(model0.vertices[:, 0] == 0, np.nan, model0.vertices[:, 0])
    assert np.nanmedian(ratio) == pytest.approx(3.0, rel=0.2)
    lines_k = c["d4_w1_k"]
    assert np.nanmin(lines_k) > 150.0  # kMean = x + 200 (+100 mean offset): the 'k' set would be ~100
    m = json.loads(c["d4_metrics"])
    assert m["drag_force_n"] == pytest.approx(np.mean([100.0 + i for i in range(51, 151)]))  # mean of last 100 rows
    assert m["converged"] is False


def test_force_history_across_restart_dirs(root, tmp_path):
    cfd, model0 = make_cfd(), make_model0()
    result = write_fake_result(root, cfd, model0, 4, forces=[(i, 100.0, 5.0, 20.0) for i in range(1, 151)])
    restart = result / "postProcessing" / "forces" / "100"
    restart.mkdir()
    (restart / "force.dat").write_text(force_text([(i, 777.0, 5.0, 20.0) for i in range(101, 201)]))
    (restart / "moment.dat").write_text(force_text([(i, 30.0, 40.0, 1.0) for i in range(101, 201)]))
    _, out, _, _ = run_import(root, tmp_path, cfd, model0=model0)
    assert json.loads(read_cache(out)["d4_metrics"])["drag_force_n"] == 777.0  # last row of the later segment


def test_fake_cache_is_a_valid_v2_cache(tmp_path):
    cfg = {"id": "box", "cfd": {"widgets": ["wind_load"]}}
    path = tmp_path / "fake.npz"
    write_fake_cache(path, cfg, n_dir=5, n_wand=4)
    c = read_cache(path)
    assert c["directions_deg"].shape == (5,) and c["wand_heights"].shape == (4,)
    for d in range(5):
        assert np.all(c[f"d{d}_pressure"] == d)  # constant d: interpolation is checkable
        for w in range(4):
            lines = c[f"d{d}_w{w}_lines"]
            real = lines[~np.isnan(lines).any(axis=1)]
            assert np.isnan(lines).any() and len(real) > 1
            assert np.allclose(real[:, 2], real[0, 2])  # straight, at wand height
        assert "wind_load" in json.loads(c[f"d{d}_metrics"])


def test_read_cache_rejects_other_versions(tmp_path):
    old = tmp_path / "old.npz"
    np.savez_compressed(old, z_steps=np.arange(3))  # a v1-style cache has no format_version
    with pytest.raises(ValueError, match="format"):
        read_cache(old)
    wrong = tmp_path / "wrong.npz"
    np.savez_compressed(wrong, format_version=np.array(1))
    with pytest.raises(ValueError, match="format"):
        read_cache(wrong)


def test_write_cache_roundtrip(tmp_path):
    path = tmp_path / "c.npz"
    write_cache(path, {"format_version": 2, "a": np.arange(3, dtype=np.float32), "config_json": json.dumps({"x": 1})})
    c = read_cache(path)
    assert c["format_version"] == 2 and c["a"].dtype == np.float32
    assert json.loads(c["config_json"]) == {"x": 1}
