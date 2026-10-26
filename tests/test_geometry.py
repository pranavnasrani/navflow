"""Tests for solver.foam.geometry: model preparation and wind-tunnel geometry."""
import numpy as np
import pytest
import trimesh

from solver.foam.geometry import (
    ModelNotWatertight,
    frontal_area,
    prepare_model,
    rotation_matrix,
    run_geometry,
    seed_rake,
    tunnel_bounds,
    view_bounds,
    wand_heights,
)
from solver.foam.settings import ObjectCfd

FACTORS = {"upstream": 3, "downstream": 8, "lateral": 3, "top": 3}


def make_cfd(**over):
    base = dict(
        obj_id="box", speed_m_s=10.0, real_size_m=10.0, size_axis="x",
        base_rotation_deg=(0.0, 0.0, 0.0), rotation_axis="y",
        directions_deg=(-20, -15, -10, -5, 0, 5, 10, 15, 20), direction_label="angle of attack",
        floor="moving", density_kg_m3=1.225, kinematic_viscosity_m2_s=1.5e-5,
        reference_cd=0.3, cruise=None, abl_z0_m=None, widgets=(), surface_refinement=None,
    )
    base.update(over)
    return ObjectCfd(**base)


def stl_box(lx, ly, lz, centre=(0.0, 0.0, 0.0)):
    return trimesh.creation.box(extents=(lx, ly, lz)).apply_translation(centre)


def save(mesh, tmp_path, name="m.stl"):
    path = tmp_path / name
    mesh.export(path)
    return path


@pytest.fixture
def box_cfd_airplane():
    return make_cfd(obj_id="airplane", floor="none")


def test_rotation_matrix_right_hand():
    assert np.allclose(rotation_matrix("z", 90) @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(rotation_matrix("y", 90) @ [1, 0, 0], [0, 0, -1])
    assert np.allclose(rotation_matrix("x", 90) @ [0, 1, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        rotation_matrix("w", 10)


def test_angle_of_attack_lifts_nose(box_cfd_airplane, tmp_path):   # nose = xmin end
    m0 = prepare_model(save(stl_box(4, 1, 1), tmp_path), box_cfd_airplane)
    def nose_z(m):   # mean height of the forward (x < 0) half of the body
        return m.vertices[m.vertices[:, 0] < 0, 2].mean()

    nose0 = nose_z(m0)
    m10 = run_geometry(m0, box_cfd_airplane, 10.0)
    assert nose_z(m10) > nose0
    assert nose_z(m0) == pytest.approx(nose0)   # the original is untouched (run_geometry returns a copy)


def test_prepare_model_scales_and_places_on_floor(tmp_path):
    m = prepare_model(save(stl_box(2, 1, 1, centre=(5, -3, 7)), tmp_path), make_cfd())
    assert np.allclose(m.extents, (10, 5, 5))
    lo, hi = m.bounds
    assert lo[2] == pytest.approx(0.0, abs=1e-9)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-9)
    assert (lo[1] + hi[1]) / 2 == pytest.approx(0.0, abs=1e-9)


def test_prepare_model_centres_z_in_free_air(tmp_path):
    m = prepare_model(save(stl_box(2, 1, 1, centre=(0, 0, 7)), tmp_path), make_cfd(floor="none"))
    lo, hi = m.bounds
    assert (lo[2] + hi[2]) / 2 == pytest.approx(0.0, abs=1e-9)


def test_prepare_model_size_axis_z(tmp_path):
    m = prepare_model(save(stl_box(1, 1, 4), tmp_path), make_cfd(size_axis="z", real_size_m=40.0))
    assert m.extents[2] == pytest.approx(40.0)


def test_prepare_model_applies_base_rotation(tmp_path):
    # A long-in-z bar rotated 90 deg about y (base rotation) becomes long in x (z -> x via Ry(90): z -> +x).
    cfd = make_cfd(base_rotation_deg=(0.0, 90.0, 0.0), real_size_m=8.0)
    m = prepare_model(save(stl_box(1, 1, 4), tmp_path), cfd)
    assert np.allclose(m.extents, (8, 2, 2))


def test_prepare_model_unions_overlapping_parts(tmp_path):
    parts = trimesh.util.concatenate([stl_box(2, 1, 1), stl_box(1, 2, 1, centre=(0.5, 0, 0))])
    assert not parts.is_watertight or parts.body_count > 1
    m = prepare_model(save(parts, tmp_path), make_cfd())
    assert m.is_watertight and m.body_count == 1


def test_prepare_model_rejects_open_surface(tmp_path):
    box = stl_box(2, 1, 1)
    open_box = trimesh.Trimesh(box.vertices, box.faces[2:], process=False)   # delete one quad (2 triangles)
    with pytest.raises(ModelNotWatertight) as err:
        prepare_model(save(open_box, tmp_path), make_cfd())
    assert "hole" in str(err.value).lower()


def test_tunnel_bounds_floor_and_free_air():
    cube = trimesh.creation.box(extents=(1, 1, 1))
    on_floor = cube.copy().apply_translation((0, 0, 0.5))
    b = tunnel_bounds(on_floor, FACTORS, "moving")
    assert b == pytest.approx((-3.5, 8.5, -3.5, 3.5, 0.0, 4.0))
    b = tunnel_bounds(cube, FACTORS, "none")
    assert b[4] == pytest.approx(-3.5) and b[5] == pytest.approx(3.5)
    assert (b[0], b[1]) == pytest.approx((-3.5, 8.5))


def test_frontal_area_of_box():
    box = trimesh.creation.box(extents=(4, 2, 1))
    assert frontal_area(box) == pytest.approx(2.0, rel=0.02)


def test_view_bounds():
    m = trimesh.creation.box(extents=(4, 2, 1)).apply_translation((0, 0, 0.5))   # z 0..1
    b = view_bounds(m, "moving")
    assert b == pytest.approx((-2 - 4, 2 + 12, -1 - 2, 1 + 2, 0.0, 1 + 0.5))
    b = view_bounds(m, "none")
    assert b[4] == pytest.approx(-0.5) and b[5] == pytest.approx(1.5)


def test_wand_and_rake_shapes():
    m = trimesh.creation.box(extents=(4, 2, 1))
    z = wand_heights(m, 10)
    assert z.shape == (10,) and np.all(np.diff(z) > 0)
    assert z[0] == pytest.approx(-0.5 - 0.1) and z[-1] == pytest.approx(0.5 + 0.1)
    rake = seed_rake(m, z[3], 24)
    assert rake.shape == (24, 3)
    assert np.all(rake[:, 0] < m.bounds[0][0])
    assert np.all(rake[:, 2] == pytest.approx(z[3]))
    assert rake[0, 1] == pytest.approx(-1 - 0.2) and rake[-1, 1] == pytest.approx(1 + 0.2)
