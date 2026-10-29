"""Tests for solver.foam.settings: the typed view of config/exhibition_config.json."""
import json
from pathlib import Path

import pytest

from solver.foam.settings import load_settings

CONFIG = Path(__file__).resolve().parent.parent / "config" / "exhibition_config.json"


def _write_modified(tmp_path, mutate):
    """Copy the real config into tmp_path after letting `mutate(raw)` edit it."""
    raw = json.loads(CONFIG.read_text())
    mutate(raw)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return path


def _cfd(raw, obj_id):
    return next(o for o in raw["objects"] if o["id"] == obj_id)["cfd"]


def test_load_settings_values():
    s, objs, raw = load_settings(CONFIG)
    assert s.image == "opencfd/openfoam-default:2512" and s.volume == "navflow-cases"
    assert (s.n_procs, s.max_iterations, s.preview_iterations, s.min_iterations) == (4, 1000, 500, 200)
    assert (s.convergence_window, s.convergence_tolerance, s.write_interval, s.average_window) == (50, 0.002, 50, 100)
    assert s.wand_count == 10
    assert [objs[k].speed_m_s for k in ("airplane", "f1_car", "building", "human")] == [102.0, 40.0, 50.0, 10.0]
    assert all(len(o.directions_deg) == 9 for o in objs.values())
    assert objs["f1_car"].floor == "moving" and objs["building"].floor == "rough" and objs["airplane"].floor == "none"
    assert objs["airplane"].cruise == {"density_kg_m3": 0.364, "speed_m_s": 250.0, "altitude_km": 11, "mass_kg": 500000}
    assert objs["airplane"].widgets == ("fuel",) and objs["building"].widgets == ("wind_load",)


def test_load_settings_meshing_and_object_details():
    s, objs, raw = load_settings(CONFIG)
    assert s.domain_factors == {"upstream": 3, "downstream": 8, "lateral": 3, "top": 3}
    assert (s.base_cells_per_size, s.surface_refinement, s.wake_refinement, s.n_layers) == (6, (4, 5), 2, 3)
    assert s.seeds_per_wand == 24
    assert objs["human"].widgets == ()
    assert objs["human"].cruise is None and objs["airplane"].abl_z0_m is None
    assert objs["building"].abl_z0_m == 0.5
    assert objs["airplane"].surface_refinement == (6, 7)  # finer surface cells for the thin wings
    assert all(o.surface_refinement is None for k, o in objs.items() if k != "airplane")
    assert objs["airplane"].directions_deg == (0, 2, 4, 6, 8, 10, 12, 14, 16)
    assert objs["building"].directions_deg[1] == 7.5 and objs["human"].directions_deg[-1] == 180
    assert (objs["airplane"].real_size_m, objs["airplane"].size_axis) == (72.7, "x")
    assert (objs["building"].real_size_m, objs["building"].size_axis) == (828.0, "z")
    assert objs["airplane"].rotation_axis == "y" and objs["f1_car"].rotation_axis == "z"
    assert objs["human"].floor == "slip"
    assert all(o.density_kg_m3 == 1.225 and o.kinematic_viscosity_m2_s == 1.5e-5 for o in objs.values())
    # Real models (2026-10-27): A380 nose was +y, RB22 nose was -y, Burj and human upright.
    assert objs["airplane"].base_rotation_deg == (0.0, 0.0, 90.0)
    assert objs["f1_car"].base_rotation_deg == (0.0, 0.0, -90.0)
    assert objs["building"].base_rotation_deg == objs["human"].base_rotation_deg == (0.0, 0.0, 0.0)
    assert objs["f1_car"].real_size_m == 5.4
    assert all(not hasattr(o, "reference_cd") for o in objs.values())
    assert all("reference_cd" not in o["cfd"] for o in raw["objects"])
    assert objs["airplane"].direction_label == "Angle of attack"
    assert "info_card" in next(o for o in raw["objects"] if o["id"] == "building")
    assert raw["controls"]["remote_port"] == 8765 and raw["controls"]["knob_step"] == 0.03125
    assert "wind_tunnel" not in raw and "solver_defaults" not in raw
    assert set(raw["controls"]) == {"remote_port", "knob_step"}  # the keys themselves are hard-coded in input_controller
    assert all("baseline_drag_power_kw" not in o for o in raw["objects"])
    assert "Moving the air jet" not in next(o for o in raw["objects"] if o["id"] == "airplane")["description"]


def test_surface_refinement_override(tmp_path):
    path = _write_modified(tmp_path, lambda r: _cfd(r, "human").update(surface_refinement=[3, 4]))
    _, objs, _ = load_settings(path)
    assert objs["human"].surface_refinement == (3, 4)


def test_load_settings_rejects_unknown_floor(tmp_path):
    path = _write_modified(tmp_path, lambda r: _cfd(r, "f1_car").update(floor="lava"))
    with pytest.raises(ValueError, match="f1_car"):
        load_settings(path)


def test_load_settings_rejects_bad_rotation_axis(tmp_path):
    path = _write_modified(tmp_path, lambda r: _cfd(r, "building").update(rotation_axis="x"))
    with pytest.raises(ValueError, match="building"):
        load_settings(path)


def test_load_settings_rejects_wrong_direction_count(tmp_path):
    path = _write_modified(tmp_path, lambda r: _cfd(r, "human").update(directions_deg=[0, 45, 90]))
    with pytest.raises(ValueError, match="human"):
        load_settings(path)


def test_load_settings_rejects_unsorted_directions(tmp_path):
    path = _write_modified(
        tmp_path, lambda r: _cfd(r, "airplane").update(directions_deg=[0, 2, 4, 6, 8, 8, 12, 14, 16])
    )
    with pytest.raises(ValueError, match="airplane"):
        load_settings(path)
