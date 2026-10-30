"""Tests for solver.foam.case_writer: the automatic OpenFOAM case writer."""
import re
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import trimesh

from solver.foam.case_writer import RunSpec, write_case
from solver.foam.geometry import load_prepared, prepare_model, run_geometry, tunnel_bounds
from solver.foam.settings import load_settings

ROOT = Path(__file__).resolve().parent.parent
SETTINGS, OBJECTS, _RAW = load_settings(ROOT / "config" / "exhibition_config.json")


def box_model(obj_id: str, tmp_path: Path):
    """A 4 x 1 x 1 box, prepared exactly like a real model, then rotated to direction 0."""
    cfd = OBJECTS[obj_id]
    src = tmp_path / "box.stl"
    trimesh.creation.box(extents=(4.0, 1.0, 1.0)).export(src)
    model0 = prepare_model(src, replace(cfd, real_size_m=4.0, size_axis="x", base_rotation_deg=(0.0, 0.0, 0.0)))
    return cfd, run_geometry(model0, cfd, cfd.directions_deg[0])


def write(tmp_path: Path, obj_id: str, max_iterations: int = 1000, sub: str = "run"):
    work = tmp_path / sub
    work.mkdir(exist_ok=True)
    cfd, model = box_model(obj_id, work)
    case = work / "case"
    spec = RunSpec(f"{obj_id}_d0", obj_id, 0, cfd.directions_deg[0], max_iterations)
    write_case(case, model, cfd, SETTINGS, spec)
    return case, cfd, model


def test_run_spec_is_frozen():
    spec = RunSpec("airplane_d3", "airplane", 3, 6.0, 1000)
    with pytest.raises(Exception):
        spec.run_id = "x"


def test_write_case_creates_all_files(tmp_path):
    case, _, _ = write(tmp_path, "f1_car")
    expected = [
        "system/blockMeshDict", "system/snappyHexMeshDict", "system/surfaceFeatureExtractDict",
        "system/controlDict", "system/decomposeParDict", "system/fvSchemes", "system/fvSolution",
        "constant/transportProperties", "constant/turbulenceProperties", "constant/triSurface/model.stl",
        "0/U", "0/p", "0/k", "0/omega", "0/nut",
    ]
    for rel in expected:
        assert (case / rel).is_file(), rel


def test_write_case_overwrites_and_creates_dir(tmp_path):
    case, cfd, model = write(tmp_path, "f1_car", max_iterations=500)
    assert "endTime         500;" in (case / "system/controlDict").read_text()
    spec = RunSpec("f1_car_d0", "f1_car", 0, 0.0, 700)
    write_case(case, model, cfd, SETTINGS, spec)
    assert "endTime         700;" in (case / "system/controlDict").read_text()


def test_stl_solid_name_is_model(tmp_path):
    case, _, model = write(tmp_path, "f1_car")
    text = (case / "constant/triSurface/model.stl").read_text()
    assert text.startswith("solid model")
    assert text.rstrip().endswith("endsolid model")
    back = trimesh.load(case / "constant/triSurface/model.stl")
    assert back.is_watertight
    assert np.allclose(back.bounds, model.bounds, atol=1e-5)


def test_control_dict_values(tmp_path):
    case, _, _ = write(tmp_path, "f1_car")
    text = (case / "system/controlDict").read_text()
    for pattern in (
        r"endTime\s+1000;", r"writeInterval\s+50;", r"purgeWrite\s+2;", r"writeFormat\s+binary;",
        r"rhoInf\s+1\.225;", r"window\s+100;", r"application\s+simpleFoam;", r"startFrom\s+latestTime;",
        r"deltaT\s+1;", r"windowType\s+approximate;", r"patches\s+\(model\);",
    ):
        assert re.search(pattern, text), pattern


def test_moving_floor_and_rough_floor_fields(tmp_path):
    case, _, _ = write(tmp_path, "f1_car")
    u = (case / "0/U").read_text()
    floor = re.search(r"floor\s*\{(.*?)\}", u, re.S).group(1)
    assert "fixedValue" in floor and "uniform (40 0 0)" in floor
    bcase, _, _ = write(tmp_path, "building", sub="b")
    assert "atmBoundaryLayerInletVelocity" in (bcase / "0/U").read_text()
    assert "atmBoundaryLayerInletK" in (bcase / "0/k").read_text()
    assert "atmBoundaryLayerInletOmega" in (bcase / "0/omega").read_text()
    nut = (bcase / "0/nut").read_text()
    assert "atmNutkWallFunction" in nut and re.search(r"z0\s+uniform 0\.5;", nut)


def test_floor_patch_type_and_library(tmp_path):
    """Wall functions need a wall patch; atmospheric BCs need their library loaded."""
    for obj_id, floor_type in (("f1_car", "wall"), ("building", "wall"), ("human", "patch"), ("airplane", "patch")):
        case, _, _ = write(tmp_path, obj_id, sub=obj_id)
        block = re.search(r"floor\s*\{\s*type\s+(\w+);", (case / "system/blockMeshDict").read_text())
        assert block.group(1) == floor_type, obj_id
        assert re.search(r"libs\s+\(atmosphericModels\);", (case / "system/controlDict").read_text())


def test_floor_variants(tmp_path):
    for floor, u_type in (("none", "slip"), ("slip", "slip"), ("rough", "noSlip")):
        d = tmp_path / floor
        d.mkdir()
        cfd, model = box_model("human", d)
        cfd = replace(cfd, floor=floor, abl_z0_m=0.5 if floor == "rough" else None)
        case = d / "case"
        write_case(case, model, cfd, SETTINGS, RunSpec("human_d0", "human", 0, 0.0, 1000))
        floor_block = re.search(r"floor\s*\{(.*?)\}", (case / "0/U").read_text(), re.S).group(1)
        assert u_type in floor_block, floor


def test_inlet_turbulence_values(tmp_path):
    case, cfd, model = write(tmp_path, "f1_car")
    d = float(model.extents.max())
    k_in = 1.5 * (0.05 * 40.0) ** 2
    omega_in = k_in ** 0.5 / (0.09 ** 0.25 * 0.05 * d)
    k_text = (case / "0/k").read_text()
    omega_text = (case / "0/omega").read_text()
    k_val = float(re.search(r"inlet\s*\{[^}]*?uniform ([0-9.e+-]+);", k_text).group(1))
    om_val = float(re.search(r"inlet\s*\{[^}]*?uniform ([0-9.e+-]+);", omega_text).group(1))
    assert k_val == pytest.approx(k_in, rel=1e-5)
    assert om_val == pytest.approx(omega_in, rel=1e-5)


def test_layers_and_procs(tmp_path):
    case, _, _ = write(tmp_path, "f1_car")
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert re.search(r"nSurfaceLayers\s+3;", snappy)
    assert re.search(r"expansionRatio\s+1\.2;", snappy)
    assert re.search(r"finalLayerThickness\s+0\.4;", snappy)
    assert re.search(r"numberOfSubdomains\s+4;", (case / "system/decomposeParDict").read_text())
    assert re.search(r"method\s+scotch;", (case / "system/decomposeParDict").read_text())


def test_surface_refinement_override_wins(tmp_path):
    cfd, model = box_model("f1_car", tmp_path)
    cfd = replace(cfd, surface_refinement=(2, 3))
    case = tmp_path / "case"
    write_case(case, model, cfd, SETTINGS, RunSpec("f1_car_d0", "f1_car", 0, 0.0, 1000))
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert re.search(r"level\s+\(2 3\);", snappy)
    assert re.search(r"level\s+3;", snappy)  # featureEdges level = surface max
    default, _, _ = write(tmp_path, "f1_car", sub="d")
    assert re.search(r"level\s+\(4 5\);", (default / "system/snappyHexMeshDict").read_text())


def test_block_mesh_covers_tunnel(tmp_path):
    case, cfd, model = write(tmp_path, "f1_car")
    xmin, xmax, ymin, ymax, zmin, zmax = tunnel_bounds(model, SETTINGS.domain_factors, cfd.floor)
    text = (case / "system/blockMeshDict").read_text()
    d = float(model.extents.max())
    h = d / SETTINGS.base_cells_per_size
    counts = [int(np.ceil((b - a) / h)) for a, b in ((xmin, xmax), (ymin, ymax), (zmin, zmax))]
    assert re.search(rf"\({counts[0]} {counts[1]} {counts[2]}\)", text)
    for name in ("inlet", "outlet", "sides", "top", "floor"):
        assert re.search(rf"\b{name}\b", text)
    assert str(xmin) in text and str(zmax) in text


def test_location_in_mesh_outside_model(tmp_path):
    for obj_id in ("airplane", "f1_car", "building"):
        case, cfd, model = write(tmp_path, obj_id, sub=obj_id)
        text = (case / "system/snappyHexMeshDict").read_text()
        m = re.search(r"locationInMesh\s+\(([^)]+)\);", text)
        point = [float(v) for v in m.group(1).split()]
        xmin, xmax, ymin, ymax, zmin, zmax = tunnel_bounds(model, SETTINGS.domain_factors, cfd.floor)
        h = float(model.extents.max()) / SETTINGS.base_cells_per_size
        assert point == pytest.approx([xmin + 1.37 * h, ymin + 1.41 * h, zmin + 1.73 * h], rel=1e-5)
        assert not model.contains([point])[0]


def test_fv_files_have_potential_flow_and_no_cache(tmp_path):
    case, _, _ = write(tmp_path, "f1_car")
    sol = (case / "system/fvSolution").read_text()
    assert re.search(r"potentialFlow\s*\{\s*nNonOrthogonalCorrectors\s+5;", sol)
    assert "cache" not in sol
    assert "cache" not in (case / "system/fvSchemes").read_text()


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.docker
def test_starter_airplane_meshes(tmp_path):
    if not docker_available():
        pytest.skip("Docker is not running")
    cfd = OBJECTS["airplane"]
    model0 = load_prepared(ROOT / "assets/models/airplane/model.stl", cfd)
    model = run_geometry(model0, cfd, cfd.directions_deg[0])
    settings = replace(SETTINGS, base_cells_per_size=4, surface_refinement=(2, 3))
    case = tmp_path / "case"
    write_case(case, model, cfd, settings, RunSpec("airplane_d0", "airplane", 0, 0.0, 1000))

    name = f"navflow-case-writer-{uuid.uuid4().hex[:8]}"
    image = settings.image
    try:
        subprocess.run(["docker", "create", "--name", name, image, "sleep", "3600"], check=True, capture_output=True)
        subprocess.run(["docker", "start", name], check=True, capture_output=True)
        subprocess.run(["docker", "cp", f"{case}/.", f"{name}:/tmp/case"], check=True, capture_output=True)
        run = subprocess.run(
            ["docker", "exec", name, "bash", "-lc",
             "cd /tmp/case && blockMesh && surfaceFeatureExtract && snappyHexMesh -overwrite && checkMesh"],
            capture_output=True, text=True, timeout=1500,
        )
        out = run.stdout + run.stderr
        assert "Mesh OK." in out, out[-3000:]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def test_snappy_cell_caps_are_equal_and_large(tmp_path):
    """maxLocalCells below maxGlobalCells would silently cap refinement; both are the same 6 million."""
    case, _, _ = write(tmp_path, "f1_car")
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert re.search(r"maxLocalCells\s+6000000;", snappy)
    assert re.search(r"maxGlobalCells\s+6000000;", snappy)
