"""Write a complete OpenFOAM v2512 wind-tunnel case for one (object, wind direction) run.

`write_case` turns a rotated, closed model mesh plus the run settings into the files that
blockMesh, surfaceFeatureExtract, snappyHexMesh, decomposePar, potentialFoam and simpleFoam read:

    case/constant/triSurface/model.stl      the model (solid name "model", metres)
    case/system/*                           mesh, control, decomposition, numerics dictionaries
    case/constant/{transportProperties,turbulenceProperties}
    case/0/{U,p,k,omega,nut}                boundary conditions

Patches: inlet (x-min), outlet (x-max), sides (y faces), top, floor (z-min), model.
Most dictionaries are `string.Template` files in solver/foam/templates (`$name` placeholders;
fvSchemes and fvSolution are copied verbatim).

Maths link (IGCSE 0606):
  * Cell size h = D / n, then cells per side = ceil(extent / h): division and rounding up.
  * Turbulence at the inlet: k = 1.5 (0.05 V)^2 (5 percent intensity, a power of a product) and
    omega = sqrt(k) / (0.09^0.25 * 0.05 D): a square root and a fourth root of a constant.
  * Layer thicknesses grow by a common ratio 1.2: a geometric progression.
"""
from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template

import trimesh

from solver.foam.geometry import tunnel_bounds
from solver.foam.settings import FoamSettings, ObjectCfd

TEMPLATES = Path(__file__).parent / "templates"
VERBATIM = ("fvSchemes", "fvSolution")  # copied to system/ without substitution

# Offsets of locationInMesh from the tunnel's lower upstream corner, in cell sizes h. They are
# irrational-looking on purpose so the point never lands on a face of any refined cell.
# snappyHexMesh cell budget. maxLocalCells is set equal to maxGlobalCells: a smaller local cap silently
# stops refinement part-way (it is a limit per processor, and we mesh on one). The runner warns at 90 %.
MAX_GLOBAL_CELLS = 6_000_000
LOCATION_OFFSETS = (1.37, 1.41, 1.73)
TURBULENCE_INTENSITY = 0.05
CMU = 0.09


@dataclass(frozen=True)
class RunSpec:
    """One simulation run: an object at one wind direction. `run_id` is f"{obj_id}_d{direction_index}"."""

    run_id: str
    obj_id: str
    direction_index: int
    direction_deg: float
    max_iterations: int


def _num(x: float) -> str:
    """Compact, exact-enough number for a dictionary (40.0 -> 40, 1.225 -> 1.225)."""
    return format(float(x), ".10g")


def _render(name: str, values: dict[str, object]) -> str:
    """Substitute `$name` placeholders in templates/<name>.tmpl (a literal $ is written $$)."""
    return Template((TEMPLATES / f"{name}.tmpl").read_text()).substitute({k: str(v) for k, v in values.items()})


def inlet_turbulence(speed: float, d: float) -> tuple[float, float]:
    """(k_in, omega_in) for 5 percent turbulence intensity: k = 1.5 (0.05 V)^2, omega = sqrt(k) / (0.09^0.25 * 0.05 D)."""
    k_in = 1.5 * (TURBULENCE_INTENSITY * speed) ** 2
    return k_in, math.sqrt(k_in) / (CMU ** 0.25 * TURBULENCE_INTENSITY * d)


def _block(lines: list[str]) -> str:
    """Indent boundary-condition lines to sit inside a patch's braces."""
    return "\n".join(f"        {line}" for line in lines)


def _abl(cfd: ObjectCfd, zref: float) -> list[str]:
    """Shared atmospheric boundary layer inlet entries: log-law wind profile U(z) = Uref ln((z+z0)/z0) / ln((Zref+z0)/z0)."""
    z0 = cfd.abl_z0_m
    if z0 is None:
        raise ValueError(f"{cfd.obj_id}: a rough floor needs abl_z0_m")
    return [
        f"Uref            {_num(cfd.speed_m_s)};",
        f"Zref            {_num(zref)};",
        f"z0              uniform {_num(z0)};",
        "d               uniform 0;",
        "flowDir         (1 0 0);",
        "zDir            (0 0 1);",
    ]


def _field_blocks(cfd: ObjectCfd, d: float, zref: float) -> dict[str, dict[str, str]]:
    """Inlet and floor boundary-condition text for U, k, omega and nut, from the floor type."""
    v = cfd.speed_m_s
    k_in, omega_in = inlet_turbulence(v, d)
    rough = cfd.floor == "rough"
    fixed_u = ["type            fixedValue;", f"value           uniform ({_num(v)} 0 0);"]
    zero_grad = ["type            zeroGradient;"]
    wall_k = ["type            kqRWallFunction;", f"value           uniform {_num(k_in)};"]
    wall_omega = ["type            omegaWallFunction;", f"value           uniform {_num(omega_in)};"]
    calculated = ["type            calculated;", "value           uniform 0;"]

    if rough:
        abl = _abl(cfd, zref)
        inlet = {
            "U": ["type            atmBoundaryLayerInletVelocity;", *abl, f"value           uniform ({_num(v)} 0 0);"],
            "k": ["type            atmBoundaryLayerInletK;", *abl, f"value           uniform {_num(k_in)};"],
            "omega": ["type            atmBoundaryLayerInletOmega;", *abl, f"value           uniform {_num(omega_in)};"],
        }
    else:
        inlet = {
            "U": fixed_u,
            "k": ["type            fixedValue;", f"value           uniform {_num(k_in)};"],
            "omega": ["type            fixedValue;", f"value           uniform {_num(omega_in)};"],
        }

    if cfd.floor == "moving":
        floor = {"U": fixed_u, "k": wall_k, "omega": wall_omega,
                 "nut": ["type            nutkWallFunction;", "value           uniform 0;"]}
    elif rough:
        floor = {"U": ["type            noSlip;"], "k": wall_k, "omega": wall_omega,
                 "nut": ["type            atmNutkWallFunction;", f"z0              uniform {_num(cfd.abl_z0_m)};",
                         "boundNut        false;", "value           uniform 0;"]}
    else:  # "none" (open air) and "slip": a frictionless lower boundary
        floor = {"U": ["type            slip;"], "k": zero_grad, "omega": zero_grad, "nut": calculated}

    return {
        "inlet": {f: _block(lines) for f, lines in inlet.items()},
        "floor": {f: _block(lines) for f, lines in floor.items()},
    }


def _write_stl(path: Path, mesh: trimesh.Trimesh) -> None:
    """ASCII STL whose single solid is named `model`, so snappyHexMesh's patch is called `model`."""
    text = mesh.export(file_type="stl_ascii")
    text = re.sub(r"\A\s*solid[^\n]*", "solid model", text)
    text = re.sub(r"endsolid[^\n]*\s*\Z", "endsolid model\n", text)
    path.write_text(text)


def write_case(case_dir: Path, run_model: trimesh.Trimesh, cfd: ObjectCfd, s: FoamSettings, spec: RunSpec) -> None:
    """Write every file of the case into `case_dir` (created if needed, existing files overwritten)."""
    case_dir = Path(case_dir)
    for sub in ("0", "constant/triSurface", "system"):
        (case_dir / sub).mkdir(parents=True, exist_ok=True)

    xmin, xmax, ymin, ymax, zmin, zmax = tunnel_bounds(run_model, s.domain_factors, cfd.floor)
    lo, hi = run_model.bounds
    d = float(run_model.extents.max())  # D: longest side of the model, sets every length scale
    h = d / s.base_cells_per_size  # base cell size
    counts = [math.ceil((b - a) / h) for a, b in ((xmin, xmax), (ymin, ymax), (zmin, zmax))]
    surf_min, surf_max = cfd.surface_refinement or s.surface_refinement

    _write_stl(case_dir / "constant/triSurface/model.stl", run_model)

    system = case_dir / "system"
    for name in VERBATIM:
        shutil.copyfile(TEMPLATES / name, system / name)

    (system / "blockMeshDict").write_text(_render("blockMeshDict", {
        "xmin": repr(xmin), "xmax": repr(xmax), "ymin": repr(ymin), "ymax": repr(ymax),
        "zmin": repr(zmin), "zmax": repr(zmax), "nx": counts[0], "ny": counts[1], "nz": counts[2],
        "floor_type": "wall" if cfd.floor in ("moving", "rough") else "patch",
    }))
    (system / "surfaceFeatureExtractDict").write_text(_render("surfaceFeatureExtractDict", {}))
    (system / "snappyHexMeshDict").write_text(_render("snappyHexMeshDict", {
        "wake_xmin": _num(lo[0] - 0.2 * d), "wake_xmax": _num(hi[0] + 3 * d),
        "wake_ymin": _num(lo[1] - 0.5 * d), "wake_ymax": _num(hi[1] + 0.5 * d),
        "wake_zmin": _num(lo[2] - 0.5 * d), "wake_zmax": _num(hi[2] + 0.5 * d),
        "feature_level": surf_max, "surface_min": surf_min, "surface_max": surf_max,
        "wake_level": s.wake_refinement, "max_cells": MAX_GLOBAL_CELLS, "n_layers": s.n_layers,
        "loc_x": repr(xmin + LOCATION_OFFSETS[0] * h),
        "loc_y": repr(ymin + LOCATION_OFFSETS[1] * h),
        "loc_z": repr(zmin + LOCATION_OFFSETS[2] * h),
    }))
    (system / "controlDict").write_text(_render("controlDict", {
        "end_time": spec.max_iterations, "write_interval": s.write_interval,
        "rho": _num(cfd.density_kg_m3), "average_window": s.average_window,
    }))
    (system / "decomposeParDict").write_text(_render("decomposeParDict", {"n_procs": s.n_procs}))

    constant = case_dir / "constant"
    (constant / "transportProperties").write_text(
        _render("transportProperties", {"nu": _num(cfd.kinematic_viscosity_m2_s)}))
    (constant / "turbulenceProperties").write_text(_render("turbulenceProperties", {}))

    v = cfd.speed_m_s
    k_in, omega_in = inlet_turbulence(v, d)
    blocks = _field_blocks(cfd, d, zref=float(hi[2]))  # Zref = model height (roof height for the building)
    zero = case_dir / "0"
    (zero / "U").write_text(_render("U", {"speed": _num(v), "inlet": blocks["inlet"]["U"], "floor": blocks["floor"]["U"]}))
    (zero / "p").write_text(_render("p", {}))
    (zero / "k").write_text(_render("k", {"k_in": _num(k_in), "inlet": blocks["inlet"]["k"], "floor": blocks["floor"]["k"]}))
    (zero / "omega").write_text(_render("omega", {
        "omega_in": _num(omega_in), "inlet": blocks["inlet"]["omega"], "floor": blocks["floor"]["omega"]}))
    (zero / "nut").write_text(_render("nut", {"floor": blocks["floor"]["nut"]}))
