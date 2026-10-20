"""
Automated OpenFOAM Case Generator & Docker Runner.
Sets up simpleFoam (steady-state incompressible Navier-Stokes) cases with
a circular pipe inlet at Z-height and immersed 3D STL geometry.
"""

from pathlib import Path
import json
import subprocess


class OpenFOAMBridge:
    def __init__(self, case_root: Path | str = "solver/openfoam_cases"):
        self.case_root = Path(case_root)
        self.case_root.mkdir(parents=True, exist_ok=True)

    def generate_case_structure(self,
                                case_name: str,
                                stl_path: Path | str,
                                pipe_z: float,
                                wind_speed: float) -> Path:
        """
        Creates an OpenFOAM case directory with system/, constant/, and 0/ dictionaries.
        """
        case_dir = self.case_root / case_name
        system_dir = case_dir / "system"
        constant_dir = case_dir / "constant"
        zero_dir = case_dir / "0"

        for d in [system_dir, constant_dir, zero_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 1. controlDict
        control_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  10                                    |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;
writeControl    timeStep;
writeInterval   50;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        (system_dir / "controlDict").write_text(control_dict)

        # 2. fvSchemes
        fv_schemes = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
ddtSchemes
{
    default         steadyState;
}
gradSchemes
{
    default         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear corrected;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         corrected;
}
"""
        (system_dir / "fvSchemes").write_text(fv_schemes)

        # 3. fvSolution
        fv_solution = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
}
relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
"""
        (system_dir / "fvSolution").write_text(fv_solution)

        # 4. blockMeshDict
        block_mesh = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      blockMeshDict;
}
convertToMeters 1;
vertices
(
    (0 0 0)
    (2.0 0 0)
    (2.0 1.0 0)
    (0 1.0 0)
    (0 0 1.0)
    (2.0 0 1.0)
    (2.0 1.0 1.0)
    (0 1.0 1.0)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (40 20 20) simpleGrading (1 1 1)
);
edges ();
boundary
(
    inlet
    {
        type patch;
        faces ((0 4 7 3));
    }
    outlet
    {
        type patch;
        faces ((1 2 6 5));
    }
    walls
    {
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
"""
        (system_dir / "blockMeshDict").write_text(block_mesh)

        # 5. Boundary Fields (0/U and 0/p)
        u_field = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    location    "0";
    object      U;
}}
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue    constant ({wind_speed} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            slip;
    }}
}}
"""
        (zero_dir / "U").write_text(u_field)

        p_field = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }
    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }
    walls
    {
        type            zeroGradient;
    }
}
"""
        (zero_dir / "p").write_text(p_field)

        # constant/transportProperties
        transport = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1.5e-05;
"""
        (constant_dir / "transportProperties").write_text(transport)

        return case_dir

    def run_docker_case(self, case_dir: Path | str, docker_image: str = "openfoam/openfoam10-paraview510") -> bool:
        """
        Executes blockMesh, simpleFoam, and foamToVTK inside an OpenFOAM Docker container.
        """
        case_dir = Path(case_dir).resolve()
        script = """
        source /opt/openfoam10/etc/bashrc
        blockMesh
        simpleFoam
        foamToVTK -latestTime
        """
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{case_dir}:/home/openfoam/run",
            "-w", "/home/openfoam/run",
            docker_image,
            "bash", "-c", script
        ]
        try:
            print(f"Launching OpenFOAM Docker run in {case_dir}...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("OpenFOAM Docker execution finished successfully.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Docker execution failed or Docker not running: {e}")
            return False
