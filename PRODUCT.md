# Product

<!-- impeccable:product-schema 1 -->

## Platform

desktop

Native macOS desktop app (PySide6 + PyVista/VTK), one fullscreen window per monitor. Not a web app; the iPad page is only a remote control.

## Users
- **Visitors** (students and parents) at the Math section of a school STEM exhibition, Saturday 31 Oct 2026. They walk up, press the macropad buttons and turn the knob, and watch the flow change.
- **Judges**, who question the team (Isaac, Sameer, Pranav) on the maths behind each screen. A team member is usually standing at the exhibit to explain.

## Product Purpose
Make the 3D incompressible Navier-Stokes equations visible on four real engineered shapes (airliner, F1 car, skyscraper, human), and link the flow to sustainability: fuel and CO2, drag power, wind load. Success: a visitor sees the air bend around the object within seconds, and a judge can trace every number on screen back to IGCSE Additional Maths (0606).

## Positioning
The flow is real OpenFOAM (simpleFoam, k-omega SST) solved by the team for 9 directions per object, not an animation or a toy solver. Every on-screen number comes from those runs.

## Operating Context
- Four monitors of about 20 inches, side by side, in a bright school hall, viewed from about 1 to 2 m.
- One USB macropad drives all four screens: 10 buttons (smoke wand height), 1 rotary knob (wind direction), Tab (view: surface pressure / dominant N-S term / turbulence k). An iPad remote does the same over local Wi-Fi.
- Each object runs at one fixed speed (airplane 102 m/s, F1 car 40, building 50, human 10).

## Capabilities and Constraints
- Results load from `assets/cache/<object>_states.npz` into RAM; switching is under 1 ms, so the display must never wait on computation.
- Streamlines currently carry turbulent k and the dominant N-S term per point; velocity magnitude along the lines is not yet in the cache (open decision).
- Four windows share one GPU, so rendering cost per window matters.
- Code comments must keep explaining the IGCSE 0606 maths behind each step.

## Evidence on Hand
- Real OpenFOAM preview and full results in `runs/` (git-ignored); models in `assets/models/<id>/model.stl` (A380, Red Bull RB22, Burj Khalifa, starter human).
- The airplane's fuel numbers are currently unrealistic (mesh too coarse, compared against a stalled angle); do not present them as final.

## Product Principles
1. The flow is the exhibit: everything else supports reading it.
2. Every number is real and explainable with 0606 maths; nothing decorative pretends to be data.
3. Readable from 2 m in a bright hall, by a visitor who has never seen CFD.
4. One control panel drives all four screens, so all four must read as one system.

## Accessibility & Inclusion
Term colours must stay distinguishable for colour-blind visitors; text must be legible from 2 m.
