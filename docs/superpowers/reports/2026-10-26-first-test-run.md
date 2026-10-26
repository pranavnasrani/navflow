# First full-settings test run (Task 7)

**Run:** airplane (starter placeholder model), direction index 4 (8° angle of attack), 102 m/s, cap 1000, default settings (base_cells_per_size 6, surface_refinement (4, 5), wake 2, 3 layers), 4 MPI processes in Docker on the M4 MacBook Pro.

| Measure | Value |
|---|---|
| Cells (checkMesh) | 478,668 |
| snappyHexMesh time | 62 s |
| simpleFoam, 1000 iterations | 462 s (≈ 0.46 s per iteration) |
| Whole run incl. mesh + export | ≈ 10 min |
| Result folder size | 112 MB |
| Outcome | `flagged` (hit the cap, exported 100-iteration averaged fields) |

**Convergence:** the forces settle into a steady periodic wobble rather than a constant value. Over each 50-iteration window after iteration 400, drag swings ≈ 5% and lift ≈ 15%, but the window means are steady (drag 1.513–1.515 MN). That's the blocky placeholder shedding vortices; the averaged-field export is the planned answer for this. The 0.2% rule will mainly trigger on smooth real models.

**Budget prediction:** 36 runs × ≈ 10 min ≈ 6 machine-hours (≈ 3 h across both Macs), far under the 27 machine-hour budget. No mesh coarsening needed (Task 7 Step 3). There is room to make the mesh finer once the real models arrive — e.g. try surface_refinement (5, 6) in the Tuesday preview and keep it if a run stays under ≈ 45 min.
