# NavFlow OpenFOAM build: decisions made during the build

Branch `feat/openfoam-pipeline` (from `main` at 8338a60; `v1-fallback` tags the old exhibit). Every task was built by a Sonnet implementer, checked by a separate reviewer, and the whole branch had two final reviews. Suite: 218 tests pass, plus 2 Docker tests.

## Real OpenFOAM checks done

- **Airplane** (free air), 8°, 1000 iterations: 478,668 cells, about 10 minutes per run. Flagged, with averaged fields exported. See `2026-10-26-first-test-run.md`.
- **Building** (rough ground with a real wind profile), 30°, 100 iterations: 279,512 cells. Works.
- **Human** (slip floor), 90°, 100 iterations: 293,406 cells. Works.
- **F1 car** (moving floor), 8°, 100 iterations: 382,843 cells. Works, with one mesh-quality warning recorded (see the last ruling).
- **Stop and resume** are tested in real Docker.

## Decisions I made on your behalf (each with what it costs if wrong)

1. **Viscous friction bar.** The Navier-Stokes "friction" bar uses the turbulence model's effective viscosity (ν + ν_t), and is labelled "incl. turbulent mixing, k-ω SST". With laminar ν alone it was about 10,000× smaller than the other terms everywhere, so the "friction matters near the surface" point never showed.
   - *Cost if wrong:* judges must be told the bar includes turbulence, not just molecular viscosity.
2. **Fuel and CO₂ savings.** These are now measured against the object's own worst simulated wind angle, not a "brick" with C_d = 1.0. The old comparison claimed the plane saved 67,200 L/h, while a real A380 burns about 12,000 L/h.
   - *Cost if wrong:* savings are smaller, and zero at the worst angle.
3. **Sim folders in Docker.** Test, preview and full runs each get their own folder inside Docker, and only one solver runs per machine at a time.
   - *Cost if wrong:* you can't run a test while a batch is solving on the same Mac.
4. **Never losing hours of work.** These cases no longer throw away or lose a run:
   - a Docker hiccup, which no longer counts as "volume lost";
   - a NaN blow-up, which is now caught while solving;
   - an export failure, which can now be re-exported;
   - a lost stop request, which is now retried.
5. **Retrying failed runs.** `run --retry-failed` re-queues failed runs. The Mac mini's results can never overwrite the laptop's progress file.
6. **Mesh check.** A mesh passes when its only problems are quality warnings (for example one slightly skewed face). Real errors, or a crashed or incomplete check, still fail. The warnings are saved in `run.json`.
   - *Cost if wrong:* a poor mesh could run. It stays visible in the log.
7. **Model rotation.** Prepared models are never rotated twice. Real-model rotations are set once, in the config.
8. **Smoke-wand height 1** starts just above the floor for the car, the building and the human. It used to start underground, so button 1 showed no smoke.
9. **Mesh size cap.** The cell cap was raised to 6 million, and every run reports its cell count. Before, finer meshes would have been capped silently at 2 million.
10. **The building's "worst wind direction"** is one fixed answer, taken from the solved angles. It is no longer a running guess.
11. **Plane speed:** Mach 0.3 (102 m/s). The fuel widget is worked out at cruise (11 km, 250 m/s) and ignores shock-wave drag.
12. **Streamlines** are plain cyan in the pressure view (the cache has no speed per point), coloured by the biggest Navier-Stokes term in view 2, and by turbulence k in view 3.
13. **GUI testing.** There is no automated test of the 3D window, because Qt can't open OpenGL without a screen here. It needs one manual run on real data before the exhibition.
14. **Reviewers stayed on Sonnet,** per your standing rule (no Opus unless asked).

## Still to do (needs you or the real models)

- **Models:** put your downloads in `assets/models/incoming/<object>/` and run `solver/prepare_model.py --object <id>`. Then set `base_rotation_deg` until each preview faces −x. When the real rotations land, switch the Docker test at `tests/test_runner.py:686` to `load_prepared`.
- **Preview:** `run_openfoam.py run --objects airplane f1_car building human --preview`, then import and check it in the app. After that the settings are frozen.
- **Full run:** use both Macs. Pass `--machine mini` on the Mac mini.
- **Import and commit:** run `import_results.py` and commit the new `assets/cache/*.npz` files. Until then the app exits with a message, because the committed caches are v1.
- **Rehearsal:** do one full run of all four windows with the macropad, and the iPad on the phone hotspot.
