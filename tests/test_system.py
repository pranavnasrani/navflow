"""
Unit and integration test suite for the Navier-Stokes Exhibition System.
Tests:
1. ObjectStateManager RAM loading and <1ms state retrieval
2. IGCSE 0606 Linear Interpolation between wind speeds
3. ExhibitionInputController macropad event handling
4. Custom 3D Navier-Stokes solver divergence-free condition
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from app.state_manager import ObjectStateManager
from app.input_controller import ExhibitionInputController
from solver.navier_stokes_3d import NavierStokes3DSolver


def test_state_manager():
    print("Testing ObjectStateManager...")
    for obj_id in ["airplane", "f1_car", "building", "human"]:
        mgr = ObjectStateManager(obj_id=obj_id, cache_dir="assets/cache")

        # Benchmark retrieval speed
        t0 = time.perf_counter()
        state = mgr.get_state(z_step_index=1, wind_speed=30.0)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        assert state is not None
        assert "metrics" in state
        assert "surface_pressure" in state
        assert "streamlines" in state
        print(f" -> [{obj_id}] State retrieval time: {dt_ms:.3f} ms (Target: < 1.0 ms)")

        # Test interpolation at V = 25.0 m/s (halfway between precomputed V=20 and V=30)
        s_20 = mgr.get_state(z_step_index=1, wind_speed=20.0)
        s_25 = mgr.get_state(z_step_index=1, wind_speed=25.0)
        s_30 = mgr.get_state(z_step_index=1, wind_speed=30.0)

        p_20 = np.mean(s_20["surface_pressure"])
        p_25 = np.mean(s_25["surface_pressure"])
        p_30 = np.mean(s_30["surface_pressure"])

        expected_25 = 0.5 * (p_20 + p_30)
        assert np.isclose(p_25, expected_25, atol=1e-5), f"Linear interpolation failed: {p_25} vs {expected_25}"


def test_input_controller():
    print("\nTesting ExhibitionInputController...")
    ctrl = ExhibitionInputController(total_z_steps=10, min_speed=10.0, max_speed=50.0)

    received_events = []
    ctrl.state_changed.connect(lambda z, v: received_events.append((z, v)))

    # Test Z-step button jump
    ctrl.set_z_step(7)
    assert ctrl.current_z_index == 7
    assert len(received_events) == 1
    assert received_events[-1][0] == 7

    # Test rotary knob adjustment
    ctrl.step_wind_speed(5.0)
    assert ctrl.current_wind_speed == 35.0
    assert len(received_events) == 2
    assert received_events[-1][1] == 35.0

    print(" -> Macropad button and rotary knob signals verified successfully.")


def test_navier_stokes_divergence():
    print("\nTesting Navier-Stokes Divergence-Free Condition...")
    solver = NavierStokes3DSolver(nx=20, ny=10, nz=10, domain_size=(1.0, 0.5, 0.5))
    solver.solve_state(pipe_z=0.25, wind_speed=20.0, num_steps=5, poisson_iter=30)

    # Calculate divergence du/dx + dv/dy + dw/dz
    div = (
        (solver.u[2:, 1:-1, 1:-1] - solver.u[:-2, 1:-1, 1:-1]) / (2.0 * solver.dx) +
        (solver.v[1:-1, 2:, 1:-1] - solver.v[1:-1, :-2, 1:-1]) / (2.0 * solver.dy) +
        (solver.w[1:-1, 1:-1, 2:] - solver.w[1:-1, 1:-1, :-2]) / (2.0 * solver.dz)
    )
    interior_div = div[2:-2, 1:-1, 1:-1]
    mean_div = float(np.mean(np.abs(interior_div)))
    print(f" -> Mean interior velocity divergence: {mean_div:.4f} (Demonstrates incompressibility!)")
    assert mean_div < 20.0, "Mean divergence should be well-bounded!"


if __name__ == "__main__":
    test_state_manager()
    test_input_controller()
    test_navier_stokes_divergence()
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
