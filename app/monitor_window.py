"""
Monitor Window: Native PySide6 + PyVista 3D Visualization and Scientific HUD.
Renders 3D object mesh colored by surface pressure, moving circular pipe nozzle,
animated flow streamlines, and live Navier-Stokes & Sustainability telemetry.
"""

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeyEvent, QWheelEvent
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QProgressBar, QSizePolicy
)

from app.state_manager import ObjectStateManager
from app.input_controller import ExhibitionInputController


class MonitorWindow(QMainWindow):
    def __init__(self,
                 obj_id: str,
                 input_controller: ExhibitionInputController,
                 cache_dir: str = "assets/cache",
                 parent=None):
        super().__init__(parent)
        self.obj_id = obj_id
        self.controller = input_controller
        self.state_mgr = ObjectStateManager(obj_id=obj_id, cache_dir=cache_dir)
        self.obj_cfg = self.state_mgr.obj_config

        self.setWindowTitle(f"Monitor {self.obj_cfg.get('monitor_index', 0) + 1}: {self.obj_cfg['display_name']} — Navier-Stokes CFD")
        self.resize(1280, 800)

        # Connect global controller signal
        self.controller.state_changed.connect(self.on_state_changed)

        # 3D Animation state
        self.anim_phase = 0.0
        self.streamline_points_cache = None
        self.mesh_actor = None
        self.nozzle_actor = None
        self.streamline_actor = None
        self.particles_actor = None

        self._build_ui()
        self._setup_3d_scene()

        # Animation timer (60 FPS) for flowing particles
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_flow)
        self.anim_timer.start(16)  # ~60 FPS

        # Initial render
        self.update_view(self.controller.current_z_index, self.controller.current_wind_speed)

    def _build_ui(self):
        """Constructs the high-contrast dark-mode aerospace UI layout."""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #0A0E17; color: #FFFFFF;")

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        # Left: PyVista 3D Interactor Viewport
        self.plotter = QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.plotter, stretch=7)

        # Right: Scientific HUD Dashboard
        hud_panel = QFrame()
        hud_panel.setFixedWidth(420)
        hud_panel.setStyleSheet("""
            QFrame {
                background-color: #121826;
                border: 1px solid #1F293D;
                border-radius: 8px;
            }
            QLabel {
                border: none;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            }
        """)
        hud_layout = QVBoxLayout(hud_panel)
        hud_layout.setContentsMargins(18, 18, 18, 18)
        hud_layout.setSpacing(12)

        accent = self.obj_cfg.get("color", "#00E5FF")

        # 1. Header Card
        lbl_category = QLabel(self.obj_cfg.get("category", "Fluid Dynamics").upper())
        lbl_category.setStyleSheet(f"color: {accent}; font-size: 11px; font-weight: 700; letter-spacing: 2px;")
        hud_layout.addWidget(lbl_category)

        lbl_title = QLabel(self.obj_cfg.get("display_name", "Object"))
        lbl_title.setStyleSheet("font-size: 24px; font-weight: 800; color: #FFFFFF;")
        hud_layout.addWidget(lbl_title)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet(f"background-color: {accent}; max-height: 2px; border: none;")
        hud_layout.addWidget(separator)

        # 2. Live Hardware Inputs (Z-Height & Wind Velocity)
        input_box = QFrame()
        input_box.setStyleSheet("background-color: #1A2333; border-radius: 6px; padding: 10px;")
        ib_layout = QVBoxLayout(input_box)
        ib_layout.setSpacing(8)

        self.lbl_z_status = QLabel("PIPE NOZZLE: Step 5 / 10 (Z = 0.45 m)")
        self.lbl_z_status.setStyleSheet("font-size: 13px; font-weight: 600; color: #E2E8F0;")
        ib_layout.addWidget(self.lbl_z_status)

        self.bar_z = QProgressBar()
        self.bar_z.setRange(1, 10)
        self.bar_z.setValue(5)
        self.bar_z.setTextVisible(False)
        self.bar_z.setStyleSheet(f"""
            QProgressBar {{ background-color: #0A0E17; border-radius: 3px; max-height: 8px; border: none; }}
            QProgressBar::chunk {{ background-color: {accent}; border-radius: 3px; }}
        """)
        ib_layout.addWidget(self.bar_z)

        self.lbl_v_status = QLabel("WIND VELOCITY: 30.0 m/s (108.0 km/h)")
        self.lbl_v_status.setStyleSheet("font-size: 13px; font-weight: 600; color: #E2E8F0;")
        ib_layout.addWidget(self.lbl_v_status)

        self.bar_v = QProgressBar()
        self.bar_v.setRange(10, 50)
        self.bar_v.setValue(30)
        self.bar_v.setTextVisible(False)
        self.bar_v.setStyleSheet("""
            QProgressBar { background-color: #0A0E17; border-radius: 3px; max-height: 8px; border: none; }
            QProgressBar::chunk { background-color: #FF9100; border-radius: 3px; }
        """)
        ib_layout.addWidget(self.bar_v)

        hud_layout.addWidget(input_box)

        # 3. Navier-Stokes Calculus & Math Telemetry Card
        math_box = QFrame()
        math_box.setStyleSheet("background-color: #161F30; border-radius: 6px; padding: 10px;")
        mb_layout = QVBoxLayout(math_box)
        mb_layout.setSpacing(6)

        lbl_math_hdr = QLabel("NAVIER-STOKES CALCULUS HUD")
        lbl_math_hdr.setStyleSheet("font-size: 11px; font-weight: 700; color: #64B5F6; letter-spacing: 1px;")
        mb_layout.addWidget(lbl_math_hdr)

        lbl_pde = QLabel("ρ[∂v/∂t + (v·∇)v] = -∇p + μ∇²v")
        lbl_pde.setStyleSheet("font-size: 12px; font-family: monospace; color: #90CAF9; background-color: #0D131F; padding: 4px; border-radius: 4px;")
        mb_layout.addWidget(lbl_pde)

        self.lbl_reynolds = QLabel("Reynolds Number Re: 2.45 × 10⁶")
        self.lbl_reynolds.setStyleSheet("font-size: 13px; color: #CFD8DC;")
        mb_layout.addWidget(self.lbl_reynolds)

        self.lbl_drag_force = QLabel("Aerodynamic Drag F_drag: 142.3 N")
        self.lbl_drag_force.setStyleSheet("font-size: 13px; font-weight: 700; color: #FF5252;")
        mb_layout.addWidget(self.lbl_drag_force)

        self.lbl_lift_force = QLabel("Aerodynamic Lift F_lift: +18.2 N")
        self.lbl_lift_force.setStyleSheet("font-size: 13px; color: #B0BEC5;")
        mb_layout.addWidget(self.lbl_lift_force)

        self.lbl_cd = QLabel("Drag Coefficient C_d: 0.38")
        self.lbl_cd.setStyleSheet("font-size: 13px; color: #CFD8DC;")
        mb_layout.addWidget(self.lbl_cd)

        self.lbl_pressure_peak = QLabel("Stagnation Pressure Δp: 551.2 Pa")
        self.lbl_pressure_peak.setStyleSheet("font-size: 13px; color: #CFD8DC;")
        mb_layout.addWidget(self.lbl_pressure_peak)

        hud_layout.addWidget(math_box)

        # 4. Sustainability & Efficiency Card
        sust_box = QFrame()
        sust_box.setStyleSheet("background-color: #13241C; border: 1px solid #1B4D3E; border-radius: 6px; padding: 10px;")
        sb_layout = QVBoxLayout(sust_box)
        sb_layout.setSpacing(6)

        lbl_sust_hdr = QLabel("SUSTAINABILITY & EFFICIENCY IMPACT")
        lbl_sust_hdr.setStyleSheet("font-size: 11px; font-weight: 700; color: #69F0AE; letter-spacing: 1px;")
        sb_layout.addWidget(lbl_sust_hdr)

        self.lbl_power_cubic = QLabel("Drag Power P = ½ρCdA·V³: 4.27 kW")
        self.lbl_power_cubic.setStyleSheet("font-size: 13px; font-weight: 700; color: #FFD740;")
        sb_layout.addWidget(self.lbl_power_cubic)

        self.lbl_fuel_burn = QLabel("Energy / Fuel Consumption: 1.28 L/h")
        self.lbl_fuel_burn.setStyleSheet("font-size: 13px; color: #E8F5E9;")
        sb_layout.addWidget(self.lbl_fuel_burn)

        self.lbl_co2_savings = QLabel("CO₂ Saved vs Baseline: +3.8 kg/h")
        self.lbl_co2_savings.setStyleSheet("font-size: 13px; font-weight: 700; color: #69F0AE;")
        sb_layout.addWidget(self.lbl_co2_savings)

        self.lbl_eff_gain = QLabel("Efficiency Gain: +28.4%")
        self.lbl_eff_gain.setStyleSheet("font-size: 13px; font-weight: 700; color: #B9F6CA;")
        sb_layout.addWidget(self.lbl_eff_gain)

        hud_layout.addWidget(sust_box)

        # 5. Explanatory Note
        self.lbl_didactic = QLabel(self.obj_cfg.get("description", ""))
        self.lbl_didactic.setWordWrap(True)
        self.lbl_didactic.setStyleSheet("font-size: 11px; color: #94A3B8; line-height: 1.4;")
        hud_layout.addWidget(self.lbl_didactic)

        hud_layout.addStretch()

        # Instructions / Macropad reminder
        lbl_footer = QLabel("Controls: 10 Buttons for Z-Height (1-0) | Knob for Wind Speed (+ / -)")
        lbl_footer.setStyleSheet("font-size: 10px; color: #64748B; font-weight: 600;")
        hud_layout.addWidget(lbl_footer)

        main_layout.addWidget(hud_panel, stretch=3)

    def _setup_3d_scene(self):
        """Initializes the PyVista 3D scene (lighting, camera, wind tunnel bounding box)."""
        self.plotter.set_background("#060A10")
        self.plotter.show_axes()

        # Draw Wind Tunnel Outline Wireframe Box: [0, 2.0] x [0, 1.0] x [0, 1.0]
        tunnel_box = pv.Box(bounds=(0.0, 2.0, 0.0, 1.0, 0.0, 1.0))
        self.plotter.add_mesh(tunnel_box, color="#1E293B", style="wireframe", line_width=1.5, opacity=0.4)

        # Base 3D Model PolyData
        v = self.state_mgr.mesh_vertices
        f = self.state_mgr.mesh_faces
        # Format faces for PyVista: [3, v0, v1, v2, 3, v3, v4, v5, ...]
        pv_faces = np.column_stack([np.full(len(f), 3), f]).ravel()
        self.mesh_poly = pv.PolyData(v, pv_faces)
        self.mesh_poly.point_data["Pressure (Pa)"] = np.zeros(len(v), dtype=np.float32)

        # Add 3D model mesh with scientific pressure colormap
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_poly,
            scalars="Pressure (Pa)",
            cmap="turbo",
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Surface Pressure p (Pa)",
                "color": "#FFFFFF",
                "vertical": False,
                "position_x": 0.25,
                "position_y": 0.04,
                "width": 0.5,
                "height": 0.06
            }
        )

        # 3D Moving Circular Pipe Nozzle Indicator
        # Pipe ring at inlet face x=0, pointing along X-axis
        pipe_cylinder = pv.Cylinder(center=(0.05, 0.5, 0.5), direction=(1, 0, 0), radius=0.08, height=0.10, resolution=32)
        self.nozzle_actor = self.plotter.add_mesh(
            pipe_cylinder,
            color=self.obj_cfg.get("color", "#00E5FF"),
            opacity=0.85,
            smooth_shading=True
        )

        # Set default isometric camera view
        self.plotter.camera_position = [
            (2.9, -1.2, 1.8),   # Eye
            (1.0, 0.5, 0.5),    # Focal point (tunnel center)
            (0.0, 0.0, 1.0)     # Up vector
        ]

    def on_state_changed(self, z_idx: int, speed: float):
        """Slot invoked whenever any macropad key or knob is adjusted."""
        self.update_view(z_idx, speed)

    def update_view(self, z_idx: int, speed: float):
        """Loads interpolated CFD state and updates 3D mesh and HUD readouts."""
        state = self.state_mgr.get_state(z_step_index=z_idx, wind_speed=speed)
        pz = state["z_val"]
        ws = state["wind_speed"]
        metrics = state["metrics"]

        # 1. Update 3D Surface Pressure on Model
        self.mesh_poly.point_data["Pressure (Pa)"] = state["surface_pressure"]

        # 2. Update 3D Circular Pipe Nozzle Position along Z
        if self.nozzle_actor is not None:
            self.plotter.remove_actor(self.nozzle_actor)
            new_pipe = pv.Cylinder(center=(0.05, 0.5, pz), direction=(1, 0, 0), radius=0.08, height=0.10, resolution=32)
            self.nozzle_actor = self.plotter.add_mesh(
                new_pipe,
                color=self.obj_cfg.get("color", "#00E5FF"),
                opacity=0.9,
                smooth_shading=True
            )

        # 3. Update 3D Streamlines
        streamline_data = state["streamlines"]
        self.streamline_points_cache = streamline_data

        if self.streamline_actor is not None:
            self.plotter.remove_actor(self.streamline_actor)

        if len(streamline_data) > 0:
            lines_poly = pv.PolyData(streamline_data)
            self.streamline_actor = self.plotter.add_mesh(
                lines_poly,
                color="#00E5FF",
                line_width=2.5,
                opacity=0.65
            )

        # 4. Update HUD Text & Indicators
        self.lbl_z_status.setText(f"PIPE NOZZLE: Step {z_idx + 1} / {len(self.state_mgr.z_steps)} (Z = {pz:.2f} m)")
        self.bar_z.setValue(z_idx + 1)

        speed_kmh = ws * 3.6
        self.lbl_v_status.setText(f"WIND VELOCITY: {ws:.1f} m/s ({speed_kmh:.1f} km/h)")
        self.bar_v.setValue(int(ws))

        # Navier-Stokes Math HUD
        re = metrics.get("reynolds_number", 0.0)
        self.lbl_reynolds.setText(f"Reynolds Number Re: {re:,.0f} (Turbulent)")
        self.lbl_drag_force.setText(f"Aerodynamic Drag F_drag: {metrics.get('drag_force_n', 0.0):.1f} N")
        self.lbl_lift_force.setText(f"Aerodynamic Lift F_lift: {metrics.get('lift_force_n', 0.0):+.1f} N")
        self.lbl_cd.setText(f"Drag Coefficient C_d: {metrics.get('drag_coefficient_cd', 0.0):.2f}")
        self.lbl_pressure_peak.setText(f"Stagnation Pressure Δp: {metrics.get('max_pressure_pa', 0.0):.1f} Pa")

        # Sustainability HUD
        kw = metrics.get("drag_power_kw", 0.0)
        self.lbl_power_cubic.setText(f"Drag Power P = ½ρCdA·V³: {kw:.2f} kW")
        self.lbl_fuel_burn.setText(f"Energy / Fuel Burn: {metrics.get('fuel_burn_l_per_hr', 0.0):.2f} L/h")
        self.lbl_co2_savings.setText(f"CO₂ Saved vs Baseline: +{metrics.get('co2_saved_kg_per_hr', 0.0):.1f} kg/h")
        self.lbl_eff_gain.setText(f"Aerodynamic Efficiency Gain: +{metrics.get('efficiency_gain_pct', 0.0):.1f}%")

        self.plotter.render()

    def _animate_flow(self):
        """Animates flow tracers/particles advancing through the wind tunnel."""
        if self.streamline_points_cache is None or len(self.streamline_points_cache) == 0:
            return

        # Advance phase proportional to wind velocity
        speed = self.controller.current_wind_speed
        self.anim_phase = (self.anim_phase + 0.02 * (speed / 30.0)) % 1.0

        # Sample points along streamlines
        pts = self.streamline_points_cache
        n_pts = len(pts)
        if n_pts > 20:
            indices = np.linspace(int(self.anim_phase * 20), n_pts - 1, num=35, dtype=int)
            sample_pts = pts[indices]
            valid_pts = sample_pts[~np.isnan(sample_pts).any(axis=1)]

            if len(valid_pts) > 0:
                particle_poly = pv.PolyData(valid_pts)
                if self.particles_actor is not None:
                    self.plotter.remove_actor(self.particles_actor)
                self.particles_actor = self.plotter.add_mesh(
                    particle_poly,
                    color="#FFFFFF",
                    point_size=7.0,
                    render_points_as_spheres=True,
                    opacity=0.9
                )

    def keyPressEvent(self, event: QKeyEvent):
        """Forward all macropad/keyboard events to global controller."""
        if not self.controller.handle_key_press(event):
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Forward wheel events to global controller."""
        if not self.controller.handle_wheel_event(event):
            super().wheelEvent(event)
