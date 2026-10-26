"""
Monitor Window: native PySide6 + PyVista 3D view and scientific HUD for one exhibition object.

One window per object (airplane, F1 car, tower, human). All four share a single
ExhibitionInputController, so a button press or knob turn on the macropad updates every screen.

The visitor controls
    * the smoke wand: 10 heights (the macropad buttons) - a thin rod across y that releases streamlines;
    * the wind direction: a rotary knob mapped onto the object's direction sweep (angle of attack,
      yaw, wind angle ...). The wind arrow in the 3D scene turns with it;
    * the view (Tab): 0 airflow + surface pressure, 1 which Navier-Stokes term dominates each streamline,
      2 turbulence energy k.
The wind speed is fixed per object (it is what OpenFOAM solved), and is shown as such.

Maths link (IGCSE 0606): the camera is aimed at the midpoint of the display box, ((min + max) / 2 on each
axis, coordinate geometry); the wind arrow is a 3D vector (direction cosines of the wind in the model
frame); the tracer dots slide along the lines by linear interpolation; the bars are logarithmic.
"""

from __future__ import annotations

import numpy as np
import pyvista as pv
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.hud_widgets import (
    TERM_COLORS,
    TERM_NAMES,
    ColorBar,
    NsTermsPanel,
    SustainabilityOverlay,
    TermLegend,
    hex_to_rgb,
    line_runs,
    particles_on_lines,
    polyline_from_nan_joined,
)
from app.input_controller import ExhibitionInputController
from app.state_manager import ObjectStateManager

VIEW_NAMES = (
    "VIEW 1/3 · Airflow and surface pressure",
    "VIEW 2/3 · Dominant Navier–Stokes term",
    "VIEW 3/3 · Turbulence energy k",
)
PRESSURE_TITLE = "Surface pressure p (Pa)"
K_TITLE = "Turbulence energy k (m²/s²) · k-ω SST"
MODEL_GREY = "#8D99AE"
ANIM_INTERVAL_MS = 33  # about 30 frames per second; four windows share one GPU
HUD_MIN_WIDTH, HUD_MAX_WIDTH = 330, 440
COMPACT_HEIGHT = 820  # below this window height the long explanatory texts are hidden so nothing is squeezed
PHASE_PER_FRAME = 0.006  # fraction of a streamline the tracers advance per frame (a lap takes about 5.5 s)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else np.array([1.0, 0.0, 0.0])


class MonitorWindow(QMainWindow):
    def __init__(
        self,
        obj_id: str,
        input_controller: ExhibitionInputController,
        cache_dir: str = "assets/cache",
        remote_url: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.obj_id = obj_id
        self.controller = input_controller
        self.remote_url = remote_url
        self.state_mgr = ObjectStateManager(obj_id=obj_id, cache_dir=cache_dir)
        self.obj_cfg = self.state_mgr.obj_config
        self.cfd_cfg = self.obj_cfg.get("cfd", {})
        self.view_bounds = np.asarray(self.state_mgr.view_bounds, dtype=float)

        self.setWindowTitle(
            f"Monitor {self.obj_cfg.get('monitor_index', 0) + 1}: {self.obj_cfg['display_name']} — Navier-Stokes CFD"
        )
        self.resize(1280, 800)
        self.setFocusPolicy(Qt.StrongFocus)

        # Animation state
        self.anim_phase = 0.0
        self._line_points = np.empty((0, 3))
        self._run_starts = np.empty(0, dtype=int)
        self._run_lengths = np.empty(0, dtype=int)

        self._build_ui()
        self._setup_3d_scene()

        # Every key/wheel event that lands on the 3D widget goes through the controller first.
        self.plotter.installEventFilter(self)

        self.controller.state_changed.connect(self.on_state_changed)
        self.controller.view_changed.connect(self.on_view_changed)

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate_flow)
        self.anim_timer.start(ANIM_INTERVAL_MS)

        self.update_view(self.controller.current_wand_index, self.controller.current_knob)
        self.on_view_changed(self.controller.current_view)

    # ------------------------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #0A0E17; color: #F1F5F9;")
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        # Left: the 3D view, with the sustainability cards floating over its top-left corner.
        self.plotter = QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plotter.setFocusPolicy(Qt.StrongFocus)
        main_layout.addWidget(self.plotter, stretch=7)

        self.overlay = SustainabilityOverlay(tuple(self.cfd_cfg.get("widgets", ())), self.plotter)
        self.overlay.move(14, 14)
        self.overlay.adjustSize()
        self.overlay.raise_()

        # Colour bars and the term legend are Qt widgets too: VTK's own scalar bars have no Greek letters.
        self.pressure_bar = ColorBar(PRESSURE_TITLE, "turbo", log=False, parent=self.plotter)
        self.k_bar = ColorBar(K_TITLE, "plasma", log=True, parent=self.plotter)
        self.term_legend = TermLegend(self.plotter)
        for w in (self.pressure_bar, self.k_bar, self.term_legend):
            w.hide()

        # Right: HUD
        hud = QFrame()
        self.hud = hud
        hud.setFixedWidth(HUD_MAX_WIDTH)
        hud.setStyleSheet(
            "QFrame#hud { background-color: #121826; border: 1px solid #1F293D; border-radius: 8px; }"
            "QLabel { border: none; background: transparent; }"
        )
        hud.setObjectName("hud")
        lay = QVBoxLayout(hud)
        self.hud_layout = lay
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(8)

        accent = self.obj_cfg.get("color", "#00E5FF")

        category = QLabel(self.obj_cfg.get("category", "Fluid Dynamics").upper())
        category.setStyleSheet(f"color: {accent}; font-size: 12px; font-weight: 800; letter-spacing: 2px;")
        lay.addWidget(category)
        title = QLabel(self.obj_cfg.get("display_name", "Object"))
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #FFFFFF;")
        title.setWordWrap(True)
        lay.addWidget(title)
        rule = QFrame()
        rule.setFixedHeight(2)
        rule.setStyleSheet(f"background-color: {accent}; border: none;")
        lay.addWidget(rule)

        # Controls readout
        controls = QFrame()
        controls.setObjectName("controls")
        controls.setStyleSheet("QFrame#controls { background-color: #1A2333; border-radius: 6px; }")
        cl = QVBoxLayout(controls)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(4)
        label_style = "font-size: 15px; font-weight: 700; color: #F1F5F9;"
        self.lbl_wand = QLabel("")
        self.lbl_direction = QLabel("")
        self.lbl_speed = QLabel("")
        for lb in (self.lbl_wand, self.lbl_direction, self.lbl_speed):
            lb.setStyleSheet(label_style)
            cl.addWidget(lb)
        lay.addWidget(controls)

        # View name
        self.lbl_view = QLabel("")
        self.lbl_view.setStyleSheet(f"font-size: 13px; font-weight: 800; color: {accent};")
        self.lbl_view.setWordWrap(True)
        lay.addWidget(self.lbl_view)

        # Navier-Stokes bars
        self.ns_panel = NsTermsPanel()
        lay.addWidget(self.ns_panel)

        # Telemetry
        tele = QFrame()
        tele.setObjectName("tele")
        tele.setStyleSheet("QFrame#tele { background-color: #161F30; border-radius: 6px; }")
        tl = QVBoxLayout(tele)
        tl.setContentsMargins(12, 8, 12, 8)
        tl.setSpacing(3)
        self.lbl_re = QLabel("")
        self.lbl_forces = QLabel("")
        self.lbl_coeffs = QLabel("")
        self.lbl_power = QLabel("")
        self.lbl_run = QLabel("")
        for lb, style in (
            (self.lbl_re, "font-size: 13px; color: #E2E8F0;"),
            (self.lbl_forces, "font-size: 14px; font-weight: 700; color: #FF8A80;"),
            (self.lbl_coeffs, "font-size: 13px; color: #E2E8F0;"),
            (self.lbl_power, "font-size: 14px; font-weight: 700; color: #FFD740;"),
            (self.lbl_run, "font-size: 11px; color: #AAB6C8;"),
        ):
            lb.setStyleSheet(style)
            lb.setWordWrap(True)
            tl.addWidget(lb)
        lay.addWidget(tele)

        # Honest note when this direction was copied from the nearest simulated one
        self.lbl_filled = QLabel("")
        self.lbl_filled.setWordWrap(True)
        self.lbl_filled.setStyleSheet(
            "font-size: 12px; font-weight: 700; color: #FFD740; background-color: #3A2F0B;"
            "border-radius: 4px; padding: 4px 8px;"
        )
        self.lbl_filled.hide()
        lay.addWidget(self.lbl_filled)

        # Building info card (earthquake / tsunami scope note)
        self.lbl_info = QLabel(self.obj_cfg.get("info_card", ""))
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet(
            "font-size: 12px; color: #E2E8F0; background-color: #1F2A44; border-radius: 6px; padding: 8px;"
        )
        self.lbl_info.setVisible(bool(self.obj_cfg.get("info_card")))
        lay.addWidget(self.lbl_info)

        desc = QLabel(self.obj_cfg.get("description", ""))
        self.lbl_desc = desc
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 11px; color: #AAB6C8;")
        lay.addWidget(desc)

        lay.addStretch()

        footer_text = "Buttons 1–0: smoke wand · Knob: wind direction · Tab: change view"
        if self.remote_url:
            footer_text += f"\nRemote control: {self.remote_url}"
        footer = QLabel(footer_text)
        footer.setWordWrap(True)
        footer.setStyleSheet("font-size: 11px; color: #8A98AE; font-weight: 600;")
        lay.addWidget(footer)

        main_layout.addWidget(hud, stretch=3)

    # ------------------------------------------------------------------------------------------
    # 3D scene
    # ------------------------------------------------------------------------------------------
    def _setup_3d_scene(self) -> None:
        pl = self.plotter
        pl.set_background("#060A10")
        pl.show_axes()

        b = self.view_bounds
        centre = (b[0::2] + b[1::2]) / 2.0  # midpoint on each axis
        self._box_centre = centre
        self._box_size = b[1::2] - b[0::2]

        # Wireframe of the display box (the region the streamlines are drawn in)
        pl.add_mesh(pv.Box(bounds=tuple(b)), color="#2B3A55", style="wireframe", line_width=1.5, opacity=0.6)

        # The model, coloured by surface pressure in view 0 and plain grey in views 1 and 2
        v, f = self.state_mgr.mesh_vertices, self.state_mgr.mesh_faces
        faces = np.column_stack([np.full(len(f), 3), f]).ravel()
        self.mesh_poly = pv.PolyData(np.asarray(v, dtype=np.float32), faces)
        self.mesh_poly.point_data[PRESSURE_TITLE] = np.zeros(len(v), dtype=np.float32)
        self.mesh_actor = pl.add_mesh(
            self.mesh_poly,
            scalars=PRESSURE_TITLE,
            cmap="turbo",
            smooth_shading=True,
            show_scalar_bar=False,
        )

        # Streamlines: one shared PolyData, three actors (cyan / by dominant term / by k)
        dummy = polyline_from_nan_joined(
            np.array([[b[0], b[2], b[4]], [b[1], b[3], b[5]]]),
            {"k": np.array([1.0, 2.0]), "dominant_rgb": np.zeros((2, 3), dtype=np.uint8)},
        )
        self.lines_poly = dummy
        self.line_actors = [
            pl.add_mesh(self.lines_poly, color="#00E5FF", line_width=3, opacity=0.8, name="lines_cyan"),
            pl.add_mesh(
                self.lines_poly, scalars="dominant_rgb", rgb=True, line_width=3.5, opacity=0.95,
                name="lines_dominant", show_scalar_bar=False,
            ),
            pl.add_mesh(
                self.lines_poly, scalars="k", cmap="plasma", log_scale=True, clim=(1e-2, 1e2), line_width=3.5,
                opacity=0.95, name="lines_k", show_scalar_bar=False,
            ),
        ]

        # Flow tracer dots
        self.particles_poly = pv.PolyData(np.zeros((1, 3), dtype=np.float32))
        pl.add_mesh(
            self.particles_poly, color="#FFFFFF", point_size=8, render_points_as_spheres=True, opacity=0.95,
            name="particles",
        )

        self._fit_camera()

    def _fit_camera(self) -> None:
        """Aim at the box midpoint from the upstream side, above, then zoom so the whole box fits the view."""
        centre = self._box_centre
        diag = float(np.linalg.norm(self._box_size))
        eye = centre + diag * np.array([-0.30, -0.80, 0.40])
        self.plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0)]
        self.plotter.reset_camera(bounds=tuple(self.view_bounds))
        self.plotter.camera.zoom(0.9)  # a little air around the box

    def _wind_arrow(self, wind_vector: np.ndarray) -> pv.PolyData:
        """Arrow at the upstream edge of the display box, pointing along the wind.

        Start from the model centre and walk back against the wind until the ray leaves the box:
        P = C - s d, where d is the unit wind vector and s the smallest positive value that puts one
        coordinate on a face of the box (s_i = distance to that face / |d_i|).
        """
        d = _unit(np.asarray(wind_vector, dtype=float))
        v = np.asarray(self.state_mgr.mesh_vertices, dtype=float)
        c = (v.min(axis=0) + v.max(axis=0)) / 2.0
        lo, hi = self.view_bounds[0::2], self.view_bounds[1::2]
        s = np.inf
        for i in range(3):
            if abs(d[i]) > 1e-9:
                face = lo[i] if d[i] > 0 else hi[i]  # walking against the wind: toward the low face if d_i > 0
                s = min(s, (c[i] - face) / abs(d[i]))
        length = 0.2 * float(self._box_size.max())
        if not np.isfinite(s):
            s = 0.0
        start = c - 0.97 * s * d
        # keep the whole arrow inside the box: if the upstream room is shorter than the arrow, shrink it
        length = min(length, max(0.97 * s, 1e-6))
        return pv.Arrow(start=start, direction=d, scale=length, tip_length=0.35, tip_radius=0.14, shaft_radius=0.05)

    def _wand_rod(self, lines: np.ndarray, z: float) -> pv.PolyData:
        """Thin rod across the wand's seed points (the first point of every streamline)."""
        pts = np.asarray(lines, dtype=float).reshape(-1, 3)
        valid = np.isfinite(pts).all(axis=1)
        prev_nan = np.concatenate([[True], ~valid[:-1]])
        seeds = pts[valid & prev_nan]
        b = self.view_bounds
        if len(seeds) >= 2:
            ext = seeds.max(axis=0) - seeds.min(axis=0)
            axis = int(np.argmax(ext))
            a, e = seeds[np.argmin(seeds[:, axis])], seeds[np.argmax(seeds[:, axis])]
        else:  # no streamlines: a rod across y at the box's upstream end
            a = np.array([b[0] + 0.05 * (b[1] - b[0]), b[2], z])
            e = np.array([b[0] + 0.05 * (b[1] - b[0]), b[3], z])
        radius = 0.007 * float(self._box_size.max())
        return pv.Line(a, e).tube(radius=radius)

    # ------------------------------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------------------------------
    def on_state_changed(self, wand_index: int, knob: float) -> None:
        self.update_view(wand_index, knob)

    def update_view(self, wand_index: int, knob: float) -> None:
        """Look up the state for (wand, knob) in RAM and push it to the 3D scene and the HUD."""
        state = self.state_mgr.get_state(wand_index, knob)
        metrics = state["metrics"]

        # 1. Surface pressure on the model (1st..99th percentile, so one spike does not wash out the colours)
        p = np.asarray(state["surface_pressure"], dtype=np.float32)
        self.mesh_poly.point_data[PRESSURE_TITLE] = p
        if p.size:
            lo, hi = np.percentile(p, [1, 99])
            if hi - lo < 1e-9:
                lo, hi = lo - 1.0, hi + 1.0
            self.mesh_actor.mapper.scalar_range = (float(lo), float(hi))
            self.pressure_bar.set_range(float(lo), float(hi))

        # 2. Streamlines, clipped to the display box. Dominant term -> a fixed colour per term.
        dom = np.asarray(state["dominant"])
        palette = np.array([hex_to_rgb(TERM_COLORS[n]) for n in TERM_NAMES] + [(255, 255, 255)], dtype=np.uint8)
        rgb = palette[np.where(dom >= 0, dom, 3)] if len(dom) else np.zeros((0, 3), dtype=np.uint8)
        poly = polyline_from_nan_joined(
            state["lines"], {"k": state["k"], "dominant_rgb": rgb}, bounds=self.view_bounds
        )
        self.lines_poly.copy_from(poly)
        self._line_points = np.asarray(poly.points, dtype=float)
        self._run_starts, self._run_lengths = line_runs(poly) if poly.n_lines else (np.empty(0, int), np.empty(0, int))
        if poly.n_points:
            k = np.asarray(poly.point_data["k"], dtype=float)
            k = k[np.isfinite(k) & (k > 0)]
            if k.size:
                k_lo = max(float(np.percentile(k, 2)), 1e-8)
                k_hi = max(float(np.percentile(k, 99.5)), k_lo * 10.0)  # log scale needs 0 < lo < hi
                self.line_actors[2].mapper.scalar_range = (k_lo, k_hi)
                self.k_bar.set_range(k_lo, k_hi)

        # 3. Wind arrow and the wand rod
        self.plotter.add_mesh(
            self._wind_arrow(state["wind_vector"]), color="#FF9100", name="wind_arrow", smooth_shading=True
        )
        self.plotter.add_mesh(
            self._wand_rod(state["lines"], state["wand_z"]), color="#FFFFFF",
            name="wand_rod", smooth_shading=True,
        )

        # 4. HUD
        n_wand = len(self.state_mgr.wand_heights)
        self.lbl_wand.setText(f"SMOKE WAND: step {state['wand_index'] + 1}/{n_wand} (z = {state['wand_z']:.2f} m)")
        self.lbl_direction.setText(
            f"{self.cfd_cfg.get('direction_label', 'Wind direction')}: {state['direction_deg']:.1f}°"
        )
        self.lbl_speed.setText(f"WIND SPEED: {state['wind_speed']:.0f} m/s (fixed)")

        self.ns_panel.set_terms(state["ns_terms"])

        def m(key: str) -> float:
            v = metrics.get(key, 0.0)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

        self.lbl_re.setText(f"Reynolds number Re = VD/ν = {m('reynolds_number'):.3g}")
        self.lbl_forces.setText(f"Drag {m('drag_force_n'):,.0f} N   ·   Lift {m('lift_force_n'):+,.0f} N")
        self.lbl_coeffs.setText(
            f"Cd = {m('drag_coefficient_cd'):.3f}   Cl = {m('lift_coefficient_cl'):+.3f}   "
            f"A = {m('frontal_area_m2'):.2f} m²"
        )
        self.lbl_power.setText(f"Drag power P = ½ρCdAV³ = {m('drag_power_kw'):,.1f} kW")
        converged = metrics.get("converged")
        self.lbl_run.setText(
            f"OpenFOAM k-ω SST, {int(m('iterations'))} iterations, "
            + ("converged" if converged else "not fully converged (time-averaged)")
        )

        if state["filled"]:
            src = metrics.get("filled_from")
            where = ""
            if isinstance(src, (int, float)) and 0 <= int(src) < len(self.state_mgr.directions_deg):
                where = f" ({self.state_mgr.directions_deg[int(src)]:g}°)"
            self.lbl_filled.setText(f"Nearest simulated direction shown{where}: this angle is not solved yet.")
            self.lbl_filled.show()
        else:
            self.lbl_filled.hide()

        self.overlay.set_metrics(metrics)
        self.overlay.adjustSize()
        self.overlay.raise_()

        self.plotter.render()

    def on_view_changed(self, view: int) -> None:
        """Switch all actors and scalar bars to `view` (0 pressure, 1 dominant term, 2 turbulence k)."""
        self.lbl_view.setText(VIEW_NAMES[view % len(VIEW_NAMES)])
        cyan, dominant, kmap = self.line_actors
        cyan.SetVisibility(view == 0)
        dominant.SetVisibility(view == 1)
        kmap.SetVisibility(view == 2)

        # The model carries pressure colours only in view 0; elsewhere it is grey so the single
        # colour scale on screen belongs to the streamlines.
        self.mesh_actor.mapper.scalar_visibility = view == 0
        self.mesh_actor.prop.color = MODEL_GREY
        self.pressure_bar.setVisible(view == 0)
        self.k_bar.setVisible(view == 2)
        self.term_legend.setVisible(view == 1)
        self._layout_overlays()
        self.plotter.render()

    def _layout_overlays(self) -> None:
        """Place the floating widgets on the 3D view: cards top-left, colour bar / legend bottom-centre."""
        pw, ph = self.plotter.width(), self.plotter.height()
        self.overlay.setFixedWidth(int(min(400, max(340, 0.62 * pw))))
        self.overlay.adjustSize()
        self.overlay.move(14, 14)
        self.overlay.raise_()
        bar_w = int(min(560, max(280, 0.6 * pw)))
        for bar in (self.pressure_bar, self.k_bar):
            bar.setFixedWidth(bar_w)
            bar.move((pw - bar_w) // 2, ph - bar.height() - 14)
            bar.raise_()
        self.term_legend.adjustSize()
        self.term_legend.move(14, ph - self.term_legend.height() - 14)
        self.term_legend.raise_()

    def _apply_responsive_layout(self) -> None:
        """HUD width follows the window; long texts are dropped on short windows rather than squeezed."""
        self.hud.setFixedWidth(int(min(HUD_MAX_WIDTH, max(HUD_MIN_WIDTH, 0.3 * self.width()))))
        compact = self.height() < COMPACT_HEIGHT
        self.lbl_desc.setVisible(not compact)
        self.ns_panel.caption.setVisible(not compact)
        self.lbl_run.setVisible(not compact)
        self.hud_layout.setSpacing(4 if compact else 8)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------------------------------
    def _animate_flow(self) -> None:
        """Slide white tracer dots along the streamlines (linear interpolation between line points)."""
        if not self.isVisible() or len(self._run_starts) == 0:
            return
        self.anim_phase = (self.anim_phase + PHASE_PER_FRAME) % 1.0
        pos = particles_on_lines(self._line_points, self._run_starts, self._run_lengths, self.anim_phase)
        self.particles_poly.copy_from(pv.PolyData(pos.astype(np.float32)))
        self.plotter.render()

    # ------------------------------------------------------------------------------------------
    # Keyboard / wheel -> controller
    # ------------------------------------------------------------------------------------------
    def focusNextPrevChild(self, next_: bool) -> bool:  # noqa: N802 (Qt override)
        """Tab must reach the controller (view switch), never move the keyboard focus."""
        return False

    def _route_key(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Backtab:  # Shift+Tab: same as Tab, the views only cycle one way
            self.controller.cycle_view()
            return True
        return self.controller.handle_key_press(event)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt override)
        """The 3D widget would swallow keys for itself (VTK shortcuts) and the wheel (zoom): route them first."""
        if obj is self.plotter:
            if event.type() == QEvent.KeyPress and self._route_key(event):
                return True
            if event.type() == QEvent.Wheel and self.controller.handle_wheel_event(event):
                return True
            if event.type() == QEvent.Resize:
                self._layout_overlays()
                self._fit_camera()
        return super().eventFilter(obj, event)

    def event(self, event) -> bool:
        # Qt turns Tab into a focus change inside QWidget.event(); take it before that happens.
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Tab, Qt.Key_Backtab):
            if self._route_key(event):
                return True
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if not self._route_key(event):
            super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        if not self.controller.handle_wheel_event(event):
            super().wheelEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.plotter.setFocus()  # so a macropad keystroke has somewhere to land

    def closeEvent(self, event) -> None:  # noqa: N802
        self.anim_timer.stop()
        try:
            self.plotter.close()
        finally:
            super().closeEvent(event)
