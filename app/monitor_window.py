"""
Monitor Window: native PySide6 + PyVista results window for one exhibition object.

One window per object (airplane, F1 car, tower, human). All four share a single
ExhibitionInputController, so a button press or knob turn on the macropad updates every screen.

The window is laid out like a CFD post-processor, kept uncluttered: a white graphics window (like the
A2 poster renders) with two small colour legends, and a light report panel on the right. Space switches
the panel between a simple and a detailed version. Smoke shows at all 10 wand heights: the selected height
in thick lines, the other nine thin.

The picture is static between inputs: it is redrawn only when a control changes or the camera moves,
so the GPU is idle while nobody is touching the exhibit.

The visitor controls
    * the smoke wand: 10 heights (the macropad buttons) - a thin rod across y that releases pathlines;
    * the wind direction: a rotary knob mapped onto the object's direction sweep (angle of attack,
      yaw, wind angle ...). The wind arrow in the 3D scene turns with it;
    * the view (Tab): 0 pathlines by air speed + surface pressure coefficient, 1 pathlines by the
      Navier-Stokes term that dominates, 2 pathlines by turbulence energy k.
The wind speed is fixed per object (it is what OpenFOAM solved), and is shown as such.

Maths link (IGCSE 0606): the camera is aimed at the midpoint of the model's box ((min + max) / 2 on each
axis, coordinate geometry); the wind arrow is a 3D vector (direction cosines of the wind in the model
frame); the tracer dots slide along the lines by linear interpolation; the term bars are logarithmic;
the pressure coefficient Cp = p / (1/2 rho V^2) divides by the dynamic pressure, so every object shares
one colour scale (Cp = 1 where the air is brought to rest, below 0 where it is sucked away).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.hud_widgets import (
    MUTED,
    N_BANDS,
    PANEL_BG,
    RULE,
    TERM_COLORS,
    TERM_LABELS,
    TERM_NAMES,
    TEXT,
    NsTermsPanel,
    SustainabilityOverlay,
    hex_to_rgb,
    polyline_from_nan_joined,
    rainbow_cmap,
    wand_rod_endpoints,
    wind_arrow_geometry,
)
from app.input_controller import ExhibitionInputController
from app.state_manager import ObjectStateManager

VIEW_NAMES = (
    "View 1 of 3 · Air speed and surface pressure",
    "View 2 of 3 · Largest Navier–Stokes term",
    "View 3 of 3 · Turbulence energy k",
)
VIEW_TITLES = (
    "Pathlines coloured by velocity magnitude  ·  Contours of pressure coefficient Cp",
    "Pathlines coloured by the largest term of the Navier–Stokes equation",
    "Pathlines coloured by turbulence kinetic energy k  ·  k-ω SST",
)
SPEED_TITLE = "Velocity magnitude (m/s)"
CP_TITLE = "Pressure coefficient Cp"
K_TITLE = "Turbulence energy k (m²/s²)"
CP_RANGE = (-1.5, 1.0)  # fixed on every screen: 1 = air brought to rest, negative = suction
SPEED_TOP = 1.3  # the speed scale runs from 0 to 1.3 x the wind speed (air speeds up over curved surfaces)
BACKGROUND = "white"  # plain white, like the poster renders
TUBE_FRAC, GHOST_TUBE_FRAC = 0.003, 0.0016  # streamline tube radius as a fraction of the model's longest side
MODEL_GREY = "#B9C0C9"
DARK = "#2E3744"
FONT_FILE = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")  # VTK's own fonts have no Greek letters
DISPLAY_TRIANGLES = 100_000  # enough for a monitor seen from 1-2 m; the full car mesh is 344,000
PANEL_MIN_WIDTH, PANEL_MAX_WIDTH = 340, 520



def display_mesh(full: pv.PolyData) -> tuple[pv.PolyData, np.ndarray]:
    """A lighter copy of a big model for drawing, and for each of its points the nearest original vertex.

    The solver's pressure is stored per original vertex; `pressure[idx]` carries it onto the display copy
    (nearest-neighbour lookup, done once at start-up). Small models are drawn as they are.
    """
    n = full.n_cells
    if n <= DISPLAY_TRIANGLES:
        return full, np.arange(full.n_points)
    try:
        light = full.decimate(1.0 - DISPLAY_TRIANGLES / n)  # quadric decimation keeps the shape's curvature
    except Exception:  # noqa: BLE001 - an odd mesh is still drawable at full detail
        return full, np.arange(full.n_points)
    if light.n_points == 0:
        return full, np.arange(full.n_points)
    from scipy.spatial import cKDTree

    idx = cKDTree(np.asarray(full.points)).query(np.asarray(light.points))[1]
    return light, np.asarray(idx, dtype=np.int64)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0.0 else np.array([1.0, 0.0, 0.0])


def _style_text(prop, size: int, color: str = TEXT, bold: bool = False) -> None:
    """One look for every VTK text: the Unicode font, flat dark ink, no shadow."""
    if FONT_FILE.is_file():
        prop.SetFontFamily(vtk.VTK_FONT_FILE)
        prop.SetFontFile(str(FONT_FILE))
    prop.SetFontSize(int(size))
    prop.SetColor(*(c / 255.0 for c in hex_to_rgb(color)))
    prop.SetBold(bold)
    prop.SetItalic(False)
    prop.ShadowOff()


def panel_style(s: float) -> str:
    """Style sheet of the report panel at scale `s` (1.0 = a 1920 x 1080 monitor)."""
    px = lambda n: max(9, int(round(n * s)))  # noqa: E731
    return f"""
    QWidget#panel {{ background-color: {PANEL_BG}; border-left: 1px solid {RULE}; }}
    QLabel {{ color: {TEXT}; background: transparent; border: none; font-family: Arial; }}
    QLabel[role="title"] {{ font-size: {px(30)}px; font-weight: 700; }}
    QLabel[role="sub"] {{ font-size: {px(17)}px; color: {MUTED}; }}
    QLabel[role="section"] {{ font-size: {px(17)}px; font-weight: 700; padding-top: {px(14)}px; }}
    QFrame[role="rule"] {{ background-color: {RULE}; border: none; }}
    QLabel[role="key"] {{ font-size: {px(16)}px; color: {MUTED}; }}
    QLabel[role="value"] {{ font-size: {px(16)}px; font-weight: 700; }}
    QLabel[role="big"] {{ font-size: {px(21)}px; font-weight: 700; }}
    QLabel[role="small"] {{ font-size: {px(13)}px; color: {MUTED}; }}
    QLabel[role="equation"] {{ font-size: {px(22)}px; font-family: "STIX Two Text", "Times New Roman"; }}
    QLabel[role="term"] {{ font-size: {px(14)}px; font-weight: 700; }}
    QLabel[role="termValue"] {{ font-size: {px(15)}px; }}
    QLabel[role="termValue"][dominant="true"] {{ font-weight: 700; }}
    QLabel[role="warn"] {{ font-size: {px(14)}px; font-weight: 700; color: #7A4B00; background-color: #FFF1CC;
                          padding: {px(6)}px {px(10)}px; }}
    QLabel[role="note"] {{ font-size: {px(13)}px; color: {TEXT}; background-color: #DDE2E8; padding: {px(8)}px; }}
    QLabel[role="console"] {{ font-family: Menlo; font-size: {px(13)}px; color: #D5DBE3; background-color: #1E232A;
                             padding: {px(10)}px {px(14)}px; }}
    """


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
        self.rho = float(self.cfd_cfg.get("density_kg_m3", 1.225))
        self.view = 0

        self.setWindowTitle(
            f"Monitor {self.obj_cfg.get('monitor_index', 0) + 1}: {self.obj_cfg['display_name']} — Navier-Stokes CFD"
        )
        self.resize(1280, 800)
        self.setFocusPolicy(Qt.StrongFocus)

        self.detailed = False

        self._build_ui()
        self._setup_3d_scene()

        # Every key/wheel event that lands on the 3D widget goes through the controller first.
        self.plotter.installEventFilter(self)

        self.controller.state_changed.connect(self.on_state_changed)
        self.controller.view_changed.connect(self.on_view_changed)
        self.controller.detail_changed.connect(self.on_detail_changed)
        self.controller.camera_reset.connect(self.reset_camera)

        # No animation loop: the scene is drawn when something changes, so an idle exhibit costs no GPU.
        # (The timer is kept, stopped, for tools that expect it.)
        self.anim_timer = QTimer(self)

        self.update_view(self.controller.current_wand_index, self.controller.current_knob)
        self.on_view_changed(self.controller.current_view)
        self.on_detail_changed(self.controller.detailed)

    # ------------------------------------------------------------------------------------------
    # Report panel
    # ------------------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet(f"background-color: {PANEL_BG};")
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.plotter = QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plotter.setFocusPolicy(Qt.StrongFocus)
        main_layout.addWidget(self.plotter, stretch=1)

        panel = QWidget()
        panel.setObjectName("panel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        self.panel = panel
        lay = QVBoxLayout(panel)
        self.panel_layout = lay
        lay.setContentsMargins(28, 26, 28, 0)
        lay.setSpacing(4)

        def label(text: str, role: str, wrap: bool = False) -> QLabel:
            lb = QLabel(text)
            lb.setProperty("role", role)
            lb.setWordWrap(wrap)
            return lb

        self._detail_only: list[QWidget] = []  # hidden in the simple panel (Space)

        def section(title: str) -> list[QWidget]:
            head = label(title, "section")
            lay.addWidget(head)
            rule = QFrame()
            rule.setProperty("role", "rule")
            rule.setFixedHeight(1)
            lay.addWidget(rule)
            lay.addSpacing(4)
            return [head, rule]

        lay.addWidget(label(self.obj_cfg.get("display_name", "Object"), "title", wrap=True))
        self.lbl_direction = label("", "sub")
        self.lbl_wand = label("", "sub")
        lay.addWidget(self.lbl_direction)
        lay.addWidget(self.lbl_wand)

        section("Forces")
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(5)
        self.force_values: dict[str, QLabel] = {}
        for i, (key, name) in enumerate((
            ("drag", "Drag"), ("lift", "Lift"), ("cd", "Cd"), ("cl", "Cl"),
            ("area", "Frontal area"), ("re", "Reynolds no. VD/ν"),
        )):
            r, c = divmod(i, 2)
            key_lb = label(name, "key")
            grid.addWidget(key_lb, r, 2 * c)
            val = label("", "value")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(val, r, 2 * c + 1)
            self.force_values[key] = val
            if r > 0:  # drag and lift stay in the simple panel; Cd, Cl, area and Re are detail
                self._detail_only += [key_lb, val]
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay.addLayout(grid)
        power_row = QHBoxLayout()
        power_row.addWidget(label("Drag power  P = ½ρCdAV³", "key"))
        power_row.addStretch()
        self.lbl_power = label("", "big")
        power_row.addWidget(self.lbl_power)
        lay.addLayout(power_row)

        self._detail_only += section("Navier–Stokes terms near the model")
        self.ns_panel = NsTermsPanel()
        lay.addWidget(self.ns_panel)
        self._detail_only.append(self.ns_panel)

        widgets = tuple(self.cfd_cfg.get("widgets", ()))
        self.overlay = SustainabilityOverlay(widgets)
        if widgets:
            section("Wind load on the tower" if "wind_load" in widgets else "Sustainability")
            lay.addWidget(self.overlay)

        # Honest note when this direction was copied from the nearest simulated one
        self.lbl_filled = label("", "warn", wrap=True)
        self.lbl_filled.hide()
        lay.addSpacing(8)
        lay.addWidget(self.lbl_filled)

        # Building info card (earthquake / tsunami scope note)
        self.lbl_info = label(self.obj_cfg.get("info_card", ""), "note", wrap=True)
        self.lbl_info.setVisible(bool(self.obj_cfg.get("info_card")))
        lay.addWidget(self.lbl_info)

        self._detail_only.append(self.lbl_info)
        lay.addStretch()
        self.lbl_mode = label("", "small", wrap=True)
        lay.addWidget(self.lbl_mode)
        self.lbl_run = label("", "console", wrap=True)
        lay.addWidget(self.lbl_run)
        self._detail_only.append(self.lbl_run)

        main_layout.addWidget(panel)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------------------------------
    # 3D scene
    # ------------------------------------------------------------------------------------------
    def _setup_3d_scene(self) -> None:
        pl = self.plotter
        pl.set_background(BACKGROUND)
        pl.enable_anti_aliasing("msaa", multi_samples=4)

        b = self.view_bounds
        v = np.asarray(self.state_mgr.mesh_vertices, dtype=float)
        self._model_lo, self._model_hi = v.min(axis=0), v.max(axis=0)
        self._box_size = b[1::2] - b[0::2]
        self.wind_speed = float(self.state_mgr.wind_speed)
        cmap = rainbow_cmap(N_BANDS)

        # Streamlines are kept from a little upstream to about one model length behind it (like the posters),
        # so the far wake does not crowd the picture, and drawn as real tubes of radius r = fraction x size.
        L = float((self._model_hi - self._model_lo).max())
        self._line_bounds = b.copy()
        self._line_bounds[0] = max(b[0], self._model_lo[0] - 0.35 * L)
        self._line_bounds[1] = min(b[1], self._model_hi[0] + 1.0 * L)
        self._tube_r, self._ghost_r = TUBE_FRAC * L, GHOST_TUBE_FRAC * L

        # The model: Cp contours in view 0, plain grey in views 1 and 2
        f = self.state_mgr.mesh_faces
        faces = np.column_stack([np.full(len(f), 3), f]).ravel()
        self.mesh_poly, self._display_idx = display_mesh(pv.PolyData(np.asarray(v, dtype=np.float32), faces))
        self.mesh_poly.point_data[CP_TITLE] = np.zeros(self.mesh_poly.n_points, dtype=np.float32)
        self.mesh_actor = pl.add_mesh(
            self.mesh_poly, scalars=CP_TITLE, cmap=cmap, clim=CP_RANGE,
            interpolate_before_map=True, smooth_shading=True, show_scalar_bar=False,
            specular=0.25, specular_power=30,
        )

        # Pathlines: one shared PolyData, three actors (by speed / by dominant term / by k)
        dummy = polyline_from_nan_joined(
            np.array([[b[0], b[2], b[4]], [b[1], b[3], b[5]]]),
            {"speed": np.array([0.0, 1.0]), "k": np.array([1.0, 2.0]), "dominant_rgb": np.zeros((2, 3), np.uint8)},
        )
        self.lines_poly = self._tubes(dummy, self._tube_r)
        tube = dict(smooth_shading=True, specular=0.2, show_scalar_bar=False)
        self.line_actors = [
            pl.add_mesh(self.lines_poly, scalars="speed", cmap=cmap,
                        clim=(0.0, SPEED_TOP * self.wind_speed), name="lines_speed", **tube),
            pl.add_mesh(self.lines_poly, scalars="dominant_rgb", rgb=True, name="lines_dominant", **tube),
            pl.add_mesh(self.lines_poly, scalars="k", cmap=cmap, log_scale=True,
                        clim=(1e-2, 1e2), name="lines_k", **tube),
        ]
        self._has_speed = True

        # Every other wand height as thinner tubes, so the flow shows over the whole height of the model
        # while the selected height stays the one that stands out (all ten at once buried the model).
        self.ghost_poly = self._tubes(dummy, self._ghost_r)
        self.ghost_actors = [
            pl.add_mesh(self.ghost_poly, scalars="speed", cmap=cmap,
                        clim=(0.0, SPEED_TOP * self.wind_speed), name="ghost_speed", **tube),
            pl.add_mesh(self.ghost_poly, scalars="dominant_rgb", rgb=True, name="ghost_dominant", **tube),
            pl.add_mesh(self.ghost_poly, scalars="k", cmap=cmap, log_scale=True,
                        clim=(1e-2, 1e2), name="ghost_k", **tube),
        ]
        self._ghost_key = None

        # Wind arrow and wand rod: persistent PolyData, updated in place by update_view
        self._last_lines = None
        self.arrow_poly = self._wind_arrow(np.array([1.0, 0.0, 0.0]))
        self.rod_poly = self._wand_rod(np.empty((0, 3)), float(self.state_mgr.wand_heights[0]))
        pl.add_mesh(self.arrow_poly, color=MUTED, smooth_shading=True, name="wind_arrow", render=False)
        pl.add_mesh(self.rod_poly, color="#A9B1BB", smooth_shading=True, name="wand_rod", render=False)


        # Legends: VTK scalar bars, banded like the surfaces they describe
        self.bars = {}
        for title, mapper, fmt in (
            (SPEED_TITLE, self.line_actors[0].mapper, "%.0f"),
            (CP_TITLE, self.mesh_actor.mapper, "%.1f"),
            (K_TITLE, self.line_actors[2].mapper, "%.3g"),
        ):
            # horizontal and short, just the two end labels, so none lands on the title
            pl.add_scalar_bar(title=title, mapper=mapper, vertical=False, n_labels=2, fmt=fmt)
            bar = pl.scalar_bars[title]
            bar.SetUnconstrainedFontSize(True)
            bar.SetDrawFrame(False)
            bar.SetTextPositionToPrecedeScalarBar()
            bar.SetTextPad(8)
            self.bars[title] = bar
        # Cp ticks at the exact ends of the fixed scale
        ticks = vtk.vtkDoubleArray()
        for t in CP_RANGE:
            ticks.InsertNextValue(float(t))
        self.bars[CP_TITLE].SetUseCustomLabels(True)
        self.bars[CP_TITLE].SetCustomLabels(ticks)

        # Titles in the graphics window, Fluent style: what is shown, then the controls
        self.txt_title = vtk.vtkTextActor()
        self.txt_view = vtk.vtkTextActor()
        self.txt_hint = vtk.vtkTextActor()
        self.txt_terms = {n: vtk.vtkTextActor() for n in TERM_NAMES}
        for name, actor in self.txt_terms.items():
            actor.SetInput("■  " + TERM_LABELS[name])
        hint = ("Buttons 1–0: smoke wand  ·  Knob: wind direction  ·  Tab: change view  ·  Space: more / less detail\n"
                "Drag: turn  ·  Pinch or ⌘ + scroll: zoom  ·  R: reset view")
        if self.remote_url:
            hint += f"\nRemote control: {self.remote_url}"
        self.txt_hint.SetInput(hint)
        for actor in (self.txt_title, self.txt_view, self.txt_hint, *self.txt_terms.values()):
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            pl.renderer.AddViewProp(actor)

        self._fit_camera()

    def _focus_bounds(self) -> np.ndarray:
        """The model's box plus room for the wake behind it, kept inside the display box."""
        lo, hi = self._model_lo, self._model_hi
        L = float((hi - lo).max())
        f = np.array([lo[0] - 0.2 * L, hi[0] + 0.7 * L, lo[1] - 0.25 * L, hi[1] + 0.25 * L, lo[2], hi[2] + 0.15 * L])
        b = self.view_bounds
        f[0::2] = np.maximum(f[0::2], b[0::2])
        f[1::2] = np.minimum(f[1::2], b[1::2])
        return f

    def _fit_camera(self) -> None:
        """Aim at the midpoint of the model-and-wake box from upstream, to the side and above."""
        f = self._focus_bounds()
        centre = (f[0::2] + f[1::2]) / 2.0  # midpoint on each axis
        diag = float(np.linalg.norm(f[1::2] - f[0::2]))
        eye = centre + diag * np.array([-0.62, -0.95, 0.42])
        self.plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0)]
        self.plotter.reset_camera(bounds=tuple(f))
        # a tall, narrow view (aspect < 1.3) is limited by its width, so zoom in less
        aspect = self.plotter.width() / max(self.plotter.height(), 1)
        self.plotter.camera.zoom(1.2 * min(1.0, aspect / 1.3))

    @staticmethod
    def _tubes(lines: pv.PolyData, radius: float) -> pv.PolyData:
        """Line cells as tubes of the given radius (8 sides), keeping the point data used for colouring."""
        if lines.n_lines == 0:
            return lines
        return lines.tube(radius=radius, n_sides=8, capping=False)

    def _wind_arrow(self, wind_vector: np.ndarray) -> pv.PolyData:
        """Arrow just upstream of the model, pointing along the wind (geometry: hud_widgets).

        It is placed on the upstream edge of the model-and-wake box, not the whole tunnel box, so it stays
        a modest size in front of the camera."""
        centre = (self._model_lo + self._model_hi) / 2.0
        start, d, length = wind_arrow_geometry(wind_vector, centre, self._focus_bounds(), length_frac=0.14)
        return pv.Arrow(start=start, direction=d, scale=length, tip_length=0.3, tip_radius=0.07, shaft_radius=0.018)

    def _wand_rod(self, lines: np.ndarray, z: float) -> pv.PolyData:
        """Thin rod across the wand's seed points, where the drawn smoke starts."""
        a, e = wand_rod_endpoints(lines, z, self._line_bounds)
        a[0] = e[0] = max(a[0], self._line_bounds[0])  # the seeds sit further upstream than the drawn lines start
        return pv.Line(a, e).tube(radius=0.0012 * float((self._model_hi - self._model_lo).max()))

    # ------------------------------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------------------------------
    def on_state_changed(self, wand_index: int, knob: float) -> None:
        self.update_view(wand_index, knob)

    def update_view(self, wand_index: int, knob: float) -> None:
        """Look up the state for (wand, knob) in RAM and push it to the 3D scene and the report panel."""
        state = self.state_mgr.get_state(wand_index, knob)
        metrics = state["metrics"]

        # 1. Surface pressure as Cp = p / (1/2 rho V^2) on the fixed scale CP_RANGE
        q = 0.5 * self.rho * self.wind_speed**2  # dynamic pressure, Pa
        p = np.asarray(state["surface_pressure"], dtype=np.float32)
        self.mesh_poly.point_data[CP_TITLE] = p[self._display_idx] / np.float32(q)

        # 2. Pathlines, k range and the wand rod only change with (wand, nearest direction); the state
        #    manager hands back the very same array object in that case, so an identity check is enough.
        if state["lines"] is not self._last_lines:
            self._last_lines = state["lines"]
            dom = np.asarray(state["dominant"])
            palette = np.array([hex_to_rgb(TERM_COLORS[n]) for n in TERM_NAMES] + [(255, 255, 255)], dtype=np.uint8)
            rgb = palette[np.where(dom >= 0, dom, 3)] if len(dom) else np.zeros((0, 3), dtype=np.uint8)
            speed = state.get("speed")
            self._has_speed = speed is not None
            if speed is None:  # a cache imported before speed was stored: draw the lines in one dark colour
                speed = np.zeros(len(state["lines"]), dtype=np.float32)
            poly = polyline_from_nan_joined(
                state["lines"], {"speed": speed, "k": state["k"], "dominant_rgb": rgb}, bounds=self._line_bounds
            )
            self.lines_poly.copy_from(self._tubes(poly, self._tube_r))
            if poly.n_points:
                k = np.asarray(poly.point_data["k"], dtype=float)
                k = k[np.isfinite(k) & (k > 0)]
                if k.size:
                    k_lo = max(float(np.percentile(k, 2)), 1e-8)
                    k_hi = max(float(np.percentile(k, 99.5)), k_lo * 10.0)  # log scale needs 0 < lo < hi
                    self.line_actors[2].mapper.scalar_range = (k_lo, k_hi)
                    self.ghost_actors[2].mapper.scalar_range = (k_lo, k_hi)  # one k scale for every height
            self.rod_poly.copy_from(self._wand_rod(state["lines"], state["wand_z"]))
            self._apply_view_visibility()

        # 2b. The other nine heights, rebuilt only when the solved direction or the selected height changes
        key = (state["direction_index"], state["wand_index"])
        if key != self._ghost_key:
            self._ghost_key = key
            self._update_ghosts(*key)

        # 3. Wind arrow: same shape every time, so its points are updated in place (no new actor, no render).
        self.arrow_poly.copy_from(self._wind_arrow(state["wind_vector"]))

        # 4. Report panel
        n_wand = len(self.state_mgr.wand_heights)
        label = self.cfd_cfg.get("direction_label", "Wind direction")
        self.lbl_direction.setText(f"{state['wind_speed']:.0f} m/s (fixed)  ·  {label} {state['direction_deg']:.1f}°")
        self.lbl_wand.setText(f"Smoke wand {state['wand_index'] + 1} of {n_wand}  ·  z = {state['wand_z']:.2f} m")

        def m(key: str) -> float:
            v = metrics.get(key, 0.0)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

        fv = self.force_values
        fv["drag"].setText(f"{m('drag_force_n'):,.0f} N")
        fv["lift"].setText(f"{m('lift_force_n'):+,.0f} N")
        fv["cd"].setText(f"{m('drag_coefficient_cd'):.3f}")
        fv["cl"].setText(f"{m('lift_coefficient_cl'):+.3f}")
        fv["area"].setText(f"{m('frontal_area_m2'):.2f} m²")
        fv["re"].setText(f"{m('reynolds_number'):.2e}")
        power = m("drag_power_kw")
        self.lbl_power.setText(f"{power / 1000:,.2f} MW" if power >= 1000 else f"{power:,.1f} kW")

        self.ns_panel.set_terms(state["ns_terms"])

        converged = metrics.get("converged")
        self.lbl_run.setText(
            f"> simpleFoam  k-ω SST\n  {int(m('iterations'))} iterations, "
            + ("converged" if converged else "not fully converged: time-averaged fields")
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

        self.overlay.set_metrics(metrics, state.get("worst_direction_deg"))
        self.plotter.render()

    def _update_ghosts(self, direction_index: int, selected: int) -> None:
        """Join the lines of every other wand height (counting from the selected one) into the thin tubes."""
        palette = np.array([hex_to_rgb(TERM_COLORS[n]) for n in TERM_NAMES] + [(160, 160, 160)], dtype=np.uint8)
        sep_pt, pts, speed, k, rgb = np.full((1, 3), np.nan), [], [], [], []
        for j, (lines, kk, dom, sp) in enumerate(self.state_mgr.wand_sets(direction_index)):
            if j == selected or (j - selected) % 2 or len(lines) == 0:
                continue
            n = len(lines)
            dom = np.asarray(dom)
            pts += [np.asarray(lines, dtype=float).reshape(-1, 3), sep_pt]
            speed += [np.zeros(n) if sp is None else np.asarray(sp, dtype=float), [np.nan]]
            k += [np.asarray(kk, dtype=float), [np.nan]]
            rgb += [palette[np.where(dom >= 0, dom, 3)] if len(dom) else np.zeros((n, 3), np.uint8),
                    np.zeros((1, 3), np.uint8)]
        if not pts:
            for a in self.ghost_actors:
                a.SetVisibility(False)
            return
        poly = polyline_from_nan_joined(
            np.concatenate(pts),
            {"speed": np.concatenate(speed), "k": np.concatenate(k), "dominant_rgb": np.concatenate(rgb)},
            bounds=self._line_bounds,
        )
        if poly.n_points:
            self.ghost_poly.copy_from(self._tubes(poly, self._ghost_r))
            self._ghost_ok = True
        else:
            self._ghost_ok = False
        self._apply_view_visibility()

    def on_detail_changed(self, detailed: bool) -> None:
        """Simple panel: name, state, drag, lift, power and the sustainability story. Detailed: everything."""
        self.detailed = bool(detailed)
        for w in self._detail_only:
            w.setVisible(self.detailed)
        if not self.obj_cfg.get("info_card"):
            self.lbl_info.setVisible(False)
        self.lbl_mode.setText("Press Space for less detail" if self.detailed else "Press Space for the Navier–Stokes terms and more")

    def reset_camera(self) -> None:
        self._fit_camera()
        self.plotter.render()

    def _zoom(self, factor: float) -> None:
        """Move the camera towards (factor > 1) or away from what it looks at."""
        self.plotter.camera.zoom(min(max(factor, 0.5), 2.0))
        self.plotter.render()

    def on_view_changed(self, view: int) -> None:
        """Switch all actors and legends to `view` (0 speed + Cp, 1 dominant term, 2 turbulence k)."""
        self.view = view % len(VIEW_NAMES)
        self.txt_title.SetInput(VIEW_TITLES[self.view])
        self.txt_view.SetInput(VIEW_NAMES[self.view])
        self._apply_view_visibility()
        self._layout_overlays()
        self.plotter.render()

    def _apply_view_visibility(self) -> None:
        view = self.view
        speed, dominant, kmap = self.line_actors
        speed.SetVisibility(view == 0)
        dominant.SetVisibility(view == 1)
        kmap.SetVisibility(view == 2)
        ghost_ok = getattr(self, "_ghost_ok", False)
        for i, actor in enumerate(getattr(self, "ghost_actors", [])):
            actor.SetVisibility(ghost_ok and view == i)
        if getattr(self, "ghost_actors", None) and view == 0 and not self._has_speed:
            self.ghost_actors[0].mapper.scalar_visibility = False
            self.ghost_actors[0].prop.color = DARK
        if view == 0 and not self._has_speed:  # no speed stored: plain dark pathlines, no speed legend
            speed.mapper.scalar_visibility = False
            speed.prop.color = DARK
        else:
            speed.mapper.scalar_visibility = True
        # The model carries Cp colours only in view 0; elsewhere it is grey so the single colour scale
        # on screen belongs to the pathlines.
        self.mesh_actor.mapper.scalar_visibility = view == 0
        self.mesh_actor.prop.color = MODEL_GREY
        self.bars[SPEED_TITLE].SetVisibility(view == 0 and self._has_speed)
        self.bars[CP_TITLE].SetVisibility(view == 0)
        self.bars[K_TITLE].SetVisibility(view == 2)
        for actor in self.txt_terms.values():
            actor.SetVisibility(view == 1)

    def _layout_overlays(self) -> None:
        """Place legends and titles in the graphics window; text sizes follow the window height."""
        h = max(self.plotter.height(), 1)
        w = max(self.plotter.width(), 1)
        title_px, label_px = max(10, int(0.019 * h)), max(9, int(0.0155 * h))
        self.txt_title.SetVisibility(False)  # one short line is enough: the view name
        _style_text(self.txt_view.GetTextProperty(), max(11, int(0.021 * h)), TEXT, bold=True)
        self.txt_view.SetPosition(0.018, 0.945)
        _style_text(self.txt_hint.GetTextProperty(), max(9, int(0.0145 * h)), MUTED)
        self.txt_hint.GetTextProperty().SetJustificationToRight()
        self.txt_hint.GetTextProperty().SetVerticalJustificationToBottom()
        self.txt_hint.SetPosition(0.985, 0.018)

        # Small horizontal legends side by side under the view name: wide enough to read, never a column.
        shown = [t for t in (CP_TITLE, SPEED_TITLE) if self.bars[t].GetVisibility()] or [K_TITLE]
        bw, bh = min(0.24, 340.0 / w), max(0.075, 70.0 / h)
        for title in (SPEED_TITLE, CP_TITLE, K_TITLE):
            bar = self.bars[title]
            slot = shown.index(title) if title in shown else 0
            bar.SetPosition(0.018 + slot * (bw + 0.03), 0.92 - bh - 0.012)
            bar.SetPosition2(bw, bh)
            bar.SetBarRatio(0.28)
            bar.SetMaximumWidthInPixels(int(bw * w))
            _style_text(bar.GetTitleTextProperty(), title_px, TEXT, bold=True)
            _style_text(bar.GetLabelTextProperty(), label_px, TEXT)
            bar.SetVerticalTitleSeparation(int(0.9 * label_px))
            bar.GetTitleTextProperty().SetJustificationToLeft()

        for i, name in enumerate(TERM_NAMES):
            actor = self.txt_terms[name]
            _style_text(actor.GetTextProperty(), label_px, TERM_COLORS[name], bold=True)
            actor.SetPosition(0.018, 0.88 - i * 0.04)

    def _apply_responsive_layout(self) -> None:
        """Panel width and text size follow the window (1.0 = a 1920 x 1080 monitor)."""
        s = min(self.width() / 1920.0, self.height() / 1080.0)
        s = min(max(s, 0.5), 1.4)
        self.panel.setFixedWidth(int(min(PANEL_MAX_WIDTH, max(PANEL_MIN_WIDTH, 0.27 * self.width()))))
        margin = int(28 * s)
        self.panel_layout.setContentsMargins(margin, int(26 * s), margin, 0)
        self.panel.setStyleSheet(panel_style(s))
        # On short windows the explanatory caption under the equation is dropped rather than squeezed
        self.ns_panel.caption.setVisible(self.height() >= 820)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------------------------------
    def _animate_flow(self) -> None:
        """No moving tracers any more (they kept the GPU busy all the time); just draw the current frame."""
        if self.isVisible():
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
            if event.type() == QEvent.Wheel and event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
                self._zoom(1.1 ** (event.angleDelta().y() / 120.0))  # Cmd + scroll zooms; plain scroll is the knob
                return True
            if event.type() == QEvent.NativeGesture and event.gestureType() == Qt.ZoomNativeGesture:
                self._zoom(1.0 + event.value())  # trackpad pinch
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
