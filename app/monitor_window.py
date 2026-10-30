"""
Monitor Window: native PySide6 + PyVista results window for one exhibition object.

One window per object (airplane, F1 car, tower). All of them share a single
ExhibitionInputController, so a button press or knob turn on the macropad updates every screen.

The window has two parts around the graphics: a top bar with the object's name and one tab per view
(clicking a tab, Tab on the keyboard, the macropad or the iPad all switch the same view on every screen),
and a sidebar with the three terms of the Navier-Stokes equation and their current sizes. Nothing else:
no captions, no explanations. Smoke shows at all 10 wand heights: the selected height in thick lines,
the other nine thin.

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

import math
import time
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QWheelEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.hud_widgets import (
    MUTED,
    N_BANDS,
    PANEL_BG,
    RULE,
    TERM_COLORS,
    TERM_NAMES,
    TEXT,
    NsTermsPanel,
    hex_to_rgb,
    polyline_from_nan_joined,
    rainbow_cmap,
    wand_rod_endpoints,
    wind_arrow_geometry,
)
from app.input_controller import ExhibitionInputController
from app.state_manager import ObjectStateManager
from app.theme import THEMES, Theme

VIEW_TABS = ("Velocity & pressure", "Navier–Stokes", "Turbulence")  # one tab per view, in Tab-key order
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
PANEL_MIN_WIDTH, PANEL_MAX_WIDTH = 320, 460
ORBIT_SECONDS = 60.0       # the camera circles the model once a minute, always
ORBIT_TICK_MS = 33         # about 30 frames a second while it moves
ORBIT_PAUSE_SECONDS = 8.0  # a visitor dragging or zooming holds the camera still, then it carries on



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


FONT_STACK = '".AppleSystemUIFont", "Helvetica Neue", Arial'


def panel_style(s: float, theme: Theme) -> str:
    """Style sheet of the top bar and the sidebar at scale `s` (1.0 = a 1920 x 1080 monitor)."""
    px = lambda n: max(9, int(round(n * s)))  # noqa: E731
    tints = "\n".join(
        f'QFrame[role="nsRow"][term="{n}"][dominant="true"] {{ background-color: {c}; }}' for n, c in theme.term_tints.items()
    )
    return f"""
    QWidget {{ font-family: {FONT_STACK}; }}
    QWidget#topbar {{ background-color: {theme.topbar}; border-bottom: 1px solid {theme.rule}; }}
    QLabel[role="name"] {{ color: {theme.text}; background: transparent; font-size: {px(23)}px; font-weight: 700; }}
    QPushButton[role="tab"] {{ color: {theme.muted}; background-color: transparent; border: none; border-radius: {px(9)}px;
                              font-size: {px(18)}px; font-weight: 600; padding: {px(9)}px {px(20)}px; }}
    QPushButton[role="tab"]:hover {{ background-color: {theme.tab_hover}; color: {theme.text}; }}
    QPushButton[role="tab"][active="true"] {{ background-color: {theme.tab_on_bg}; color: {theme.tab_on_fg}; }}
    QWidget#panel {{ background-color: {theme.panel}; border-left: 1px solid {theme.rule}; }}
    QFrame[role="nsRow"] {{ background-color: transparent; border-radius: {px(14)}px; }}
    {tints}
    QLabel[role="math"] {{ background: transparent; }}
    QLabel[role="nsValue"] {{ color: {theme.text}; background: transparent; font-size: {px(36)}px; font-weight: 600; }}
    QLabel[role="nsValue"][dominant="true"] {{ font-weight: 800; }}
    QLabel[role="nsUnit"] {{ color: {theme.muted}; background: transparent; font-size: {px(16)}px; padding: 0 0 0 {px(4)}px; }}
    """


class MonitorWindow(QMainWindow):
    def __init__(
        self,
        obj_id: str,
        input_controller: ExhibitionInputController,
        cache_dir: str = "assets/cache",
        remote_url: str | None = None,
        orbit_seconds: float = ORBIT_SECONDS,
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

        self.setWindowTitle(f"NavFlow: {self.obj_cfg['display_name']}")
        self.theme: Theme = THEMES.get(getattr(input_controller, "theme", "light"), THEMES["light"])
        self.resize(1280, 800)
        self.setFocusPolicy(Qt.StrongFocus)

        self.orbit_seconds = float(orbit_seconds)  # 0 holds the camera still

        self._build_ui()
        self._setup_3d_scene()

        # Every key/wheel event that lands on the 3D widget goes through the controller first.
        self.plotter.installEventFilter(self)

        self.controller.state_changed.connect(self.on_state_changed)
        self.controller.view_changed.connect(self.on_view_changed)
        self.controller.camera_reset.connect(self.reset_camera)
        self.controller.theme_changed.connect(self.apply_theme)

        # The camera circles the model all the time (the one thing that moves while nobody is touching the exhibit).
        self._orbit_t = time.monotonic()
        self._orbit_pause_until = 0.0
        self.orbit_timer = QTimer(self)
        self.orbit_timer.setInterval(ORBIT_TICK_MS)
        self.orbit_timer.timeout.connect(self._orbit_tick)
        if self.orbit_seconds > 0:
            self.orbit_timer.start()

        self.anim_timer = QTimer(self)  # kept, stopped, for tools that expect it

        self.apply_theme(self.theme.name)
        self.update_view(self.controller.current_wand_index, self.controller.current_knob)
        self.on_view_changed(self.controller.current_view)

    # ------------------------------------------------------------------------------------------
    # Report panel
    # ------------------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central.setStyleSheet(f"background-color: {self.theme.panel};")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar: the object's name, then one tab per view
        bar = QWidget()
        bar.setObjectName("topbar")
        bar.setAttribute(Qt.WA_StyledBackground, True)
        self.topbar = bar
        bar_lay = QHBoxLayout(bar)
        bar_lay.setContentsMargins(28, 0, 28, 0)
        bar_lay.setSpacing(6)
        self.top_layout = bar_lay
        name = QLabel(self.obj_cfg.get("display_name", "Object"))
        name.setProperty("role", "name")
        bar_lay.addWidget(name)
        bar_lay.addSpacing(28)
        self.tabs: list[QPushButton] = []
        for i, title in enumerate(VIEW_TABS):
            tab = QPushButton(title.replace("&", "&&"))  # a single & would mark a keyboard shortcut
            tab.setProperty("role", "tab")
            tab.setProperty("active", i == 0)
            tab.setFocusPolicy(Qt.NoFocus)  # keys keep going to the controller, never to a button
            tab.setCursor(Qt.PointingHandCursor)
            tab.clicked.connect(lambda _=False, v=i: self.controller.set_view(v))
            bar_lay.addWidget(tab)
            self.tabs.append(tab)
        bar_lay.addStretch(1)
        outer.addWidget(bar)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        outer.addLayout(body, stretch=1)

        self.plotter = QtInteractor(self)
        self.plotter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plotter.setFocusPolicy(Qt.StrongFocus)
        body.addWidget(self.plotter, stretch=1)

        # Sidebar: the equation and the current size of each of its three terms
        panel = QWidget()
        panel.setObjectName("panel")
        panel.setAttribute(Qt.WA_StyledBackground, True)
        self.panel = panel
        lay = QVBoxLayout(panel)
        self.panel_layout = lay
        lay.setContentsMargins(24, 28, 24, 24)
        lay.setSpacing(0)
        self.ns_panel = NsTermsPanel()
        lay.addWidget(self.ns_panel)
        body.addWidget(panel)
        self._apply_responsive_layout()

    # ------------------------------------------------------------------------------------------
    # 3D scene
    # ------------------------------------------------------------------------------------------
    def _setup_3d_scene(self) -> None:
        pl = self.plotter
        pl.set_background(self.theme.bg3d)
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
        pl.add_mesh(self.arrow_poly, color=self.theme.arrow, smooth_shading=True, name="wind_arrow", render=False)
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
            bar.SetTitle("")  # the title is drawn by our own text actor above the bar (see _layout_overlays)
            self.bars[title] = bar
        self.bar_titles = {}
        for title in self.bars:
            actor = vtk.vtkTextActor()
            actor.SetInput(title)
            actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            pl.renderer.AddViewProp(actor)
            self.bar_titles[title] = actor
        # Cp ticks at the exact ends of the fixed scale
        ticks = vtk.vtkDoubleArray()
        for t in CP_RANGE:
            ticks.InsertNextValue(float(t))
        self.bars[CP_TITLE].SetUseCustomLabels(True)
        self.bars[CP_TITLE].SetCustomLabels(ticks)

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

    def _fit_distance(self, d: np.ndarray) -> float:
        """How far the camera must be, looking along `d` (from the focus to the camera), for the model-and-wake
        box to fill the window without leaving it.

        Seen from direction d, the box's half-width on screen is hw = ex |r_x| + ey |r_y| where r is the screen's
        horizontal axis and (ex, ey) the box's half sizes: a long box seen side-on is wide, end-on it is narrow.
        With half field-of-view angles a_v (up-down) and a_h = atan(tan(a_v) x aspect) (left-right), a sphere of
        radius r just fits at distance r / sin(a), so the distance is the larger of hw / sin(a_h) and
        (half height + the pitch's share of the footprint) / sin(a_v)."""
        f = self._focus_bounds()
        centre = (f[0::2] + f[1::2]) / 2.0
        lb = self._line_bounds  # the drawn streamlines can reach further than the focus box along the wind
        lo, hi = np.minimum(f[0::2], lb[0::2]), np.maximum(f[1::2], lb[1::2])
        half = np.maximum(hi - centre, centre - lo)
        half[1:] = 0.5 * (f[3::2] - f[2::2])  # across the wind and up, the focus box is what is drawn
        a_v = math.radians(self.plotter.camera.GetViewAngle()) / 2.0
        aspect = max(self.plotter.width(), 1) / max(self.plotter.height(), 1)
        a_h = math.atan(math.tan(a_v) * aspect)
        flat = math.hypot(d[0], d[1])
        r = np.array([-d[1], d[0]]) / max(flat, 1e-9)  # the screen's horizontal axis, in the ground plane
        half_w = half[0] * abs(r[0]) + half[1] * abs(r[1])
        depth = half[0] * abs(d[0]) + half[1] * abs(d[1])  # how far the footprint reaches along the view
        need_v = half[2] + depth * d[2]
        return 1.08 * max(half_w / math.sin(a_h), need_v / math.sin(a_v))

    def _fit_camera(self) -> None:
        """Aim at the midpoint of the model-and-wake box at a fixed, shallow angle (about 10 degrees above
        level, never changing) and back off just far enough to fill the window."""
        f = self._focus_bounds()
        centre = (f[0::2] + f[1::2]) / 2.0  # midpoint on each axis
        direction = _unit(np.array([-0.62, -0.95, 0.20]))
        eye = centre + self._fit_distance(direction) * direction
        self.plotter.camera_position = [tuple(eye), tuple(centre), (0.0, 0.0, 1.0)]
        self.plotter.renderer.ResetCameraClippingRange()

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
            palette = np.array([hex_to_rgb(self.theme.term_colors[n]) for n in TERM_NAMES] + [(255, 255, 255)], dtype=np.uint8)
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

        # 4. Sidebar: the three Navier-Stokes terms of this state
        self.ns_panel.set_terms(state["ns_terms"])
        self.plotter.render()

    def _orbit_tick(self) -> None:
        """Turn the model like a turntable: the camera circles the upright axis through the point it looks at,
        azimuth growing by 360 degrees every `orbit_seconds`. The height and the pitch never change."""
        now = time.monotonic()
        dt = min(now - self._orbit_t, 0.25)  # a stalled frame must not make the camera jump
        self._orbit_t = now
        if now < self._orbit_pause_until or not self.isVisible():
            return
        cam = self.plotter.camera
        cam.Azimuth(360.0 * dt / self.orbit_seconds)
        # the box is long: end-on it is narrow, side-on wide, so the camera glides in and out to keep it filling the window
        focus = np.array(cam.GetFocalPoint())
        d = np.array(cam.GetPosition()) - focus
        d /= np.linalg.norm(d)
        cam.SetPosition(*(focus + self._fit_distance(d) * d))
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    def apply_theme(self, name: str) -> None:
        """Switch every colour to the light or dark theme: graphics background, bars, text, model and lines."""
        self.theme = THEMES.get(name, THEMES["light"])
        t = self.theme
        self.centralWidget().setStyleSheet(f"background-color: {t.panel};")
        pl = self.plotter
        pl.set_background(t.bg3d)
        for actor_name, colour in (("wind_arrow", t.arrow), ("wand_rod", t.rod)):
            actor = pl.actors.get(actor_name)
            if actor is not None:
                actor.prop.color = colour
        self.ns_panel.set_theme(t)
        self._apply_responsive_layout()
        self._last_lines, self._ghost_key = None, None  # the streamline colours come from the theme: rebuild them
        self._apply_view_visibility()
        self._layout_overlays()
        self.update_view(self.controller.current_wand_index, self.controller.current_knob)

    def _update_ghosts(self, direction_index: int, selected: int) -> None:
        """Join the lines of every other wand height (counting from the selected one) into the thin tubes."""
        palette = np.array([hex_to_rgb(self.theme.term_colors[n]) for n in TERM_NAMES] + [(160, 160, 160)], dtype=np.uint8)
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

    def reset_camera(self) -> None:
        self._fit_camera()
        self.plotter.render()

    def _zoom(self, factor: float) -> None:
        """Move the camera towards (factor > 1) or away from what it looks at."""
        self.plotter.camera.zoom(min(max(factor, 0.5), 2.0))
        self.plotter.render()

    def on_view_changed(self, view: int) -> None:
        """Switch all actors and legends to `view` (0 speed + Cp, 1 dominant term, 2 turbulence k)."""
        self.view = view % len(VIEW_TABS)
        for i, tab in enumerate(self.tabs):
            tab.setProperty("active", i == self.view)
            tab.style().unpolish(tab)
            tab.style().polish(tab)
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
            self.ghost_actors[0].prop.color = self.theme.lines_plain
        if view == 0 and not self._has_speed:  # no speed stored: plain dark pathlines, no speed legend
            speed.mapper.scalar_visibility = False
            speed.prop.color = self.theme.lines_plain
        else:
            speed.mapper.scalar_visibility = True
        # The model carries Cp colours only in view 0; elsewhere it is grey so the single colour scale
        # on screen belongs to the pathlines.
        self.mesh_actor.mapper.scalar_visibility = view == 0
        self.mesh_actor.prop.color = self.theme.model_grey
        self.bars[SPEED_TITLE].SetVisibility(view == 0 and self._has_speed)
        self.bars[CP_TITLE].SetVisibility(view == 0)
        self.bars[K_TITLE].SetVisibility(view == 2)
        for title, actor in self.bar_titles.items():
            actor.SetVisibility(bool(self.bars[title].GetVisibility()))

    def _layout_overlays(self) -> None:
        """Place the colour legends in the graphics window; their text size follows the window height."""
        h = max(self.plotter.height(), 1)
        w = max(self.plotter.width(), 1)
        title_px, label_px = max(10, int(0.019 * h)), max(9, int(0.0155 * h))

        # Small horizontal legends side by side at the top left: a title above each bar, the tick numbers
        # under it. Each slot is as wide as its title needs (about 0.55 em per character).
        shown = [t for t in (CP_TITLE, SPEED_TITLE) if self.bars[t].GetVisibility()] or [K_TITLE]
        bw, bh = min(0.24, 340.0 / w), max(0.07, 62.0 / h)
        x, top = 0.022, 0.982
        for title in self.bars:
            actor = self.bar_titles[title]
            _style_text(actor.GetTextProperty(), title_px, self.theme.text, bold=True)
            actor.GetTextProperty().SetJustificationToLeft()
            actor.GetTextProperty().SetVerticalJustificationToTop()
        for title in shown:
            bar, actor = self.bars[title], self.bar_titles[title]
            slot = max(bw, 0.55 * title_px * len(title) / w)
            actor.SetPosition(x, top)
            bar.SetPosition(x, top - 1.5 * title_px / h - bh)
            bar.SetPosition2(bw, bh)
            bar.SetBarRatio(0.40)
            bar.SetTitleRatio(0.0)
            bar.SetMaximumWidthInPixels(int(bw * w))
            _style_text(bar.GetLabelTextProperty(), label_px, self.theme.text)
            x += slot + 0.03

    def _apply_responsive_layout(self) -> None:
        """Bar and sidebar sizes follow the window (1.0 = a 1920 x 1080 monitor)."""
        s = min(self.width() / 1920.0, self.height() / 1080.0)
        s = min(max(s, 0.5), 1.4)
        self.panel.setFixedWidth(int(min(PANEL_MAX_WIDTH, max(PANEL_MIN_WIDTH, 0.25 * self.width()))))
        self.topbar.setFixedHeight(int(66 * s))
        margin = int(28 * s)
        self.top_layout.setContentsMargins(margin, 0, margin, 0)
        self.panel_layout.setContentsMargins(int(24 * s), int(28 * s), int(24 * s), int(24 * s))
        for widget in (self.topbar, self.panel):
            widget.setStyleSheet(panel_style(s, self.theme))
        self.ns_panel.set_scale(s)

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
            if event.type() in (QEvent.MouseButtonPress, QEvent.NativeGesture) or (
                event.type() == QEvent.Wheel and event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier)
            ):
                self.controller.touch()  # dragging, pinching and zooming are visitors too
                self._orbit_pause_until = time.monotonic() + ORBIT_PAUSE_SECONDS  # hold the camera while they use it
            if event.type() == QEvent.MouseMove and event.buttons():
                self._orbit_pause_until = time.monotonic() + ORBIT_PAUSE_SECONDS
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
