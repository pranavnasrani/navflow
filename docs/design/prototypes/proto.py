"""Three throwaway prototypes of the NavFlow monitor window, rendered from real OpenFOAM preview data.

A  worked solution   (exam-paper grammar: printed question in black, live answers in biro blue)
B  smoke tunnel      (wind-tunnel light-sheet photograph: black, white smoke, glowing stagnation)
C  results window    (Fluent-style post-processing played straight: gradient, banded rainbow legends)

Run:  QT_SCALE_FACTOR=0.5 python proto.py OUT_DIR   (a 1920x1080 window = one 20" 1080p monitor)
"""
from __future__ import annotations

import faulthandler, os
if os.environ.get("PROTO_DEBUG"):
    faulthandler.dump_traceback_later(int(os.environ["PROTO_DEBUG"]), exit=True)

import math
import sys
from pathlib import Path

ROOT = Path("/Users/isaac/Elemental/Water/navflow")
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from app.hud_widgets import line_runs, particles_on_lines, polyline_from_nan_joined, wind_arrow_geometry
from app.state_manager import ObjectStateManager

CACHE = Path(__file__).resolve().parent.parent / "proto_cache"
W, H = 1920, 1080
TERMS = ("convective", "pressure", "viscous")
TERM_SYM = {"convective": "ρ(v·∇)v", "pressure": "−∇p", "viscous": "μ∇²v"}
TERM_WORD = {"convective": "inertia", "pressure": "pressure", "viscous": "friction"}
CP_RANGE = (-1.5, 1.0)
CAPTURE = True
ZERO = (0.0 - CP_RANGE[0]) / (CP_RANGE[1] - CP_RANGE[0])  # where Cp = 0 sits on the bar: 0.6


def num(m: dict, k: str) -> float:
    v = m.get(k, 0.0)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0


def sci(v: float, digits: int = 3) -> str:
    """1.44 × 10⁷ style."""
    if v == 0 or not math.isfinite(v):
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{v / 10**e:.{digits - 1}f} × 10{sup}"


def grouped(v: float, dp: int = 0) -> str:
    """Thin-space thousands, exam style: 1 247."""
    return f"{v:,.{dp}f}".replace(",", " ")


def font(family: str, px: int, weight: int = 400, tnum: bool = True) -> QFont:
    f = QFont(family)
    f.setPixelSize(px)
    f.setWeight(QFont.Weight(weight))
    if tnum:
        f.setFeature(QFont.Tag("tnum"), 1)
    return f


def label(text: str, f: QFont, color: str, wrap: bool = False, align=Qt.AlignLeft | Qt.AlignVCenter) -> QLabel:
    lb = QLabel(text)
    lb.setFont(f)
    lb.setStyleSheet(f"color: {color}; background: transparent; border: none;")
    lb.setWordWrap(wrap)
    lb.setAlignment(align)
    return lb


# ----------------------------------------------------------------------------------------------
# Painted legends
# ----------------------------------------------------------------------------------------------
class Legend(QWidget):
    """Colour legend painted with QPainter. orient 'h' or 'v'; bands>0 draws discrete bands (Fluent)."""

    def __init__(self, cmap, lo, hi, title, ticks, *, orient="h", bands=0, fg="#111", bg=None,
                 family="Arial", title_px=15, tick_px=13, bar=10, parent=None):
        super().__init__(parent)
        self.cmap, self.lo, self.hi, self.title, self.ticks = cmap, lo, hi, title, ticks
        self.orient, self.bands, self.fg, self.bg = orient, bands, fg, bg
        self.family, self.title_px, self.tick_px, self.bar = family, title_px, tick_px, bar

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if self.bg:
            p.fillRect(self.rect(), QColor(self.bg))
        p.setPen(QColor(self.fg))
        p.setFont(font(self.family, self.title_px, 600))
        n = self.bands or 96
        if self.orient == "h":
            p.drawText(QRectF(0, 0, self.width(), self.title_px + 6), Qt.AlignLeft | Qt.AlignVCenter, self.title)
            y0, x0, x1 = self.title_px + 10, 0.0, float(self.width())
            for i in range(n):
                c = self.cmap((i + 0.5) / n)
                p.fillRect(QRectF(x0 + i * (x1 - x0) / n, y0, (x1 - x0) / n + 0.6, self.bar),
                           QColor.fromRgbF(*c[:3]))
            p.setFont(font(self.family, self.tick_px))
            for v, t in self.ticks:
                x = x0 + (v - self.lo) / (self.hi - self.lo) * (x1 - x0)
                al = Qt.AlignLeft if x < 30 else (Qt.AlignRight if x > x1 - 30 else Qt.AlignHCenter)
                rx = x if al == Qt.AlignLeft else (x - 120 if al == Qt.AlignRight else x - 60)
                p.drawText(QRectF(rx, y0 + self.bar + 4, 120, self.tick_px + 6), al | Qt.AlignTop, t)
        else:
            p.drawText(QRectF(0, 0, self.width(), self.title_px * 2 + 8), Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
                       self.title)
            y0, y1, x0 = self.title_px * 2 + 16.0, self.height() - 10.0, 2.0
            for i in range(n):
                c = self.cmap(1.0 - (i + 0.5) / n)
                p.fillRect(QRectF(x0, y0 + i * (y1 - y0) / n, self.bar, (y1 - y0) / n + 0.6), QColor.fromRgbF(*c[:3]))
            if self.bands:
                p.setPen(QPen(QColor(0, 0, 0, 90), 1))
                p.drawRect(QRectF(x0, y0, self.bar, y1 - y0))
                p.setPen(QColor(self.fg))
            p.setFont(font(self.family, self.tick_px))
            for v, t in self.ticks:
                y = y1 - (v - self.lo) / (self.hi - self.lo) * (y1 - y0)
                p.drawText(QRectF(x0 + self.bar + 8, y - 9, 140, 18), Qt.AlignLeft | Qt.AlignVCenter, t)
        p.end()


class TermBars(QWidget):
    """Three flat log-scale bars (C)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.vals = {t: 1.0 for t in TERMS}
        self.setFixedHeight(3 * 34)

    def set(self, vals):
        self.vals = vals
        self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        dom = max(TERMS, key=lambda t: self.vals[t])
        for i, t in enumerate(TERMS):
            y = i * 34
            p.setPen(QColor("#1D232B"))
            p.setFont(font("Arial", 15, 700 if t == dom else 400))
            p.drawText(QRectF(0, y, 110, 26), Qt.AlignLeft | Qt.AlignVCenter, TERM_SYM[t])
            frac = min(max((math.log10(max(self.vals[t], 1e-3)) + 3) / 8.0, 0.0), 1.0)
            bw = self.width() - 110 - 110
            p.fillRect(QRectF(110, y + 9, bw, 8), QColor("#D3D8DF"))
            p.fillRect(QRectF(110, y + 9, bw * frac, 8), QColor("#2E3B4E" if t == dom else "#8A96A6"))
            p.drawText(QRectF(self.width() - 104, y, 104, 26), Qt.AlignRight | Qt.AlignVCenter,
                       f"{float(f'{self.vals[t]:.3g}'):,.4g} m/s²")
        p.end()


_OFFSCREEN: dict = {}


class OffscreenView(QWidget):
    """Stands in for QtInteractor when capturing: a plain Qt widget for layout and overlays, with the
    pyvista calls forwarded to an off-screen plotter rendered at this widget's size."""

    def __init__(self, parent=None):
        super().__init__(parent)
        _OFFSCREEN[id(self)] = pv.Plotter(off_screen=True, window_size=(W, H))

    def __getattr__(self, name):
        return getattr(_OFFSCREEN[id(self)], name)


# ----------------------------------------------------------------------------------------------
# Base prototype window
# ----------------------------------------------------------------------------------------------
class Proto(QMainWindow):
    design = "?"

    def __init__(self, obj_id: str, wand: int = 4, knob: float = 0.0, view: int = 0):
        super().__init__()
        self.obj_id, self.wand, self.knob, self.view = obj_id, wand, knob, view
        self.mgr = ObjectStateManager(obj_id, cache_dir=CACHE)
        self.cfg, self.cfd = self.mgr.obj_config, self.mgr.obj_config["cfd"]
        self.V = float(self.cfd["speed_m_s"])
        self.rho = float(self.cfd.get("density_kg_m3", 1.225))
        self.grounded = self.cfd.get("floor", "none") != "none"
        self.setWindowTitle(f"{self.design} · {self.cfg['display_name']}")
        self.resize(W, H)
        self.anim_phase = 0.0
        self.state = self.mgr.get_state(wand, knob)
        d = self.mgr.direction_for_knob(knob)
        i1, i2, t = self.mgr._bracket(d)
        self.near = i2 if t > 0.5 else i1
        self.build()
        self.scene()
        self.fill()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(33)

    # hooks -------------------------------------------------------------------------------
    def build(self): ...
    def fill(self): ...

    # helpers -----------------------------------------------------------------------------
    def make_plotter(self, parent) -> QtInteractor:
        pl = OffscreenView(parent) if CAPTURE else QtInteractor(parent)
        pl.enable_anti_aliasing("ssaa" if CAPTURE else "msaa")
        return pl

    def cp(self) -> np.ndarray:
        """Cp = p / (1/2 rho V^2): surface pressure over the dynamic pressure, the same scale on every screen."""
        return np.asarray(self.state["surface_pressure"], float) / (0.5 * self.rho * self.V**2)

    def wand_poly(self, w: int, bounds) -> pv.PolyData:
        lines, k, dom, speed = self.mgr._wands[self.near][w]
        sc = {"k": k, "dom": dom.astype(float)}
        if speed is not None:
            sc["speed"] = speed
        return polyline_from_nan_joined(lines, sc, bounds=bounds)

    def scene(self):
        pl = self.plotter
        b = np.asarray(self.mgr.view_bounds, float)
        self.bounds = b
        v, f = self.mgr.mesh_vertices, self.mgr.mesh_faces
        self.mesh = pv.PolyData(np.asarray(v, np.float32), np.column_stack([np.full(len(f), 3), f]).ravel())
        self.mesh.point_data["Cp"] = self.cp().astype(np.float32)
        self.active = self.wand_poly(self.wand, b)
        self.context = pv.merge([self.wand_poly(w, b) for w in range(len(self.mgr.wand_heights)) if w != self.wand and (w - self.wand) % 2 == 0])
        self.style_scene(pl, b)
        self.run_pts = np.asarray(self.active.points, float)
        self.starts, self.lengths = line_runs(self.active) if self.active.n_lines else (np.empty(0, int),) * 2
        self.camera(pl, b)

    def ground(self, pl, b, color, opacity=1.0):
        if not self.grounded:
            return
        cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        size = 1.6 * max(b[1] - b[0], b[3] - b[2])
        pl.add_mesh(pv.Plane(center=(cx, cy, b[4]), i_size=size, j_size=size), color=color, opacity=opacity,
                    ambient=1.0, diffuse=0.0, specular=0.0)

    def arrow(self, pl, color):
        v = np.asarray(self.mgr.mesh_vertices, float)
        c = (v.min(0) + v.max(0)) / 2
        start, d, length = wind_arrow_geometry(self.state["wind_vector"], c, self.bounds, 0.12)
        pl.add_mesh(pv.Arrow(start=start, direction=d, scale=length, tip_length=0.3, tip_radius=0.07,
                             shaft_radius=0.018), color=color, ambient=0.6, diffuse=0.4, specular=0.0)

    def focus_bounds(self):
        """Model box plus some room for the wake behind it, kept inside the display box."""
        v = np.asarray(self.mgr.mesh_vertices, float)
        lo, hi = v.min(0), v.max(0)
        L = float((hi - lo).max())
        f = np.array([lo[0] - 0.2 * L, hi[0] + 0.7 * L, lo[1] - 0.25 * L, hi[1] + 0.25 * L, lo[2], hi[2] + 0.15 * L])
        b = self.bounds
        f[0::2] = np.maximum(f[0::2], b[0::2])
        f[1::2] = np.minimum(f[1::2], b[1::2])
        return f

    def camera(self, pl, b, eye_dir=(-0.62, -0.95, 0.42), zoom=1.45):
        f = self.focus_bounds()
        centre = (f[0::2] + f[1::2]) / 2
        diag = float(np.linalg.norm(f[1::2] - f[0::2]))
        eye = centre + diag * np.array(eye_dir)
        pl.camera_position = [tuple(eye), tuple(centre), (0, 0, 1)]
        pl.reset_camera(bounds=tuple(f))
        pl.camera.zoom(zoom)
        self._cam = (eye_dir, zoom)

    def animate(self):
        if len(self.starts) == 0 or not hasattr(self, "particles"):
            return
        self.anim_phase = (self.anim_phase + 0.005) % 1.0
        pos = particles_on_lines(self.run_pts, self.starts, self.lengths, self.anim_phase, per_line=2)
        self.particles.copy_from(pv.PolyData(pos.astype(np.float32)))
        (_OFFSCREEN[id(self.plotter)] if CAPTURE else self.plotter).render()

    def speed_clim(self):
        s = self.active.point_data.get("speed")
        if s is None or not len(s):
            return 0.0, self.V
        s = np.asarray(s)
        return float(np.percentile(s, 1)), float(np.percentile(s, 99))

    def metric(self, k):
        return num(self.state["metrics"], k)

    def ns(self):
        return {t: num(self.state["ns_terms"], t) for t in TERMS}


# ----------------------------------------------------------------------------------------------
# A  Worked solution
# ----------------------------------------------------------------------------------------------
PAPER, INK, INK2, RULE, BIRO = "#FBFBFA", "#121212", "#5A5A5A", "#D6D6D2", "#1B3A9A"
A_TERM = {"convective": "#0072B2", "pressure": "#D55E00", "viscous": "#AA3377"}


class ProtoA(Proto):
    design = "A worked solution"

    def build(self):
        root = QWidget()
        root.setStyleSheet(f"background: {PAPER};")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(56, 36, 56, 32)
        outer.setSpacing(0)

        head = QHBoxLayout()
        head.addWidget(label(self.cfg["display_name"], font("Arial", 34, 700), INK))
        head.addStretch()
        self.head_right = label("", font("Arial", 19), INK2, align=Qt.AlignRight | Qt.AlignVCenter)
        head.addWidget(self.head_right)
        outer.addLayout(head)
        outer.addSpacing(14)
        rule = QFrame()
        rule.setFixedHeight(2)
        rule.setStyleSheet(f"background: {INK};")
        outer.addWidget(rule)
        outer.addSpacing(26)

        body = QHBoxLayout()
        body.setSpacing(56)
        fig = QVBoxLayout()
        fig.setSpacing(12)
        frame = QFrame()
        frame.setStyleSheet(f"QFrame {{ background: #FFFFFF; border: 1px solid {RULE}; }}")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(1, 1, 1, 1)
        self.plotter = self.make_plotter(frame)
        fl.addWidget(self.plotter)
        fig.addWidget(frame, 1)
        self.caption = label("", font("Arial", 18), INK, wrap=True)
        fig.addWidget(self.caption)
        body.addLayout(fig, 1)

        col = QVBoxLayout()
        col.setSpacing(0)
        colw = QWidget()
        colw.setFixedWidth(600)
        colw.setLayout(col)
        self.col = col
        body.addWidget(colw)
        outer.addLayout(body, 1)

        self.cp_legend = Legend(self.cp_cmap(), *CP_RANGE, "Pressure coefficient Cp on the surface",
                                [(-1.5, "−1.5  suction"), (0.0, "0"), (1.0, "+1  stagnation")], bg="#FFFFFF",
                                fg=INK2, title_px=15, tick_px=14, bar=9, parent=self.plotter)
        self.sp_legend = Legend(self.speed_cmap(), 0, 1, "Air speed along the streamlines (m s⁻¹)", [],
                                bg="#FFFFFF", fg=INK2, title_px=15, tick_px=14, bar=9, parent=self.plotter)

    def cp_cmap(self):
        return LinearSegmentedColormap.from_list("cpA", [(0.0, "#2B5FA8"), (0.3, "#9DB6D8"), (ZERO, "#E6E6E3"), (0.8, "#E3A07F"), (1.0, "#B23A1E")])

    def speed_cmap(self):
        return LinearSegmentedColormap.from_list("biro", ["#B7C3E6", "#5B74C4", BIRO, "#0B1E5C"])

    def style_scene(self, pl, b):
        pl.set_background("#FFFFFF")
        pl.add_mesh(self.mesh, scalars="Cp", cmap=self.cp_cmap(), clim=CP_RANGE, smooth_shading=True,
                    show_scalar_bar=False, ambient=0.25, diffuse=0.75, specular=0.08, specular_power=12)
        lo, hi = self.speed_clim()
        self.sp_range = (lo, hi)
        pl.add_mesh(self.context, color="#C9CFDC", line_width=1.0, opacity=0.55)
        if "speed" in self.active.point_data:
            pl.add_mesh(self.active, scalars="speed", cmap=self.speed_cmap(), clim=(lo, hi), line_width=2.6,
                        show_scalar_bar=False)
        else:
            pl.add_mesh(self.active, color=BIRO, line_width=2.6)
        self.particles = pv.PolyData(np.zeros((1, 3), np.float32))
        pl.add_mesh(self.particles, color=BIRO, point_size=6, render_points_as_spheres=True)
        self.arrow(pl, INK2)

    def fill(self):
        s, m = self.state, self.state["metrics"]
        n_wand = len(self.mgr.wand_heights)
        dl = self.cfd.get("direction_label", "Wind direction").lower()
        self.head_right.setText(f"V = {self.V:g} m s⁻¹   ·   ρ = {self.rho:g} kg m⁻³   ·   ν = 1.5 × 10⁻⁵ m² s⁻¹")
        self.caption.setText(
            f"<b>Fig. 1</b>&nbsp;&nbsp;Streamlines from smoke-wand height "
            f"<span style='color:{BIRO}'>{s['wand_index'] + 1}</span> of {n_wand} "
            f"(z = <span style='color:{BIRO}'>{s['wand_z']:.2f}</span> m), {dl} "
            f"<span style='color:{BIRO}'>{s['direction_deg']:.1f}°</span>.")
        lo, hi = self.sp_range
        self.sp_legend.lo, self.sp_legend.hi = lo, hi
        self.sp_legend.ticks = [(lo, f"{lo:.0f}"), ((lo + hi) / 2, f"{(lo + hi) / 2:.0f}"), (hi, f"{hi:.0f}")]

        col = self.col
        q = lambda n, t: self._question(n, t)
        # 1. Which term wins
        q("1", "Which term of the momentum equation is largest near the model?")
        vals = self.ns()
        dom = max(TERMS, key=lambda t: vals[t])
        logs = {t: math.log10(max(vals[t], 1e-3)) for t in TERMS}
        lmax = max(logs.values())
        size = {t: int(round(max(20, 46 - 13 * (lmax - logs[t])))) for t in TERMS}
        def term(t):
            c = A_TERM[t]
            w = 700 if t == dom else 400
            return f"<span style='font-size:{size[t]}px; color:{c}; font-weight:{w}'>{TERM_SYM[t]}</span>"
        eq = (f"<span style='font-family:Times New Roman; font-style:italic'>{term('convective')}"
              f"<span style='font-size:30px; color:{INK}'> = </span>{term('pressure')}"
              f"<span style='font-size:30px; color:{INK}'> + </span>{term('viscous')}</span>")
        eql = label(eq, font("Times New Roman", 30, 400, tnum=False), INK)
        eql.setTextFormat(Qt.RichText)
        eql.setMinimumHeight(70)
        col.addWidget(eql)
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(6)
        for i, t in enumerate(TERMS):
            grid.addWidget(label(TERM_WORD[t], font("Arial", 17), A_TERM[t]), i, 0)
            grid.addWidget(label(f"{grouped(float(f'{vals[t]:.3g}'))} m s⁻²", font("Arial", 19, 700 if t == dom else 400), BIRO,
                                 align=Qt.AlignRight | Qt.AlignVCenter), i, 1)
        col.addLayout(grid)
        col.addSpacing(10)
        self._answer(f"Largest: {TERM_WORD[dom]}")

        # 2. Forces
        col.addSpacing(30)
        q("2", "Find the drag coefficient and the power needed to push through the air.")
        D, L, A = self.metric("drag_force_n"), self.metric("lift_force_n"), self.metric("frontal_area_m2")
        Re = self.metric("reynolds_number")
        work = [
            f"Re = VD / ν = <span style='color:{BIRO}'>{sci(Re)}</span>",
            f"D = <span style='color:{BIRO}'>{grouped(D)} N</span>,&nbsp;&nbsp; L = "
            f"<span style='color:{BIRO}'>{grouped(L)} N</span>,&nbsp;&nbsp; A = "
            f"<span style='color:{BIRO}'>{A:.3g} m²</span>",
            f"C<sub>d</sub> = D / (½ρV²A) = <span style='color:{BIRO}'>{self.metric('drag_coefficient_cd'):.3f}</span>",
        ]
        for wline in work:
            lb = label(wline, font("Arial", 19), INK)
            lb.setTextFormat(Qt.RichText)
            col.addWidget(lb)
            col.addSpacing(6)
        self._answer(f"P = ½ρC<sub>d</sub>AV³ = {self._power()}")

        # 3. Sustainability
        col.addSpacing(30)
        widgets = tuple(self.cfd.get("widgets", ()))
        if "fuel" in widgets:
            ref = self.metric("reference_direction_deg")
            q("3", f"How much fuel does this angle save compared with the worst angle ({ref:g}°)?")
            lb = label(f"Burn <span style='color:{BIRO}'>{grouped(self.metric('fuel_burn_l_per_hr'), 1)} L h⁻¹</span>"
                       f"&nbsp;&nbsp;·&nbsp;&nbsp;CO₂ saved <span style='color:{BIRO}'>"
                       f"{grouped(self.metric('co2_saved_kg_per_hr'), 1)} kg h⁻¹</span>", font("Arial", 19), INK)
            lb.setTextFormat(Qt.RichText)
            col.addWidget(lb)
            self._answer(f"Saved {grouped(self.metric('fuel_saved_l_per_hr'), 1)} L h⁻¹")
        elif "wind_load" in widgets:
            q("3", "How hard does a hurricane push on the tower?")
            for row in (m.get("wind_load") or [])[1:]:
                lb = label(f"{row['label']}&nbsp;&nbsp;({row['v_m_s']:.0f} m s⁻¹):&nbsp;&nbsp;"
                           f"<span style='color:{BIRO}'>{num(row, 'force_n') / 1e6:.1f} MN</span>",
                           font("Arial", 19), INK)
                lb.setTextFormat(Qt.RichText)
                col.addWidget(lb)
                col.addSpacing(4)
        else:
            q("3", "Why does a person need so much power to run into the wind?")
            lb = label("A body is a bluff shape: the air separates and leaves a wide, slow wake behind it.",
                       font("Arial", 19), INK, wrap=True)
            col.addWidget(lb)
        col.addStretch()
        col.addWidget(label("Printed in black: the question. In blue: OpenFOAM's answer for the knob and wand now.",
                            font("Arial", 15), INK2, wrap=True))

    def _power(self):
        p = self.metric("drag_power_kw")
        return f"{grouped(p / 1000, 2)} MW" if p >= 1000 else f"{p:.1f} kW"

    def _question(self, n, text):
        row = QHBoxLayout()
        row.setSpacing(14)
        nl = label(n, font("Arial", 20, 700), INK, align=Qt.AlignLeft | Qt.AlignTop)
        nl.setFixedWidth(22)
        row.addWidget(nl, 0, Qt.AlignTop)
        row.addWidget(label(text, font("Arial", 20, 700), INK, wrap=True), 1)
        self.col.addLayout(row)
        self.col.addSpacing(12)

    def _answer(self, text):
        row = QHBoxLayout()
        dots = label("Answer", font("Arial", 17), INK2)
        row.addWidget(dots)
        row.addStretch()
        a = label(text, font("Arial", 24, 700), BIRO, align=Qt.AlignRight | Qt.AlignVCenter)
        a.setTextFormat(Qt.RichText)
        row.addWidget(a)
        self.col.addLayout(row)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"border-top: 1px dotted {INK2}; background: transparent;")
        self.col.addWidget(line)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        QTimer.singleShot(0, self.place)

    def place(self):
        pw, ph = self.plotter.width(), self.plotter.height()
        for i, lg in enumerate((self.cp_legend, self.sp_legend)):
            lg.setFixedSize(340, 58)
            lg.move(24 + i * 380, ph - 58 - 20)
            lg.show()


# ----------------------------------------------------------------------------------------------
# B  Smoke tunnel
# ----------------------------------------------------------------------------------------------
B_FG, B_MUTED, B_DIM = "#F2F2F2", "#8C8C8C", "#3A3A3A"


class ProtoB(Proto):
    design = "B smoke tunnel"

    def build(self):
        root = QWidget()
        root.setStyleSheet("background: #000000;")
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.plotter = self.make_plotter(root)
        lay.addWidget(self.plotter, 1)

        sheet = QWidget()
        sheet.setFixedHeight(200)
        sheet.setStyleSheet("background: #000000;")
        sl = QHBoxLayout(sheet)
        sl.setContentsMargins(64, 0, 64, 34)
        sl.setSpacing(0)
        self.sheet = sl
        rule = QFrame(sheet)
        rule.setGeometry(64, 0, W - 128, 1)
        rule.setStyleSheet(f"background: {B_DIM};")
        self.sheet_rule = rule
        lay.addWidget(sheet)

        self.title = label(self.cfg["display_name"], font("DIN Alternate", 44, 700), B_FG)
        self.title.setParent(self.plotter)
        self.subtitle = label("", font("DIN Alternate", 21, 700), B_MUTED)
        self.subtitle.setParent(self.plotter)
        self.cp_legend = Legend(self.cp_cmap(), *CP_RANGE, "Cp", [(1.0, "+1.0  air stopped"), (0.0, "0"),
                                                                   (-1.5, "−1.5  suction")], orient="v",
                                fg=B_MUTED, bg=None, family="DIN Alternate", title_px=17, tick_px=16, bar=6,
                                parent=self.plotter)

    def cp_cmap(self):
        # zero is dark: undisturbed air stays black, stagnation glows amber, suction glows blue
        return LinearSegmentedColormap.from_list("ice", [(0.0, "#8CC0FF"), (0.3, "#3E6DB5"), (ZERO, "#4A4E55"), (0.8, "#C9763A"), (1.0, "#FFC890")])

    def style_scene(self, pl, b):
        pl.set_background("#000000")
        self.ground(pl, b, "#0A0A0B")
        lo, hi = CP_RANGE
        # map Cp so that 0 sits at the dark centre of the colormap: rescale both sides
        pl.add_mesh(self.mesh, scalars="Cp", cmap=self.cp_cmap(), clim=CP_RANGE, smooth_shading=True,
                    show_scalar_bar=False, ambient=0.3, diffuse=0.8, specular=0.12, specular_power=18)
        pl.add_mesh(self.context, color="#FFFFFF", line_width=1.0, opacity=0.11)
        pl.add_mesh(self.active, color="#FFFFFF", line_width=2.2, opacity=0.92)
        self.particles = pv.PolyData(np.zeros((1, 3), np.float32))
        pl.add_mesh(self.particles, color="#FFFFFF", point_size=7, render_points_as_spheres=True, opacity=1.0,
                    ambient=1.0, diffuse=0.0)
        self.arrow(pl, "#5A5A5A")

    def fill(self):
        s, m = self.state, self.state["metrics"]
        dl = self.cfd.get("direction_label", "Wind direction").lower()
        n_wand = len(self.mgr.wand_heights)
        self.subtitle.setText(f"{self.V:g} m/s   ·   {dl} {s['direction_deg']:.1f}°   ·   smoke wand "
                              f"{s['wand_index'] + 1}/{n_wand}")
        cells = [("Drag", f"{grouped(self.metric('drag_force_n'))}", "N"),
                 ("Lift", f"{grouped(self.metric('lift_force_n'))}", "N"),
                 ("Cd", f"{self.metric('drag_coefficient_cd'):.3f}", ""),
                 ("Drag power", *self._power())]
        widgets = tuple(self.cfd.get("widgets", ()))
        if "fuel" in widgets:
            cells.append(("Fuel saved vs worst angle", f"{grouped(self.metric('fuel_saved_l_per_hr'), 1)}", "L/h"))
        elif "wind_load" in widgets:
            rows = m.get("wind_load") or []
            if rows:
                cells.append((f"Force in a {rows[-1]['label'].lower()}", f"{num(rows[-1], 'force_n') / 1e6:.1f}", "MN"))
        for title, val, unit in cells:
            c = QVBoxLayout()
            c.setSpacing(2)
            c.addStretch()
            c.addWidget(label(title, font("DIN Alternate", 17, 700), B_MUTED))
            row = QHBoxLayout()
            row.setSpacing(8)
            row.addWidget(label(val, font("DIN Alternate", 52, 700), B_FG), 0, Qt.AlignBottom)
            if unit:
                row.addWidget(label(unit, font("DIN Alternate", 22, 700), B_MUTED), 0, Qt.AlignBottom)
            row.addStretch()
            c.addLayout(row)
            self.sheet.addLayout(c, 1)
        # breathing equation
        vals = self.ns()
        dom = max(TERMS, key=lambda t: vals[t])
        logs = {t: math.log10(max(vals[t], 1e-3)) for t in TERMS}
        lmax = max(logs.values())
        def term(t):
            sz = int(round(max(20, 44 - 12 * (lmax - logs[t]))))
            col = B_FG if t == dom else "#6E6E6E"
            return f"<span style='font-size:{sz}px; color:{col}'>{TERM_SYM[t]}</span>"
        eq = label(f"{term('convective')}<span style='font-size:26px;color:#6E6E6E'> = </span>{term('pressure')}"
                   f"<span style='font-size:26px;color:#6E6E6E'> + </span>{term('viscous')}",
                   font("Times New Roman", 26, 400, tnum=False), B_FG, align=Qt.AlignRight | Qt.AlignBottom)
        eq.setTextFormat(Qt.RichText)
        c = QVBoxLayout()
        c.addStretch()
        c.addWidget(label(f"Largest term near the model: {TERM_WORD[dom]}", font("DIN Alternate", 17, 700), B_MUTED,
                          align=Qt.AlignRight))
        c.addWidget(eq)
        self.sheet.addLayout(c, 1)
        self.log = label(f"OpenFOAM k-ω SST · {int(self.metric('iterations'))} iterations · "
                         f"{'converged' if m.get('converged') else 'time-averaged'}",
                         font("DIN Alternate", 15, 700), "#5C5C5C")
        self.log.setParent(self.plotter)

    def _power(self):
        p = self.metric("drag_power_kw")
        return (f"{p / 1000:,.2f}".replace(",", " "), "MW") if p >= 1000 else (f"{p:.1f}", "kW")

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        QTimer.singleShot(0, self.place)

    def place(self):
        pw, ph = self.plotter.width(), self.plotter.height()
        self.title.adjustSize(); self.title.move(64, 48); self.title.show()
        self.subtitle.adjustSize(); self.subtitle.move(66, 48 + self.title.height() + 4); self.subtitle.show()
        self.cp_legend.setFixedSize(220, 330)
        self.cp_legend.move(pw - 220 - 40, (ph - 330) // 2)
        self.cp_legend.show()
        self.log.adjustSize(); self.log.move(pw - self.log.width() - 64, 56); self.log.show()


# ----------------------------------------------------------------------------------------------
# C  Results window (Fluent-style canon)
# ----------------------------------------------------------------------------------------------
C_PANEL, C_INK, C_MUTED, C_RULE = "#ECEEF1", "#1D232B", "#5B6573", "#C7CDD5"


def jet_bands(n):
    return matplotlib.colormaps["jet"].resampled(n)


class ProtoC(Proto):
    design = "C results window"

    def build(self):
        root = QWidget()
        root.setStyleSheet(f"background: {C_PANEL};")
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.plotter = self.make_plotter(root)
        lay.addWidget(self.plotter, 1)
        panel = QWidget()
        panel.setFixedWidth(500)
        panel.setStyleSheet(f"background: {C_PANEL}; border-left: 1px solid {C_RULE};")
        self.pl = QVBoxLayout(panel)
        self.pl.setContentsMargins(28, 26, 28, 0)
        self.pl.setSpacing(0)
        lay.addWidget(panel)

        self.anno = label("", font("Arial", 16), C_INK)
        self.anno.setParent(self.plotter)
        self.v_legend = Legend(jet_bands(12), 0, 1, "Velocity Magnitude\n[m s⁻¹]", [], orient="v", bands=12,
                               fg=C_INK, title_px=15, tick_px=14, bar=22, parent=self.plotter)
        self.c_legend = Legend(jet_bands(12), *CP_RANGE, "Pressure Coefficient\nCp", [], orient="v", bands=12,
                               fg=C_INK, title_px=15, tick_px=14, bar=22, parent=self.plotter)

    def style_scene(self, pl, b):
        pl.set_background("#A7B6CA", top="#FFFFFF")
        lo, hi = self.speed_clim()
        self.sp_range = (lo, hi)
        pl.add_mesh(self.mesh, scalars="Cp", cmap="jet", n_colors=12, clim=CP_RANGE, smooth_shading=True,
                    show_scalar_bar=False, ambient=0.3, diffuse=0.7, specular=0.2, specular_power=30)
        if "speed" in self.active.point_data:
            pl.add_mesh(self.active, scalars="speed", cmap="jet", n_colors=12, clim=(lo, hi), line_width=2.4,
                        show_scalar_bar=False)
        self.particles = pv.PolyData(np.zeros((1, 3), np.float32))
        pl.add_mesh(self.particles, color="#111111", point_size=5, render_points_as_spheres=True)
        pl.show_axes()

    def fill(self):
        s, m = self.state, self.state["metrics"]
        dl = self.cfd.get("direction_label", "Wind direction")
        lo, hi = self.sp_range
        self.v_legend.lo, self.v_legend.hi = lo, hi
        self.v_legend.ticks = [(lo + i * (hi - lo) / 6, f"{lo + i * (hi - lo) / 6:.3e}") for i in range(7)]
        self.c_legend.ticks = [(CP_RANGE[0] + i * 2.5 / 5, f"{CP_RANGE[0] + i * 2.5 / 5:.2e}") for i in range(6)]
        self.anno.setText(f"Pathlines Colored by Velocity Magnitude (m/s)  ·  Contours of Pressure Coefficient"
                          f"       {self.cfg['display_name']}  ·  {dl} {s['direction_deg']:.1f}°  ·  "
                          f"wand {s['wand_index'] + 1}/{len(self.mgr.wand_heights)}")

        def section(title):
            self.pl.addSpacing(18)
            self.pl.addWidget(label(title, font("Arial", 18, 700), C_INK))
            self.pl.addSpacing(6)
            r = QFrame()
            r.setFixedHeight(1)
            r.setStyleSheet(f"background: {C_RULE}; border: none;")
            self.pl.addWidget(r)
            self.pl.addSpacing(8)

        def kv(rows):
            g = QGridLayout()
            g.setVerticalSpacing(7)
            for i, (k, v) in enumerate(rows):
                g.addWidget(label(k, font("Arial", 17), C_MUTED), i, 0)
                g.addWidget(label(v, font("Arial", 17, 600), C_INK, align=Qt.AlignRight | Qt.AlignVCenter), i, 1)
            self.pl.addLayout(g)

        self.pl.addWidget(label(self.cfg["display_name"], font("Arial", 28, 700), C_INK))
        self.pl.addWidget(label(f"{self.V:g} m/s  ·  {dl.lower()} {s['direction_deg']:.1f}°", font("Arial", 17),
                                C_MUTED))
        section("Forces")
        kv([("Drag", f"{grouped(self.metric('drag_force_n'))} N"), ("Lift", f"{grouped(self.metric('lift_force_n'))} N"),
            ("Cd", f"{self.metric('drag_coefficient_cd'):.4f}"), ("Cl", f"{self.metric('lift_coefficient_cl'):+.4f}"),
            ("Frontal area", f"{self.metric('frontal_area_m2'):.3f} m²"),
            ("Drag power  ½ρCdAV³", f"{grouped(self.metric('drag_power_kw'), 1)} kW"),
            ("Reynolds number", f"{self.metric('reynolds_number'):.3e}")])
        section("Momentum terms near the model")
        tb = TermBars()
        tb.set(self.ns())
        self.pl.addWidget(tb)
        widgets = tuple(self.cfd.get("widgets", ()))
        if "fuel" in widgets:
            section("Sustainability")
            kv([("Fuel burn", f"{grouped(self.metric('fuel_burn_l_per_hr'), 1)} L/h"),
                (f"Saved vs worst ({self.metric('reference_direction_deg'):g}°)",
                 f"{grouped(self.metric('fuel_saved_l_per_hr'), 1)} L/h"),
                ("CO₂ saved", f"{grouped(self.metric('co2_saved_kg_per_hr'), 1)} kg/h")])
        elif "wind_load" in widgets:
            section("Wind load")
            kv([(r["label"], f"{num(r, 'force_n') / 1e6:.2f} MN") for r in (m.get("wind_load") or [])])
        self.pl.addStretch()
        console = label(f"> solve  k-omega SST  simpleFoam\n"
                        f"  iter {int(self.metric('iterations')):>5}   Cd {self.metric('drag_coefficient_cd'):.4e}   "
                        f"Cl {self.metric('lift_coefficient_cl'):+.4e}\n"
                        f"  {'converged' if m.get('converged') else 'cap reached: time-averaged fields'}",
                        font("Menlo", 14), "#D5DBE3", wrap=False)
        console.setStyleSheet("background: #1E232A; color: #D5DBE3; padding: 14px 16px; border: none;")
        self.pl.addWidget(console)

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        QTimer.singleShot(0, self.place)

    def place(self):
        pw, ph = self.plotter.width(), self.plotter.height()
        self.anno.adjustSize(); self.anno.move(24, 18); self.anno.show()
        h = (ph - 140) // 2
        self.v_legend.setFixedSize(190, h); self.v_legend.move(24, 70); self.v_legend.show()
        self.c_legend.setFixedSize(190, h); self.c_legend.move(24, 70 + h + 20); self.c_legend.show()


DESIGNS = {"A": ProtoA, "B": ProtoB, "C": ProtoC}


def capture(win, path):
    """Window -> PNG without a screen: Qt via grab(), the 3D view from the off-screen plotter at the
    placeholder's size, then the overlay widgets that float on the 3D view painted on top."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage
    view = win.plotter
    root = win.centralWidget()
    img = root.grab().toImage().convertToFormat(QImage.Format_RGB32)
    dpr = img.devicePixelRatio()
    pv_pl = _OFFSCREEN[id(view)]
    pv_pl.window_size = (int(view.width() * dpr), int(view.height() * dpr))
    win.camera(view, win.bounds, *win._cam)
    arr = np.ascontiguousarray(pv_pl.screenshot(return_img=True)[:, :, :3])
    h, w, _ = arr.shape
    shot = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    origin = view.mapTo(root, QPoint(0, 0))
    p = QPainter(img)
    p.setRenderHint(QPainter.SmoothPixmapTransform)
    p.drawImage(QRectF(origin.x(), origin.y(), view.width(), view.height()), shot)
    for child in view.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
        if child.isVisible() and child.width() > 0:
            child.render(p, origin + child.pos(), renderFlags=QWidget.RenderFlag.DrawChildren)
    p.end()
    img.save(str(path))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, o, 4, 0.0) for d in "ABC" for o in ("airplane", "f1_car", "building", "human")]
    if len(sys.argv) > 2:
        jobs = [j for j in jobs if j[0] in sys.argv[2]]
    app = QApplication.instance() or QApplication([])
    for d, o, wand, knob in jobs:
        print("build", d, o, flush=True)
        win = DESIGNS[d](o, wand, knob)
        win.timer.stop()
        win.setAttribute(Qt.WA_DontShowOnScreen, True)
        win.resize(W, H)
        win.show()
        for _ in range(30):
            app.processEvents()
        win.place()
        win.animate()
        for _ in range(10):
            app.processEvents()
        capture(win, out / f"{d}_{o}.png")
        print("saved", d, o, flush=True)
        win.close()
        app.processEvents()


    from PIL import Image
    shots = sorted(out.glob("[ABC]_*.png"))
    objs = ("airplane", "f1_car", "building", "human")
    tw, th = 960, 540
    sheet = Image.new("RGB", (4 * tw + 5 * 20, 3 * th + 4 * 20), (40, 40, 40))
    for r, d in enumerate("ABC"):
        for c, o in enumerate(objs):
            f = out / f"{d}_{o}.png"
            if f.exists():
                sheet.paste(Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS), (20 + c * (tw + 20), 20 + r * (th + 20)))
    sheet.save(out / "contact_sheet.png")

if __name__ == "__main__":
    main()
