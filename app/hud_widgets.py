"""
HUD building blocks for the monitor windows (native Qt widgets, no web content).

  * `polyline_from_nan_joined`  NaN-joined streamline points  ->  a PyVista PolyData of line cells.
  * `particles_on_lines`        flow tracer positions that slide along those lines.
  * `log_bar_value`             0..100 bar length for a number that spans many powers of ten.
  * `NsTermsPanel`              the equation  rho (v.grad)v = -grad p + mu laplacian(v)  with one bar per term.
  * `SustainabilityOverlay`     fuel / CO2, top-speed and wind-load cards with their UN SDG tag.

Maths link (IGCSE 0606): logarithms (the bars), linear interpolation (the tracers), the cube root and
the squared-speed rule (the sustainability numbers, derived in solver/foam/sustainability.py).
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

# --------------------------------------------------------------------------------------------------
# Colours. Three fixed colours name the three Navier-Stokes terms everywhere (bars, streamlines in
# view 2, the legend). Chosen to stay apart for colour-blind viewers and to read on a dark screen.
# --------------------------------------------------------------------------------------------------
TERM_NAMES = ("convective", "pressure", "viscous")
TERM_COLORS = {"convective": "#56B4E9", "pressure": "#F0E442", "viscous": "#FF6FB5"}
TERM_LABELS = {
    "convective": "(v·∇)v — convective (inertia)",
    "pressure": "−∇p — pressure gradient",
    "viscous": "μ∇²v — viscous friction (incl. turbulent mixing, k-ω SST)",
}
# The same names with an explicit line break, so a bar's title never depends on word-wrap.
TERM_LABELS_2LINE = {
    "convective": TERM_LABELS["convective"],
    "pressure": TERM_LABELS["pressure"],
    "viscous": "μ∇²v — viscous friction\n(incl. turbulent mixing, k-ω SST)",
}
EQUATION = "ρ(v·∇)v = −∇p + μ∇²v"

PANEL_BG = "rgba(8, 13, 24, 225)"
TEXT = "#F1F5F9"
MUTED = "#AAB6C8"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """'#RRGGBB' -> (r, g, b) bytes."""
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# --------------------------------------------------------------------------------------------------
# Streamlines and tracers
# --------------------------------------------------------------------------------------------------
def polyline_from_nan_joined(
    points: np.ndarray,
    scalars: dict[str, np.ndarray] | None = None,
    bounds: np.ndarray | None = None,
) -> pv.PolyData:
    """One line cell per run of consecutive valid points in a NaN-joined (M, 3) array.

    A row of NaN is a separator between streamlines. If `bounds` (xmin, xmax, ymin, ymax, zmin, zmax)
    is given, a point outside that box also acts as a separator (the importer's lines can stick out a
    little past the display box). Runs of a single point cannot be drawn as a line and are dropped.
    Each array in `scalars` (one value per input row) keeps its values for the surviving points only,
    so `len(scalars[name]) == poly.n_points`.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    if bounds is not None:
        b = np.asarray(bounds, dtype=float)
        # lo <= x <= hi on each axis: the inequality is evaluated for x, y and z at once.
        valid &= np.all((pts >= b[0::2]) & (pts <= b[1::2]), axis=1)

    idx = np.flatnonzero(valid)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1) if idx.size else []
    runs = [r for r in runs if len(r) >= 2]

    keep = np.concatenate(runs) if runs else np.empty(0, dtype=int)
    # A line cell is stored as [n, id0, id1, ..., id(n-1)]; ids count along the kept points.
    cells, offset = [], 0
    for r in runs:
        cells.append(np.concatenate([[len(r)], np.arange(offset, offset + len(r))]))
        offset += len(r)

    out = pv.PolyData(pts[keep].astype(np.float32))
    if cells:
        out.lines = np.concatenate(cells).astype(np.int64)
    if scalars:
        for name, values in scalars.items():
            out.point_data[name] = np.asarray(values)[keep]
    return out


def line_runs(poly: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    """(first point id, point count) of every line cell of `poly`."""
    conn = np.asarray(poly.lines)
    starts, lengths, i = [], [], 0
    while i < len(conn):
        n = int(conn[i])
        starts.append(int(conn[i + 1]))
        lengths.append(n)
        i += n + 1
    return np.asarray(starts, dtype=int), np.asarray(lengths, dtype=int)


def particles_on_lines(
    points: np.ndarray, starts: np.ndarray, lengths: np.ndarray, phase: float, per_line: int = 3
) -> np.ndarray:
    """Positions of `per_line` tracer dots on every line, `phase` in [0, 1) says how far along they are.

    Linear interpolation (IGCSE 0606): a dot at fractional position s = f (n - 1) lies between point
    i = floor(s) and point i + 1, at P = (1 - t) P_i + t P_(i+1) with t = s - i. Each line gets its own
    stagger so the dots do not march as one wall.
    """
    if len(starts) == 0:
        return np.empty((0, 3))
    stagger = (np.arange(len(starts)) * 0.381966) % 1.0  # golden-ratio steps spread the offsets evenly
    frac = (phase + stagger[:, None] + np.arange(per_line)[None, :] / per_line) % 1.0  # (lines, per_line)
    pos = frac * (lengths[:, None] - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, lengths[:, None] - 1)
    t = (pos - i0)[..., None]
    p0 = points[starts[:, None] + i0]
    p1 = points[starts[:, None] + i1]
    return ((1.0 - t) * p0 + t * p1).reshape(-1, 3)


def wind_arrow_geometry(
    wind_vector: np.ndarray, centre: np.ndarray, bounds: np.ndarray, length_frac: float = 0.2
) -> tuple[np.ndarray, np.ndarray, float]:
    """(start, unit direction, length) of the wind arrow: it sits on the upstream edge and points downstream.

    Walk from `centre` BACK against the wind, P = C - s d (d = unit wind vector), until the ray leaves the
    display box. Each axis gives its own s_i, the distance to the face the ray reaches first divided by
    the component of d on that axis (a rate-of-change / similar-triangles calculation):
        d_i > 0: the upstream face is the low face, s_i = (C_i - lo_i) / d_i
        d_i < 0: the upstream face is the high face, s_i = (hi_i - C_i) / (-d_i)
    s is the smallest s_i. The arrow starts at 0.97 s (just inside the face), is `length_frac` of the longest
    box side long, and is shortened if there is less room than that upstream of the centre.
    """
    d = np.asarray(wind_vector, dtype=float)
    n = float(np.linalg.norm(d))
    d = d / n if n > 0.0 else np.array([1.0, 0.0, 0.0])
    b = np.asarray(bounds, dtype=float)
    lo, hi = b[0::2], b[1::2]
    c = np.clip(np.asarray(centre, dtype=float), lo, hi)
    s = np.inf
    for i in range(3):
        if d[i] > 1e-9:
            s = min(s, (c[i] - lo[i]) / d[i])
        elif d[i] < -1e-9:
            s = min(s, (hi[i] - c[i]) / -d[i])
    if not np.isfinite(s):
        s = 0.0
    length = min(length_frac * float((hi - lo).max()), max(0.97 * s, 1e-6))
    return c - 0.97 * s * d, d, float(length)


def wand_rod_endpoints(lines: np.ndarray, z: float, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """End points of the smoke-wand rod: the two extreme seeds (first point of every streamline).

    With no streamlines the rod falls back to a line across y at height z near the upstream end of the box.
    """
    pts = np.asarray(lines, dtype=float).reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    prev_nan = np.concatenate([[True], ~valid[:-1]]) if len(valid) else valid
    seeds = pts[valid & prev_nan]
    b = np.asarray(bounds, dtype=float)
    if len(seeds) >= 2:
        axis = int(np.argmax(seeds.max(axis=0) - seeds.min(axis=0)))  # the rake runs along its longest axis
        return seeds[np.argmin(seeds[:, axis])], seeds[np.argmax(seeds[:, axis])]
    x = b[0] + 0.05 * (b[1] - b[0])
    return np.array([x, b[2], z]), np.array([x, b[3], z])


# --------------------------------------------------------------------------------------------------
# Log bars
# --------------------------------------------------------------------------------------------------
def log_bar_value(x: float, lo: float = 1e-3, hi: float = 1e5) -> int:
    """Bar length 0..100 for x on a log10 scale from `lo` to `hi`.

    A logarithm turns "times 10" into "plus 1", so numbers spread over eight powers of ten share one
    bar:  value = 100 (log10 x - log10 lo) / (log10 hi - log10 lo).  10 sits halfway between 1e-3 and 1e5.
    """
    if not (x > 0.0) or not math.isfinite(x):
        return 0
    frac = (math.log10(x) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    return int(round(100.0 * min(max(frac, 0.0), 1.0)))


def format_si(value: float, unit: str) -> str:
    """Engineering notation with a prefix: 1234 N -> '1.23 kN', 4.5e6 N -> '4.50 MN'."""
    v = abs(value)
    for scale, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if v >= scale:
            return f"{value / scale:.2f} {prefix}{unit}"
    return f"{value:.1f} {unit}"


def _num(metrics: dict, key: str, default: float = 0.0) -> float:
    v = metrics.get(key, default)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


# --------------------------------------------------------------------------------------------------
# Navier-Stokes term bars
# --------------------------------------------------------------------------------------------------
class NsTermsPanel(QFrame):
    """The momentum equation with a log-scale bar per term; the largest term is outlined and marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nsPanel")
        self.setStyleSheet(
            f"QFrame#nsPanel {{ background-color: #111A2B; border: 1px solid #2A3A57; border-radius: 8px; }}"
            f"QLabel {{ border: none; background: transparent; color: {TEXT}; }}"
        )
        self.dominant: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        hdr = QLabel("NAVIER–STOKES: WHICH TERM WINS?")
        hdr.setStyleSheet("font-size: 12px; font-weight: 800; letter-spacing: 1px; color: #7FB7FF;")
        lay.addWidget(hdr)

        self.equation_label = QLabel(EQUATION)
        self.equation_label.setAlignment(Qt.AlignCenter)
        self.equation_label.setStyleSheet(
            f"font-size: 20px; font-weight: 800; color: {TEXT}; background-color: #0A1220;"
            "border-radius: 6px; padding: 6px;"
        )
        lay.addWidget(self.equation_label)

        self.caption = QLabel("size of each term near the model, per unit mass (m/s², log scale)")
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet(f"font-size: 11px; color: {MUTED};")
        lay.addWidget(self.caption)

        self.rows: dict[str, QFrame] = {}
        self.bars: dict[str, QProgressBar] = {}
        self.value_labels: dict[str, QLabel] = {}
        self.name_labels: dict[str, QLabel] = {}
        for name in TERM_NAMES:
            row = QFrame()
            row.setObjectName(f"row_{name}")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(8, 4, 8, 4)
            rl.setSpacing(3)

            title = QLabel(TERM_LABELS_2LINE[name])
            title.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {TERM_COLORS[name]};")
            rl.addWidget(title)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            bar.setStyleSheet(
                "QProgressBar { background-color: #05080F; border: none; border-radius: 6px; }"
                f"QProgressBar::chunk {{ background-color: {TERM_COLORS[name]}; border-radius: 6px; }}"
            )
            val = QLabel("—")
            val.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {TEXT};")
            val.setMinimumWidth(110)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            line = QGridLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(bar, 0, 0)
            line.addWidget(val, 0, 1)
            line.setColumnStretch(0, 1)
            rl.addLayout(line)

            row.setMinimumHeight(row.sizeHint().height())  # never let a squeezed window clip a bar's title
            lay.addWidget(row)
            self.rows[name], self.bars[name], self.value_labels[name], self.name_labels[name] = row, bar, val, title
            self._style_row(name, False)

    def _style_row(self, name: str, dominant: bool) -> None:
        colour = TERM_COLORS[name]
        if dominant:
            style = f"QFrame#row_{name} {{ border: 2px solid {colour}; border-radius: 6px; background-color: #1B2740; }}"
        else:
            style = f"QFrame#row_{name} {{ border: 2px solid transparent; border-radius: 6px; background-color: transparent; }}"
        self.rows[name].setStyleSheet(style)

    def set_terms(self, terms: dict[str, float]) -> None:
        """Update the three bars from {'convective', 'pressure', 'viscous'} (m/s²)."""
        values = {n: _num(terms, n) for n in TERM_NAMES}
        self.dominant = max(TERM_NAMES, key=lambda n: values[n]) if any(v > 0 for v in values.values()) else None
        for name in TERM_NAMES:
            v = values[name]
            self.bars[name].setValue(log_bar_value(v))
            is_dom = name == self.dominant
            text = f"{v:.3g} m/s²" if v > 0 else "—"
            self.value_labels[name].setText(text + ("  ◀ largest" if is_dom else ""))
            self._style_row(name, is_dom)


# --------------------------------------------------------------------------------------------------
# Sustainability cards
# --------------------------------------------------------------------------------------------------
class SustainabilityOverlay(QFrame):
    """Cards laid over the 3D view. `widgets` picks which: 'fuel', 'speed', 'wind_load'."""

    def __init__(self, widgets: tuple[str, ...], parent=None):
        super().__init__(parent)
        self.widgets = tuple(widgets)
        self.setObjectName("sustainOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame#sustainOverlay {{ background-color: {PANEL_BG}; border: 1px solid #1F6F54; border-radius: 10px; }}"
            f"QLabel {{ border: none; background: transparent; color: {TEXT}; }}"
        )
        self.setFixedWidth(400)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(5)

        self._labels: list[QLabel] = []

        hdr = QLabel("SUSTAINABILITY IMPACT")
        hdr.setStyleSheet("font-size: 12px; font-weight: 800; letter-spacing: 1px; color: #5CF2B0;")
        lay.addWidget(hdr)
        self._labels.append(hdr)

        if "fuel" in self.widgets:
            self.fuel_big = self._label(lay, "", "font-size: 22px; font-weight: 800; color: #5CF2B0;")
            self.saved_big = self._label(lay, "", "font-size: 18px; font-weight: 800; color: #5CF2B0;", wrap=True)
            self.co2_big = self._label(lay, "", "font-size: 18px; font-weight: 800; color: #B9F6CA;")
            self.fuel_note = self._label(lay, "", f"font-size: 11px; color: {MUTED};", wrap=True)
        if "speed" in self.widgets:
            self.speed_big = self._label(lay, "", "font-size: 18px; font-weight: 800; color: #FFD740;", wrap=True)
            self.speed_note = self._label(lay, "", f"font-size: 11px; color: {MUTED};", wrap=True)
        if "wind_load" in self.widgets:
            self._label(lay, "WIND LOAD ON THE TOWER", "font-size: 13px; font-weight: 800; color: #FFC400;")
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(3)
            for col, text in enumerate(("Wind", "Speed", "Force", "Base moment")):
                h = QLabel(text)
                h.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {MUTED};")
                grid.addWidget(h, 0, col)
                self._labels.append(h)
            self._table = grid
            self._table_cells: list[list[QLabel]] = []
            lay.addLayout(grid)
            self.worst_label = self._label(lay, "", "font-size: 13px; font-weight: 800; color: #FFC400;")
        self.sdg_label = self._label(lay, "", "font-size: 12px; font-weight: 700; color: #5CF2B0;")
        self.sdg_label.setText(self._sdg_text())

        self.setVisible(bool(self.widgets))

    def fit(self, width: int) -> None:
        """Set the width and the height that matches it (wrapped text needs the height for that width)."""
        self.setFixedWidth(width)
        self.setFixedHeight(max(self.layout().totalHeightForWidth(width), self.layout().totalMinimumSize().height()))

    def _label(self, lay, text: str, style: str, wrap: bool = False) -> QLabel:
        lb = QLabel(text)
        lb.setStyleSheet(style)
        lb.setWordWrap(wrap)
        lay.addWidget(lb)
        self._labels.append(lb)
        return lb

    def _sdg_text(self) -> str:
        tags = []
        if "fuel" in self.widgets or "speed" in self.widgets:
            tags.append("SDG 13 · Climate action")
        if "wind_load" in self.widgets:
            tags.append("SDG 9 · Industry, innovation & infrastructure")
        return "\n".join(tags)

    def all_text(self) -> str:
        """Everything the overlay currently says (used by tests and by the remote status page)."""
        parts = [lb.text() for lb in self._labels]
        parts += [c.text() for row in getattr(self, "_table_cells", []) for c in row]
        return "\n".join(parts)

    def set_metrics(self, metrics: dict, worst_direction_deg: float | None = None) -> None:
        """Show one state's numbers. `worst_direction_deg` is the object's worst solved wind direction
        (ObjectStateManager.worst_direction_deg): one fixed value per object, not tracked while the knob moves."""
        if "fuel" in self.widgets:
            self._set_fuel(metrics)
        if "speed" in self.widgets:
            gain = _num(metrics, "speed_gain_pct")
            self.speed_big.setText(f"Top-speed gain vs worst angle: {gain:+.1f} %")
            self.speed_note.setText("same power: P = ½ρCdAV³, so V ∝ (CdA)^(−1/3), a cube root")
        if "wind_load" in self.widgets:
            self._set_wind_load(metrics, worst_direction_deg)

    def _set_fuel(self, metrics: dict) -> None:
        """Fuel card: burned, saved against the object's own worst simulated angle, CO2 saved.

        0606 link: saved = (P_worst - P) x litres per kWh, a difference of two cubic power laws, so it is
        zero at the worst angle itself. The reference angle is the same for every state of an object, so
        linear interpolation between knob positions leaves it unchanged.
        """
        burn = _num(metrics, "fuel_burn_l_per_hr")
        saved, co2 = _num(metrics, "fuel_saved_l_per_hr"), _num(metrics, "co2_saved_kg_per_hr")
        self.fuel_big.setText(f"Fuel burned: {burn:,.1f} L/h")
        ref = metrics.get("reference_direction_deg")
        has_ref = isinstance(ref, (int, float)) and not isinstance(ref, bool)
        if has_ref and abs(_num(metrics, "direction_deg") - float(ref)) < 1e-6:
            self.saved_big.setText("This is the worst angle tested")
            self.co2_big.setText("")
        else:
            where = f" ({float(ref):g}°)" if has_ref else ""
            self.saved_big.setText(f"Saved vs worst angle{where}: {saved:,.1f} L/h")
            self.co2_big.setText(f"CO₂ saved: {co2:,.1f} kg/h")
        eval_v = _num(metrics, "eval_speed_m_s")
        at_cruise = eval_v > 0.0 and abs(eval_v - _num(metrics, "wind_speed_m_s")) > 1e-6
        altitude = _num(metrics, "eval_altitude_km")  # from the config's cruise block; 0 = not given
        where_cruise = f"{altitude:g} km, " if altitude > 0.0 else ""
        self.fuel_note.setText(f"at cruise ({where_cruise}{eval_v:.0f} m/s)" if at_cruise else "")

    def _set_wind_load(self, metrics: dict, worst_direction_deg: float | None) -> None:
        rows = metrics.get("wind_load") or []
        while len(self._table_cells) < len(rows):  # grow the table once, then only change the text
            r = len(self._table_cells) + 1
            cells = []
            for col in range(4):
                c = QLabel("")
                c.setStyleSheet(f"font-size: 12px; font-weight: {'700' if col == 0 else '600'}; color: {TEXT};")
                self._table.addWidget(c, r, col)
                cells.append(c)
            self._table_cells.append(cells)
        for cells, row in zip(self._table_cells, rows):
            cells[0].setText(str(row.get("label", "")))
            cells[1].setText(f"{_num(row, 'v_m_s'):.0f} m/s")
            cells[2].setText(format_si(_num(row, "force_n"), "N"))
            cells[3].setText(format_si(_num(row, "moment_nm"), "N·m"))

        if worst_direction_deg is None:
            self.worst_label.setText("")
        else:
            self.worst_label.setText(f"Worst wind direction: {worst_direction_deg:.1f}°")


# --------------------------------------------------------------------------------------------------
# Colour bar and legend (Qt-drawn: VTK's built-in fonts have no Greek letters and overlap their labels)
# --------------------------------------------------------------------------------------------------
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def compact_si(v: float) -> str:
    """Short number text: -3540 -> '-3.5k', 52.4 -> '52.4', 1.85e6 -> '1.9M', 0.0123 -> '0.012'."""
    if v == 0.0 or not math.isfinite(v):
        return "0" if v == 0.0 else str(v)
    a = abs(v)
    for scale, prefix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if a >= scale:
            return f"{v / scale:.2g}{prefix}" if abs(v / scale) >= 10 else f"{v / scale:.1f}{prefix}".replace(".0", "")
    return f"{v:.3g}" if a >= 1 else f"{v:.2g}"


def colorbar_ticks(lo: float, hi: float, log: bool, n_linear: int = 5) -> list[tuple[float, str]]:
    """[(position 0..1 along the bar, label)] for a colour bar from lo to hi.

    Linear: `n_linear` (5 or 3) evenly spaced values in compact SI text. Log: one tick per power of ten inside the range (a logarithm turns
    "times 10" into "plus 1", so decades are evenly spaced); fewer than two decades fall back to three.
    """
    if not (hi > lo):
        return []
    if log and lo > 0:
        a, b = math.log10(lo), math.log10(hi)
        decades = list(range(math.ceil(a), math.floor(b) + 1))
        if len(decades) >= 2:
            return [((e - a) / (b - a), "10" + str(e).translate(_SUPERSCRIPT)) for e in decades]
        mids = [lo, math.sqrt(lo * hi), hi]  # geometric middle: halfway on the log scale
        return [((math.log10(v) - a) / (b - a), compact_si(v)) for v in mids]
    fracs = (0.0, 0.5, 1.0) if n_linear <= 3 else (0.0, 0.25, 0.5, 0.75, 1.0)
    return [(f, compact_si(lo + f * (hi - lo))) for f in fracs]


class ColorBar(QWidget):
    """A titled horizontal colour bar with tick labels, drawn with QPainter."""

    def __init__(self, title: str, cmap: str, log: bool = False, parent=None):
        super().__init__(parent)
        self.title, self.log = title, log
        self.lo, self.hi = 0.0, 1.0
        import matplotlib

        cm = matplotlib.colormaps[cmap]
        self._colors = [QColor.fromRgbF(*cm(i / 31.0)[:3]) for i in range(32)]
        self.setFixedHeight(66)
        self.setAttribute(Qt.WA_StyledBackground, False)

    def set_range(self, lo: float, hi: float) -> None:
        self.lo, self.hi = float(lo), float(hi)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(8, 13, 24, 225))
        p.setPen(QPen(QColor("#2A3A57"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        p.setPen(QColor(TEXT))
        f = QFont(self.font())
        f.setPixelSize(13)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(8, 4, self.width() - 16, 18), Qt.AlignCenter, self.title)

        x0, x1, y0, h = 38.0, self.width() - 38.0, 26.0, 14.0
        grad = QLinearGradient(QPointF(x0, 0), QPointF(x1, 0))
        for i, c in enumerate(self._colors):
            grad.setColorAt(i / (len(self._colors) - 1), c)
        p.fillRect(QRectF(x0, y0, x1 - x0, h), grad)
        p.setPen(QPen(QColor("#CBD5E1"), 1))
        p.drawRect(QRectF(x0, y0, x1 - x0, h))
        f.setBold(False)
        f.setPixelSize(12)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        n = 5 if self.width() >= 460 else 3  # fewer labels on a narrow bar so they never touch
        for frac, text in colorbar_ticks(self.lo, self.hi, self.log, n):
            x = x0 + frac * (x1 - x0)
            p.drawLine(QPointF(x, y0 + h), QPointF(x, y0 + h + 4))
            p.drawText(QRectF(x - 40, y0 + h + 5, 80, 16), Qt.AlignCenter, text)
        p.end()


class TermLegend(QFrame):
    """Legend for the dominant-term streamline colours (view 2): a colour swatch per Navier-Stokes term."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("termLegend")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QFrame#termLegend {{ background-color: {PANEL_BG}; border: 1px solid #2A3A57; border-radius: 8px; }}"
            f"QLabel {{ border: none; background: transparent; color: {TEXT}; }}"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)
        hdr = QLabel("Streamline colour = largest term")
        hdr.setStyleSheet(f"font-size: 12px; font-weight: 800; color: {MUTED};")
        lay.addWidget(hdr)
        self.entries: dict[str, QLabel] = {}
        for name in TERM_NAMES:
            row = QHBoxLayout()
            swatch = QFrame()
            swatch.setFixedSize(26, 6)
            swatch.setStyleSheet(f"background-color: {TERM_COLORS[name]}; border: none; border-radius: 3px;")
            text = QLabel(TERM_LABELS_2LINE[name])
            text.setStyleSheet("font-size: 12px; font-weight: 600;")
            row.addWidget(swatch)
            row.addWidget(text)
            row.addStretch()
            lay.addLayout(row)
            self.entries[name] = text
