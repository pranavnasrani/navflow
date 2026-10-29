"""
HUD building blocks for the monitor windows (native Qt widgets, no web content).

  * `polyline_from_nan_joined`  NaN-joined streamline points  ->  a PyVista PolyData of line cells.
  * `particles_on_lines`        flow tracer positions that slide along those lines.
  * `log_bar_value`             0..100 bar length for a number that spans many powers of ten.
  * `NsTermsPanel`              the equation  rho (v.grad)v = -grad p + mu laplacian(v)  with one bar per term.
  * `SustainabilityOverlay`     the fuel / CO2, top-speed and wind-load lines with their UN SDG tag.
  * `rainbow_cmap`              the banded blue-to-red colour scale of the results window.
  * `MissionPanel`              the visitor's challenge: a question, one big number and a graph of all nine runs.

Maths link (IGCSE 0606): logarithms (the bars), linear interpolation (the tracers), the cube root and
the squared-speed rule (the sustainability numbers, derived in solver/foam/sustainability.py).
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget

# --------------------------------------------------------------------------------------------------
# Colours. Three fixed colours name the three Navier-Stokes terms everywhere (bars, streamlines in
# view 2, the legend, the A2 poster). Chosen to stay apart for colour-blind viewers (Okabe-Ito family)
# and to read on the light, Fluent-style results window.
# --------------------------------------------------------------------------------------------------
TERM_NAMES = ("convective", "pressure", "viscous")
TERM_COLORS = {"convective": "#0072B2", "pressure": "#D55E00", "viscous": "#B4428E"}
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

# Results-window theme (light report panel beside a gradient graphics window, like CFD post-processors)
PANEL_BG = "#ECEEF1"
TEXT = "#1D232B"
MUTED = "#5B6573"
RULE = "#C7CDD5"

# Banded rainbow used by CFD post-processors: blue (low) - cyan - green - yellow - red (high).
RAINBOW_STOPS = ("#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000")
N_BANDS = 20


def rainbow_cmap(n_bands: int = N_BANDS):
    """The banded rainbow as a matplotlib colormap with `n_bands` flat colours (0 = continuous)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("cfd_rainbow", RAINBOW_STOPS)
    return cmap.resampled(n_bands) if n_bands else cmap


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
def _role(widget: QWidget, role: str) -> QWidget:
    """Tag a widget for the window's style sheet (QLabel[role="value"] ...), so one sheet sets every size."""
    widget.setProperty("role", role)
    return widget


def _repolish(widget: QWidget) -> None:
    """Re-apply the style sheet after a dynamic property changed (Qt does not do it by itself)."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class NsTermsPanel(QFrame):
    """The momentum equation with a log-scale bar per term; the largest term is set in bold and marked."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("nsPanel")
        self.dominant: str | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.equation_label = _role(QLabel(EQUATION), "equation")
        lay.addWidget(self.equation_label)

        self.caption = _role(QLabel("size of each term near the model, per unit mass (m/s², log scale)"), "small")
        self.caption.setWordWrap(True)
        lay.addWidget(self.caption)

        self.rows: dict[str, QFrame] = {}
        self.bars: dict[str, QProgressBar] = {}
        self.value_labels: dict[str, QLabel] = {}
        self.name_labels: dict[str, QLabel] = {}
        for name in TERM_NAMES:
            row = QFrame()
            row.setObjectName(f"row_{name}")
            rl = QVBoxLayout(row)
            rl.setContentsMargins(0, 2, 0, 2)
            rl.setSpacing(3)

            title = _role(QLabel(TERM_LABELS_2LINE[name]), "term")
            title.setStyleSheet(f"color: {TERM_COLORS[name]};")
            rl.addWidget(title)

            bar = QProgressBar()
            bar.setObjectName(f"bar_{name}")
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            bar.setStyleSheet(
                f"QProgressBar {{ background-color: #D6DBE1; border: none; border-radius: 0px; }}"
                f"QProgressBar::chunk {{ background-color: {TERM_COLORS[name]}; border-radius: 0px; }}"
            )
            val = _role(QLabel("—"), "termValue")
            val.setMinimumWidth(130)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            line = QGridLayout()
            line.setContentsMargins(0, 0, 0, 0)
            line.setHorizontalSpacing(10)
            line.addWidget(bar, 0, 0)
            line.addWidget(val, 0, 1)
            line.setColumnStretch(0, 1)
            rl.addLayout(line)

            lay.addWidget(row)
            self.rows[name], self.bars[name], self.value_labels[name], self.name_labels[name] = row, bar, val, title

    def set_terms(self, terms: dict[str, float]) -> None:
        """Update the three bars from {'convective', 'pressure', 'viscous'} (m/s²)."""
        values = {n: _num(terms, n) for n in TERM_NAMES}
        self.dominant = max(TERM_NAMES, key=lambda n: values[n]) if any(v > 0 for v in values.values()) else None
        for name in TERM_NAMES:
            v = values[name]
            self.bars[name].setValue(log_bar_value(v))
            is_dom = name == self.dominant
            text = (f"{v:,.0f} m/s²" if v >= 100 else f"{v:.3g} m/s²") if v > 0 else "—"
            self.value_labels[name].setText(text + ("  ◀ largest" if is_dom else ""))
            self.value_labels[name].setProperty("dominant", is_dom)
            _repolish(self.value_labels[name])


# --------------------------------------------------------------------------------------------------
# Sustainability section
# --------------------------------------------------------------------------------------------------
class SustainabilityOverlay(QFrame):
    """The sustainability section of the report panel. `widgets` picks its parts: 'fuel', 'speed', 'wind_load'."""

    def __init__(self, widgets: tuple[str, ...], parent=None):
        super().__init__(parent)
        self.widgets = tuple(widgets)
        self.setObjectName("sustain")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._labels: list[QLabel] = []

        if "fuel" in self.widgets:
            self.fuel_big = self._label(lay, "", "big")
            self.saved_big = self._label(lay, "", "value", wrap=True)
            self.co2_big = self._label(lay, "", "value")
            self.fuel_note = self._label(lay, "", "small", wrap=True)
        if "speed" in self.widgets:
            self.speed_big = self._label(lay, "", "value", wrap=True)
            self.speed_note = self._label(lay, "", "small", wrap=True)
        if "wind_load" in self.widgets:
            grid = QGridLayout()
            grid.setHorizontalSpacing(12)
            grid.setVerticalSpacing(3)
            for col, text in enumerate(("Wind", "Speed", "Force", "Base moment")):
                h = _role(QLabel(text), "key")
                grid.addWidget(h, 0, col, alignment=Qt.AlignLeft if col == 0 else Qt.AlignRight)
                self._labels.append(h)
            self._table = grid
            self._table_cells: list[list[QLabel]] = []
            lay.addLayout(grid)
            self.worst_label = self._label(lay, "", "value")
        self.sdg_label = self._label(lay, "", "small")
        self.sdg_label.setText(self._sdg_text())

        self._detailed = False
        self.set_detailed(False)
        self.setVisible(bool(self.widgets))

    def set_detailed(self, detailed: bool) -> None:
        """Simple: the headline and one supporting line (for the tower, the strongest wind only).
        Detailed: also the CO2 line, the formula notes and every row of the wind-load table."""
        self._detailed = bool(detailed)
        for name in ("co2_big", "fuel_note", "speed_note"):
            lb = getattr(self, name, None)
            if lb is not None:
                lb.setVisible(self._detailed)
        self._show_table_rows()

    def _show_table_rows(self) -> None:
        """The wind-load table keeps its last row (the strongest wind) in the simple view."""
        rows = getattr(self, "_table_cells", [])
        for i, cells in enumerate(rows):
            for c in cells:
                c.setVisible(self._detailed or i == len(rows) - 1)

    def fit(self, width: int) -> None:
        """Set the width and the height that matches it (wrapped text needs the height for that width)."""
        self.setFixedWidth(width)
        self.setFixedHeight(max(self.layout().totalHeightForWidth(width), self.layout().totalMinimumSize().height()))

    def _label(self, lay, text: str, role: str, wrap: bool = False) -> QLabel:
        lb = _role(QLabel(text), role)
        lb.setWordWrap(wrap)
        lay.addWidget(lb)
        self._labels.append(lb)
        return lb

    def _sdg_text(self) -> str:
        tags = []
        if "fuel" in self.widgets or "speed" in self.widgets:
            tags.append("SDG 9 · Industry, innovation & infrastructure")  # the plane and the car
        if "wind_load" in self.widgets:
            tags.append("SDG 11 · Sustainable cities & communities")  # the tower
        return "   ".join(tags)

    def all_text(self) -> str:
        """Everything the section currently says (used by tests and by the remote status page)."""
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
        """Fuel lines: burned, saved against the object's own worst simulated angle, CO2 saved.

        0606 link: saved = (P_worst - P) x litres per kWh, a difference of two cubic power laws, so it is
        zero at the worst angle itself. The reference angle is the same for every state of an object, so
        linear interpolation between knob positions leaves it unchanged.
        """
        burn = _num(metrics, "fuel_burn_l_per_hr")
        saved, co2 = _num(metrics, "fuel_saved_l_per_hr"), _num(metrics, "co2_saved_kg_per_hr")
        ref = metrics.get("reference_direction_deg")
        has_ref = isinstance(ref, (int, float)) and not isinstance(ref, bool)
        # An aircraft is compared with its BEST angle (highest L/D), so it shows extra fuel, not fuel saved.
        best = bool(metrics.get("reference_is_best"))
        if best:
            self._set_fuel_vs_best(metrics, burn, saved, ref if has_ref else None)
            return
        self.fuel_big.setText(f"Fuel burned: {burn:,.1f} L/h")
        if has_ref and abs(_num(metrics, "direction_deg") - float(ref)) < 1e-6:
            self.saved_big.setText("This is the best angle tested" if best else "This is the worst angle tested")
            self.co2_big.setText("")
        else:
            where = f" ({float(ref):g}°)" if has_ref else ""
            if best:
                self.saved_big.setText(f"Extra vs best angle{where}: {saved:,.1f} L/h")
                self.co2_big.setText(f"Extra CO₂: {co2:,.1f} kg/h")
            else:
                self.saved_big.setText(f"Saved vs worst angle{where}: {saved:,.1f} L/h")
                self.co2_big.setText(f"CO₂ saved: {co2:,.1f} kg/h")
        eval_v = _num(metrics, "eval_speed_m_s")
        at_cruise = eval_v > 0.0 and abs(eval_v - _num(metrics, "wind_speed_m_s")) > 1e-6
        altitude = _num(metrics, "eval_altitude_km")  # from the config's cruise block; 0 = not given
        where_cruise = f"{altitude:g} km, " if altitude > 0.0 else ""
        note = f"at cruise ({where_cruise}{eval_v:.0f} m/s)" if at_cruise else ""
        self.fuel_note.setText(note)

    def _set_fuel_vs_best(self, metrics: dict, burn: float, extra: float, ref) -> None:
        """Aircraft: extra fuel as a PERCENTAGE of the fuel at the best angle, never as litres.

        Level flight: lift = weight, so drag = weight / (L/D) and fuel is proportional to 1 / (L/D). Our
        absolute drag is too high (the mesh is still coarse for a 79 m wing), but the RATIO between two angles
        is reliable, so the panel shows only ratios:
            extra % = extra / (burn - extra) x 100 = (L/D_best / L/D - 1) x 100,
            L/D_best = L/D x burn / (burn - extra)      (both from fuel proportional to 1 / (L/D)).
        """
        ld = _num(metrics, "lift_to_drag")
        at_best = ref is not None and abs(_num(metrics, "direction_deg") - float(ref)) < 1e-6
        base = burn - extra  # fuel at the best angle
        where = f" ({float(ref):g}°)" if ref is not None else ""
        if at_best or base <= 0.0:
            self.fuel_big.setText("This is the best angle tested")
            self.saved_big.setText(f"Lift ÷ drag = {ld:.1f}, the highest of all nine angles" if ld > 0 else "")
        else:
            pct = extra / base * 100.0
            self.fuel_big.setText(f"Fuel vs best angle{where}: +{pct:,.0f} %")
            self.saved_big.setText(f"Lift ÷ drag = {ld:.1f} here, {ld * burn / base:.1f} at the best angle")
        self.co2_big.setText("CO₂ changes by the same percentage as the fuel")
        self.fuel_note.setText("level flight: lift = weight, so drag = weight ÷ (L/D) and fuel ∝ 1 ÷ (L/D)")

    def _set_wind_load(self, metrics: dict, worst_direction_deg: float | None) -> None:
        rows = metrics.get("wind_load") or []
        while len(self._table_cells) < len(rows):  # grow the table once, then only change the text
            r = len(self._table_cells) + 1
            cells = []
            for col in range(4):
                c = _role(QLabel(""), "value" if col else "key")
                self._table.addWidget(c, r, col, alignment=Qt.AlignLeft if col == 0 else Qt.AlignRight)
                cells.append(c)
            self._table_cells.append(cells)
        for cells, row in zip(self._table_cells, rows):
            cells[0].setText(str(row.get("label", "")))
            cells[1].setText(f"{_num(row, 'v_m_s'):.0f} m/s")
            cells[2].setText(format_si(_num(row, "force_n"), "N"))
            cells[3].setText(format_si(_num(row, "moment_nm"), "N·m"))
        self._show_table_rows()

        if worst_direction_deg is None:
            self.worst_label.setText("")
        else:
            self.worst_label.setText(f"Worst wind direction: {worst_direction_deg:.1f}°")


# --------------------------------------------------------------------------------------------------
# Colour-bar tick labels
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


# --------------------------------------------------------------------------------------------------
# The visitor's challenge
# --------------------------------------------------------------------------------------------------
LINE_BLUE = "#2563EB"
DOT_ORANGE = "#EA580C"
FOUND_GREEN = "#16A34A"


class MissionChart(QWidget):
    """The challenge's quantity against direction: one dot per solved run, straight lines between them, and a
    big orange dot for where the knob is now.

    Maths link (IGCSE 0606): between two runs the graph is a straight line, y = (1 - t) y1 + t y2 with
    t = (d - d1) / (d2 - d1), which is exactly how the app blends the results as the knob turns; the
    orange dot slides along that line. Points map to the screen by a linear scale on each axis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.best: int | None = None
        self.current: tuple[float, float] | None = None
        self.reveal = False
        self.decimals = 1
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_curve(self, xs, ys, best: int | None, decimals: int = 1) -> None:
        self.xs, self.ys, self.best, self.decimals = [float(x) for x in xs], [float(y) for y in ys], best, decimals
        self.update()

    def set_current(self, x: float, y: float, reveal: bool) -> None:
        self.current, self.reveal = (float(x), float(y)), bool(reveal)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if len(self.xs) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        fpx = max(9, int(h * 0.085))
        p.setFont(QFont("Arial", fpx))
        left, right, top, bottom = 4.2 * fpx, 10.0, 1.4 * fpx, 2.0 * fpx
        x0, x1 = self.xs[0], self.xs[-1]
        lo, hi = min(self.ys), max(self.ys)
        pad = 0.12 * (hi - lo or 1.0)
        lo, hi = lo - pad, hi + pad

        def px(x: float) -> float:  # linear scale: x0 -> left edge, x1 -> right edge
            return left + (x - x0) / (x1 - x0) * (w - left - right)

        def py(y: float) -> float:
            return top + (1.0 - (y - lo) / (hi - lo)) * (h - top - bottom)

        # axes and their end labels
        p.setPen(QPen(QColor(RULE), 1.0))
        p.drawLine(QPointF(left, h - bottom), QPointF(w - right, h - bottom))
        p.drawLine(QPointF(left, top), QPointF(left, h - bottom))
        p.setPen(QColor(MUTED))
        fmt = f"{{:.{self.decimals}f}}"
        for y in (min(self.ys), max(self.ys)):
            p.drawText(QRectF(0, py(y) - fpx, left - 6, 2 * fpx), Qt.AlignRight | Qt.AlignVCenter, fmt.format(y))
        for x in (x0, self.xs[len(self.xs) // 2], x1):
            p.drawText(QRectF(px(x) - 3 * fpx, h - bottom + 2, 6 * fpx, bottom), Qt.AlignHCenter | Qt.AlignTop, f"{x:g}°")

        # the nine runs, joined by straight lines
        path = QPainterPath(QPointF(px(self.xs[0]), py(self.ys[0])))
        for x, y in zip(self.xs[1:], self.ys[1:]):
            path.lineTo(QPointF(px(x), py(y)))
        p.setPen(QPen(QColor(LINE_BLUE), 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(LINE_BLUE))
        for x, y in zip(self.xs, self.ys):
            p.drawEllipse(QPointF(px(x), py(y)), 3.5, 3.5)

        # the answer, once found
        if self.reveal and self.best is not None:
            bx, by = px(self.xs[self.best]), py(self.ys[self.best])
            p.setPen(QPen(QColor(FOUND_GREEN), 3.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(bx, by), 10.0, 10.0)

        # where the knob is now
        if self.current is not None:
            cx, cy = px(min(max(self.current[0], x0), x1)), py(self.current[1])
            p.setPen(QPen(QColor(DOT_ORANGE), 1.2, Qt.DashLine))
            p.drawLine(QPointF(cx, top), QPointF(cx, h - bottom))
            p.setPen(QPen(QColor("white"), 2.0))
            p.setBrush(QColor(DOT_ORANGE))
            p.drawEllipse(QPointF(cx, cy), 7.5, 7.5)
        p.end()


class MissionPanel(QFrame):
    """Question, big live number, graph of all nine runs, and a hint that turns into "You found it!"."""

    def __init__(self, mission, parent=None):
        super().__init__(parent)
        self.mission = mission
        self.found = False
        self._xs: np.ndarray = np.zeros(0)
        self._ys: np.ndarray = np.zeros(0)
        self._best = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.question = _role(QLabel(mission.question), "mission")
        self.question.setWordWrap(True)
        lay.addWidget(self.question)

        row = QHBoxLayout()
        row.addWidget(_role(QLabel(mission.label), "key"), alignment=Qt.AlignBottom)
        row.addStretch()
        self.value = _role(QLabel("—"), "hero")
        row.addWidget(self.value, alignment=Qt.AlignBottom)
        lay.addLayout(row)

        self.chart = MissionChart()
        lay.addWidget(self.chart)
        self.chart_note = _role(QLabel("● one full simulation    ━ straight line between them    ● you"), "small")
        lay.addWidget(self.chart_note)

        self.hint = _role(QLabel(""), "hint")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

    def set_curve(self, xs, ys) -> None:
        self._xs, self._ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
        self._best = int(np.argmax(self._ys) if self.mission.goal == "max" else np.argmin(self._ys))
        self.chart.set_curve(self._xs, self._ys, self._best, self.mission.decimals)

    @property
    def best_direction(self) -> float:
        return float(self._xs[self._best])

    def update_direction(self, direction: float, show_mode: bool = False) -> bool:
        """Move the dot to `direction` and return True if the visitor is on the answer.

        The number is read off the straight line between the two neighbouring runs (np.interp is
        y = (1 - t) y1 + t y2), so the big number and the dot always agree. "On the answer" means within
        half a step of the best run: |d - d_best| <= (d2 - d1) / 2.
        """
        if self._xs.size < 2:
            return False
        y = float(np.interp(direction, self._xs, self._ys))
        unit = f" {self.mission.unit}" if self.mission.unit else ""
        self.value.setText(f"{y:,.{self.mission.decimals}f}{unit}")
        half_step = 0.5 * float(np.min(np.diff(self._xs)))
        self.found = abs(direction - self.best_direction) <= half_step + 1e-9
        self.chart.set_current(direction, y, reveal=self.found and not show_mode)
        if show_mode:
            self.hint.setText("Watch the orange dot move along the graph as the wind turns.")
            self.hint.setProperty("found", False)
        elif self.found:
            self.hint.setText(self.mission.found)
            self.hint.setProperty("found", True)
        else:
            turn = "clockwise ↻" if direction < self.best_direction else "anticlockwise ↺"
            self.hint.setText(f"{self.mission.better}  Try turning the knob {turn}.")
            self.hint.setProperty("found", False)
        _repolish(self.hint)
        return self.found and not show_mode
