"""
Typeset maths for the report panel.

Qt labels cannot set an equation properly (no italic Greek, no real superscripts, no fractions), so the
formulas are drawn by matplotlib's maths engine in the STIX maths font and shown as pictures:

    MathLabel([("\\rho", ink), ("(\\mathbf{v}\\cdot\\nabla)\\mathbf{v}", blue), ...], base_px=30)

Each piece is (TeX, colour), so one equation can carry the three term colours of the bars below it. Pieces
share a baseline and sit `GAP_EM` apart. The picture is drawn at the screen's pixel ratio (crisp on Retina)
and redrawn when the panel changes scale. Every picture is cached, so nothing is typeset while the exhibit
runs after the first frame.

Maths link (IGCSE 0606): the formulas on screen are the ones the exhibition explains, e.g. the momentum
equation  rho (v . grad)v = -grad p + mu laplacian(v)  and the drag power  P = 1/2 rho Cd A V^3.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

GAP_EM = 0.34            # space between two pieces, in units of the font size
ABOVE_EM, BELOW_EM = 1.05, 0.45   # height kept above / below the baseline, so every picture lines up
MATH_RC = {"mathtext.fontset": "stix", "font.family": "STIXGeneral"}

Segment = tuple[str, str]  # (TeX, "#rrggbb")


@lru_cache(maxsize=512)
def render_math(segments: tuple[Segment, ...], px: int, dpr: float = 1.0) -> QImage:
    """Draw the pieces left to right on one baseline; returns a transparent image `px * dpr` tall per em."""
    size = max(6.0, px * dpr)                       # font size in device pixels (the figure is 72 dpi)
    cols = int(size * (4 + 2.2 * sum(len(tex) for tex, _ in segments)))
    rows = int(size * (ABOVE_EM + BELOW_EM) + 2)
    base = size * ABOVE_EM + 1                      # baseline, measured down from the top
    with rc_context(MATH_RC):
        fig = Figure(figsize=(cols / 72.0, rows / 72.0), dpi=72)
        fig.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(fig)
        renderer = canvas.get_renderer()
        x = size * 0.08
        for i, (tex, colour) in enumerate(segments):
            text = fig.text(x / cols, 1.0 - base / rows, f"${tex}$", fontsize=size, color=colour,
                            ha="left", va="baseline")
            x += text.get_window_extent(renderer).width + (GAP_EM * size if i < len(segments) - 1 else 0.0)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba()).copy()
    ink = np.flatnonzero(rgba[:, :, 3].max(axis=0))
    right = int(ink[-1]) + 2 if ink.size else 1
    rgba = np.ascontiguousarray(rgba[:, :max(1, min(right, cols)), :])
    img = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.shape[1] * 4, QImage.Format_RGBA8888).copy()
    img.setDevicePixelRatio(dpr)
    return img


class MathLabel(QLabel):
    """A typeset formula as a label. `text()` still answers with the plain-text form, for tests and logs."""

    def __init__(self, segments: list[Segment], base_px: int, plain: str = "", parent=None):
        super().__init__(parent)
        self.segments = tuple(segments)
        self.base_px = base_px
        self.plain = plain
        self._scale = 1.0
        self._drawn: tuple[int, float] | None = None
        self.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setProperty("role", "math")
        self._redraw()

    def text(self) -> str:  # noqa: D102 - QLabel.text() is not virtual, direct calls are what matter here
        return self.plain

    def set_segments(self, segments: list[Segment]) -> None:
        """New pieces or new colours (a theme change); the picture is redrawn."""
        self.segments = tuple(segments)
        self._drawn = None
        self._redraw()

    def set_scale(self, s: float) -> None:
        """Panel scale (1.0 = a 1920 x 1080 monitor): the same factor the style sheet's pixel sizes use."""
        self._scale = s
        self._redraw()

    def _redraw(self) -> None:
        px = max(9, int(round(self.base_px * self._scale)))
        dpr = float(self.devicePixelRatioF())
        if self._drawn == (px, dpr):
            return
        self._drawn = (px, dpr)
        img = render_math(self.segments, px, dpr)
        self.setPixmap(QPixmap.fromImage(img))
        self.setFixedHeight(int(round(img.height() / dpr)))

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        self._redraw()  # the pixel ratio is only known once the label sits on a screen
