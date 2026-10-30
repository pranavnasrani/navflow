"""Tests for app.math_render: typeset formulas become crisp, transparent, baseline-aligned pictures."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from app.math_render import ABOVE_EM, BELOW_EM, MathLabel, render_math


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_render_is_transparent_and_has_ink(qapp):
    img = render_math(((r"\nabla^{2}\mathbf{v}", "#B4428E"),), 28, 1.0)
    assert img.hasAlphaChannel()
    assert img.pixelColor(0, 0).alpha() == 0  # background is clear
    assert any(img.pixelColor(x, y).alpha() > 200 for x in range(img.width()) for y in range(img.height()))


def test_height_is_fixed_by_the_font_size_not_the_formula(qapp):
    flat = render_math(((r"\rho", "#000000"),), 30, 1.0)
    tall = render_math(((r"\frac{1}{\rho}\nabla p", "#000000"),), 30, 1.0)
    assert flat.height() == tall.height() == int(30 * (ABOVE_EM + BELOW_EM) + 2)


def test_retina_picture_has_twice_the_pixels(qapp):
    one, two = render_math(((r"\nu", "#000000"),), 24, 1.0), render_math(((r"\nu", "#000000"),), 24, 2.0)
    assert two.height() == pytest.approx(2 * one.height(), abs=3)
    assert two.devicePixelRatio() == 2.0


def test_pieces_keep_their_own_colour_and_widen_the_picture(qapp):
    one = render_math(((r"a", "#0072B2"),), 30, 1.0)
    two = render_math(((r"a", "#0072B2"), ("=", "#1D232B"), (r"b", "#D55E00")), 30, 1.0)
    assert two.width() > one.width()
    seen = {(c.red(), c.green(), c.blue()) for x in range(two.width()) for y in range(two.height())
            if (c := two.pixelColor(x, y)).alpha() == 255}
    assert (0, 114, 178) in seen and (213, 94, 0) in seen


def test_label_reports_plain_text_and_follows_the_scale(qapp):
    label = MathLabel([(r"\mu", "#000000")], base_px=30, plain="mu")
    assert label.text() == "mu"
    h1 = label.height()
    label.set_scale(2.0)
    assert label.height() > h1
