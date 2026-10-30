"""
Draw the NavFlow Remote's app icons (app/remote/icon-*.png): air flowing round a cylinder in the three
Navier-Stokes term colours, on the dark ground of the dark theme.

The streamlines are the exact ones of ideal flow past a circle of radius a in a stream of speed 1:
    stream function  psi = y (1 - a^2 / (x^2 + y^2)),   each line is a level set psi = const.
(A straight line y = psi far away that bends round the circle.) Run:  ./.venv/bin/python scripts/make_remote_icons.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "app" / "remote"
GROUND, BODY = "#10151B", "#E8EDF3"
LEVELS = ((0.42, "#E69F00"), (0.68, "#56B4E9"), (1.05, "#CC79A7"))  # orange, sky blue, pink: the term colours
A = 0.34  # cylinder radius


def draw(px: int, path: Path) -> None:
    fig = plt.figure(figsize=(px / 100, px / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)
    half = 1.6                       # the art lives in the middle 80%, so a rounded mask never clips it
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.axis("off")
    x, y = np.meshgrid(np.linspace(-half, half, 900), np.linspace(-half, half, 900))
    r2 = x**2 + y**2
    psi = np.where(r2 > A**2, y * (1.0 - A**2 / np.maximum(r2, 1e-9)), np.nan)
    for level, colour in LEVELS:
        for sign in (1, -1):
            ax.contour(x, y, psi, levels=[sign * level], colors=[colour], linewidths=px / 52, linestyles="solid")
    ax.add_patch(plt.Circle((0, 0), A, color=BODY, zorder=3))
    fig.savefig(path, dpi=100, facecolor=GROUND)
    plt.close(fig)


if __name__ == "__main__":
    for size, name in ((180, "icon-180.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        draw(size, OUT / name)
        print("wrote", OUT / name)
