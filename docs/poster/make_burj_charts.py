"""Charts for the Burj Khalifa poster, as SVG, from the tower's full OpenFOAM results.

Run: ./.venv/bin/python docs/poster/make_burj_charts.py <building_states.npz>
Writes docs/poster/img/burj_*.svg. Every number plotted comes from the cache or the run settings.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.cache_format import read_cache  # noqa: E402

OUT = Path(__file__).parent / "img"
INK, INK2, RULE = "#141414", "#555555", "#D8D8D8"
WIND, SIDE, LEE, ACCENT = "#D55E00", "#0072B2", "#8A8A8A", "#FD9D24"
FONT = "Avenir Next, Avenir, sans-serif"
MATH = "STIX Two Text, STIXGeneral, serif"

U_ROOF, Z0, H = 50.0, 0.5, 828.0  # the tower run: 50 m/s at the roof, ground roughness z0 = 0.5 m


def u_log(z):
    """The inlet wind profile the solver used: u(z) = U_roof ln((z + z0)/z0) / ln((H + z0)/z0)."""
    return U_ROOF * np.log((z + Z0) / Z0) / np.log((H + Z0) / Z0)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}">{body}</svg>')


def text(x, y, s, size=3.6, color=INK2, anchor="start", weight=400, family=FONT, italic=False):
    st = ' font-style="italic"' if italic else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family="{family}"{st}>{s}</text>')


def line(x1, y1, x2, y2, color=RULE, w=0.3, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{w}"{d}/>'


def path(xs, ys, color, w=0.9, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round"{d}/>'


def height_axis(x0, y0, h, zmax, ticks, label=True):
    """Vertical height axis (z up). Returns (svg, z->y)."""
    zy = lambda z: y0 + h - h * z / zmax  # noqa: E731
    s = line(x0, y0, x0, y0 + h, INK, 0.3)
    for z in ticks:
        s += line(x0 - 1.2, zy(z), x0, zy(z), INK, 0.3) + text(x0 - 2, zy(z) + 1.2, f"{z:g}", 3.2, anchor="end")
    if label:
        s += text(1, y0 - 4, "height (m)", 3.2)
    return s, zy


def wind_profile():
    """u(z) from the log law, 0 to 900 m."""
    w, h = 80, 120
    x0, y0, ph, pw = 14, 10, 98, 60
    zmax, umax = 900, 55
    ax, zy = height_axis(x0, y0, ph, zmax, (0, 200, 400, 600, 828))
    ux = lambda u: x0 + pw * u / umax  # noqa: E731
    s = ax + line(x0, y0 + ph, x0 + pw, y0 + ph, INK, 0.3)
    for u in (0, 25, 50):
        s += line(ux(u), y0 + ph, ux(u), y0 + ph + 1.2, INK, 0.3) + text(ux(u), y0 + ph + 4.6, f"{u}", 3.2, anchor="middle")
    s += text(x0 + pw, y0 + ph + 9.4, "wind speed u (m/s)", 3.2, anchor="end")
    s += line(x0, zy(H), x0 + pw, zy(H), RULE, 0.3, "1 1")
    z = np.concatenate([np.linspace(0.05, 20, 60), np.linspace(20, zmax, 120)])
    s += path(ux(u_log(z)), zy(z), INK, 1.0)
    for zz in (10, 100, 828):
        uu = float(u_log(zz))
        s += f'<circle cx="{ux(uu):.2f}" cy="{zy(zz):.2f}" r="0.9" fill="{ACCENT}"/>'
        right = zz < 500
        s += text(ux(uu) + (2.2 if right else -1.8), zy(zz) + (-3.2 if zz < 50 else 1.2 if right else -1.6), f"{uu:.0f} m/s at {zz:g} m", 3.2, INK,
                  "start" if right else "end")
    return svg(w, h, s)


def cp_profile(c):
    """Median Cp on the windward, side and leeward faces, in 50 m bands, against (u/U_roof)^2."""
    v = c["mesh_vertices"].astype(float)
    f = c["mesh_faces"]
    mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel()).compute_normals(split_vertices=False)
    n = mesh.point_data["Normals"]
    cp = c["d0_pressure"] / (0.5 * 1.225 * U_ROOF**2)
    z = v[:, 2]
    bands = np.arange(0, 800, 50)
    series = {}
    for name, mask in (("windward", n[:, 0] < -0.8), ("sides", np.abs(n[:, 1]) > 0.8), ("leeward", n[:, 0] > 0.8)):
        pts = []
        for b in bands:
            sel = mask & (z >= b) & (z < b + 50)
            if sel.sum() > 5:
                pts.append((b + 25, float(np.median(cp[sel]))))
        series[name] = pts

    w, h = 80, 124
    x0, y0, ph, pw = 14, 12, 98, 60
    zmax, cmin, cmax = 900, -1.5, 1.0
    ax, zy = height_axis(x0, y0, ph, zmax, (0, 200, 400, 600, 828), label=False)
    cx = lambda p: x0 + pw * (p - cmin) / (cmax - cmin)  # noqa: E731
    # two shaded zones: Cp > 0 the wall is pushed in, Cp < 0 it is pulled out (suction)
    s = (f'<rect x="{cx(0):.2f}" y="{y0}" width="{cx(cmax) - cx(0):.2f}" height="{ph}" fill="{WIND}" fill-opacity="0.07"/>'
         f'<rect x="{x0}" y="{y0}" width="{cx(0) - x0:.2f}" height="{ph}" fill="{SIDE}" fill-opacity="0.07"/>')
    s += text(cx(0.5), y0 - 2.4, "PUSHED IN", 3.0, WIND, "middle", 600)
    s += text(cx(-0.75), y0 - 2.4, "PULLED OUT", 3.0, SIDE, "middle", 600)
    s += ax + line(x0, y0 + ph, x0 + pw, y0 + ph, INK, 0.3) + line(cx(0), y0, cx(0), y0 + ph, INK2, 0.3)
    for p in (-1.5, -1, -0.5, 0, 0.5, 1):
        s += line(cx(p), y0 + ph, cx(p), y0 + ph + 1.2, INK, 0.3)
        s += text(cx(p), y0 + ph + 4.6, f"{p:g}".replace("-", "−"), 3.2, anchor="middle")
    s += text(x0 + pw / 2, y0 + ph + 9.4, "pressure ÷ ½ρU²  (1 = full stop of the wind)", 3.0, anchor="middle")
    zz = np.linspace(1, 750, 120)
    s += path(cx((u_log(zz) / U_ROOF) ** 2), zy(zz), INK, 0.6, "1.6 1.2")
    s += text(cx(1.0) - 0.4, zy(790), "(u ÷ U)²", 3.4, INK, "end", family=MATH, italic=True)
    labels = {"windward": "front", "sides": "sides", "leeward": "back"}
    for name, col in (("windward", WIND), ("sides", SIDE), ("leeward", LEE)):
        pts = [(a, p) for a, p in series[name] if a < 750]  # above 750 m only the thin spire is left
        s += path([cx(p) for _, p in pts], [zy(a) for a, _ in pts], col, 1.0)
        for a, p in pts:
            s += f'<circle cx="{cx(p):.2f}" cy="{zy(a):.2f}" r="0.75" fill="{col}"/>'
        a, p = pts[-1]
        anchor = {"windward": "middle", "sides": "end", "leeward": "start"}[name]
        s += text(cx(p) + {"middle": 0, "end": -1.2, "start": 1.2}[anchor], zy(a) - 2.6, labels[name], 3.4, col, anchor, 600)
    return svg(w, h, s), series


def force_curve(c):
    """F = k v^2 through our four storm points (worst direction)."""
    loads = [json.loads(c[f"d{i}_metrics"])["wind_load"] for i in range(9)]
    worst = max(range(9), key=lambda i: loads[i][-1]["force_n"])
    pts = [(r["v_m_s"] * 3.6, r["force_n"] / 1e6) for r in loads[worst]]
    k = pts[-1][1] / pts[-1][0] ** 2
    w, h = 160, 74
    x0, y0, ph, pw = 14, 6, 52, 138
    vmax, fmax = 270, 90
    vx = lambda v: x0 + pw * v / vmax  # noqa: E731
    fy = lambda f: y0 + ph - ph * f / fmax  # noqa: E731
    s = line(x0, y0, x0, y0 + ph, INK, 0.3) + line(x0, y0 + ph, x0 + pw, y0 + ph, INK, 0.3)
    for f in (0, 40, 80):
        s += line(x0 - 1.2, fy(f), x0, fy(f), INK, 0.3) + text(x0 - 2, fy(f) + 1.2, f"{f}", 3.2, anchor="end")
    for v in (0, 100, 200, 250):
        s += line(vx(v), y0 + ph, vx(v), y0 + ph + 1.2, INK, 0.3) + text(vx(v), y0 + ph + 4.6, f"{v}", 3.2, anchor="middle")
    s += text(x0 - 2, y0 - 1.6, "push (MN)", 3.2, anchor="start")
    s += text(x0 + pw, y0 + ph + 9.4, "wind speed (km/h)", 3.2, anchor="end")
    vv = np.linspace(0, vmax, 120)
    s += path(vx(vv), fy(k * vv**2), INK, 0.9)
    names = ("breeze", "category 1", "category 3", "category 5")
    for (v, f), nm in zip(pts, names):
        s += f'<circle cx="{vx(v):.2f}" cy="{fy(f):.2f}" r="1.1" fill="{ACCENT}"/>'
        lab = f"{nm}: {f:.1f} MN" if f > 1 else f"{nm}: {f*1000:.0f} kN"
        if f > 1:
            s += text(vx(v) - 2, fy(f) - 2.2, lab, 3.2, INK, "end")
        else:
            s += text(vx(v) + 1, fy(f) - 4.6, lab, 3.2, INK, "start")
    return svg(w, h, s)


def reynolds():
    """Re = V D / nu for the four exhibits, on a log scale (from each cache's metrics)."""
    rows = [("Human", 1.17e6), ("F1 car", 1.44e7), ("Airbus A380", 4.94e8), ("Burj Khalifa", 2.76e9)]
    w, h = 118, 35
    x0, pw = 30, 82
    lo, hi = 5, 10
    lx = lambda r: x0 + pw * (np.log10(r) - lo) / (hi - lo)  # noqa: E731
    s = ""
    for i, (name, r) in enumerate(rows):
        y = 2 + i * 6.8
        col = ACCENT if name == "Burj Khalifa" else "#BDBDBD"
        s += text(x0 - 2, y + 3.6, name, 3.3, INK, "end", 600 if name == "Burj Khalifa" else 400)
        s += f'<rect x="{x0}" y="{y}" width="{lx(r) - x0:.2f}" height="4.6" fill="{col}"/>'
        mant, ex = f"{r:.1e}".split("e")
        s += text(lx(r) + 1.6, y + 3.6, f"{mant} × 10<tspan dy='-1.4' font-size='2.4'>{int(ex)}</tspan>", 3.2, INK)
    for e in range(lo, hi + 1):
        s += line(lx(10**e), 29, lx(10**e), 30.2, INK, 0.3)
        s += text(lx(10**e), 34, f"10<tspan dy='-1.4' font-size='2.2'>{e}</tspan>", 3, anchor="middle")
    s += line(x0, 29, x0 + pw, 29, INK, 0.3)
    return svg(w, h, s)


def main():
    c = read_cache(sys.argv[1])
    OUT.mkdir(exist_ok=True)
    (OUT / "burj_wind_profile.svg").write_text(wind_profile())
    cp_svg, series = cp_profile(c)
    (OUT / "burj_cp_profile.svg").write_text(cp_svg)
    (OUT / "burj_force_curve.svg").write_text(force_curve(c))
    (OUT / "burj_reynolds.svg").write_text(reynolds())
    print("windward", series["windward"])
    print("u at 10, 100, 828 m:", [round(float(u_log(z)), 1) for z in (10, 100, 828)])


if __name__ == "__main__":
    main()
