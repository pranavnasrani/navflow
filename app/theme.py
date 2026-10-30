"""
Colour themes for the exhibition screens: light (the default, for a bright hall) and dark.

One `Theme` carries every colour that changes: the 3D background, the top bar and sidebar, text, the three
Navier-Stokes term colours and the grey of the model. The term colours are the Okabe-Ito set, picked to
stay apart for colour-blind visitors; the dark set uses the lighter members of the same family so they
read on a dark ground. The same three colours paint the sidebar bars and the streamlines of view 2, so
they are switched together.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    bg3d: str          # graphics window background
    topbar: str
    panel: str         # sidebar
    text: str
    muted: str
    rule: str          # hairlines
    track: str         # empty part of a bar
    tab_hover: str
    tab_on_bg: str     # the active tab's pill
    tab_on_fg: str
    term_colors: dict  # convective / pressure / viscous
    term_tints: dict   # row highlight of the largest term
    model_grey: str    # the model in views 2 and 3
    rod: str           # the smoke wand's rod
    arrow: str         # the wind arrow
    lines_plain: str   # pathlines when no speed is stored


LIGHT = Theme(
    name="light", bg3d="white", topbar="#FFFFFF", panel="#ECEEF1", text="#1D232B", muted="#5B6573",
    rule="#C7CDD5", track="#D3D9E0", tab_hover="#EEF1F4", tab_on_bg="#1D232B", tab_on_fg="#FFFFFF",
    term_colors={"convective": "#0072B2", "pressure": "#D55E00", "viscous": "#B4428E"},
    term_tints={"convective": "rgba(0, 114, 178, 0.11)", "pressure": "rgba(213, 94, 0, 0.11)",
                "viscous": "rgba(180, 66, 142, 0.11)"},
    model_grey="#B9C0C9", rod="#A9B1BB", arrow="#5B6573", lines_plain="#2E3744",
)

DARK = Theme(
    name="dark", bg3d="#0E1319", topbar="#10151B", panel="#161C24", text="#E8EDF3", muted="#97A3B2",
    rule="#2B3440", track="#2A333E", tab_hover="#1E2630", tab_on_bg="#E8EDF3", tab_on_fg="#10151B",
    term_colors={"convective": "#56B4E9", "pressure": "#E69F00", "viscous": "#CC79A7"},
    term_tints={"convective": "rgba(86, 180, 233, 0.16)", "pressure": "rgba(230, 159, 0, 0.16)",
                "viscous": "rgba(204, 121, 167, 0.16)"},
    model_grey="#7D8896", rod="#5E6977", arrow="#97A3B2", lines_plain="#C5CCD6",
)

THEMES = {"light": LIGHT, "dark": DARK}
