"""
The "show" that runs when nobody is touching the exhibit: what the screens do and what they say.

The show is a loop of six steps. Each step names where the controls are (wand height, view) and where the
wind-direction knob is heading, and every object has one plain-English caption per step. All three screens
follow the same steps together, because they share one controller.

Maths link (IGCSE 0606): between two steps the knob moves along a straight line,
    knob(t) = k_from + (k_to - k_from) * s(t),     t = elapsed / seconds in [0, 1],
and s(t) = 3t^2 - 2t^3 is the "smoothstep" curve (a cubic with s(0) = 0, s(1) = 1 and zero slope at both
ends), so the knob eases in and out instead of jerking. The captions' numbers come from the solved results:
drag at 16 degrees against drag at 2 degrees is 1,238,672 / 251,895 = 4.9, "about 5 times"; the tower's drag
runs from 34.2 MN to 39.9 MN, 39.9 / 34.2 = 1.165, "up to about 16 %"; and F = m g gives 35 MN / 9.81 = 3.6
million kg, "roughly 3,500 tonnes".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShowStep:
    wand: int            # smoke wand index 0..9
    view: int            # 0 pressure + speed, 1 dominant Navier-Stokes term, 2 turbulence k
    knob_to: float | None  # where the knob is heading during this step; None = it stays where it is
    seconds: float


SHOW_STEPS: tuple[ShowStep, ...] = (
    ShowStep(wand=4, view=0, knob_to=0.0, seconds=9.0),    # 0 introduce the object
    ShowStep(wand=4, view=0, knob_to=0.0, seconds=9.0),    # 1 point out high and low pressure
    ShowStep(wand=5, view=0, knob_to=0.5, seconds=11.0),   # 2 turn the knob halfway
    ShowStep(wand=6, view=0, knob_to=1.0, seconds=11.0),   # 3 and all the way
    ShowStep(wand=4, view=1, knob_to=0.4, seconds=10.0),   # 4 the same air coloured by Navier-Stokes term
    ShowStep(wand=4, view=2, knob_to=0.4, seconds=9.0),    # 5 and by turbulence
)

_NAVIER_STOKES = "New colours: each line shows which part of the Navier–Stokes equation is biggest there: inertia, pressure or friction."
_TURBULENCE = "Now the lines show turbulence energy k: red is swirling, energetic air, blue is calm air."

CAPTIONS: dict[str, tuple[str, ...]] = {
    "airplane": (
        "A full-size airliner at 102 m/s (367 km/h). Each line is air, coloured by its speed.",
        "Red: the air is stopped and pushes on the surface. Blue: fast air pulls on the surface.",
        "Tilt the nose up (angle of attack): the wing pushes air down, so the air pushes the plane up. That is lift.",
        "Past about 6° the wake thickens and drag climbs: at 16° it is about 5 times the drag at 2°.",
        _NAVIER_STOKES,
        _TURBULENCE,
    ),
    "f1_car": (
        "A full-size Formula 1 car at 144 km/h (40 m/s). The colours on the car show air pressure.",
        "Red: the air is stopped at the nose and pushes in. Blue: the air is pulled off the surface.",
        "The air pushes the car DOWN onto the track. That is downforce, and it helps the tyres grip.",
        "Turn the wind sideways (crosswind): at 16° the car burns about 17 % more fuel than at 6°.",
        _NAVIER_STOKES,
        _TURBULENCE,
    ),
    "building": (
        "The Burj Khalifa (828 m) in a 50 m/s wind (180 km/h). Smoke lines show the air at several heights.",
        "Red: the wind hits the front face and pushes in. Blue: the sides are pulled outward.",
        "The wind pushes the whole tower with about 35 million newtons: roughly the weight of 3,500 tonnes.",
        "Turn the wind direction: the push changes by up to about 16 %, so engineers test every angle.",
        _NAVIER_STOKES,
        _TURBULENCE,
    ),
    "human": (
        "A person-shaped block in a 10 m/s wind (36 km/h). Each line is air, coloured by its speed.",
        "Red: the air is stopped and pushes on the surface. Blue: fast air pulls on the surface.",
        "A blunt shape makes a wide, slow, swirling wake behind it. That wake is what drag is made of.",
        "Turn the body: the wake changes shape and so does the drag.",
        _NAVIER_STOKES,
        _TURBULENCE,
    ),
}

HIGH_LABEL = "Air stops here:\nhigh pressure"
LOW_LABEL = "Air speeds up here:\nit pulls on the surface"


def caption_for(obj_id: str, step: int) -> str:
    texts = CAPTIONS.get(obj_id) or CAPTIONS["airplane"]
    return texts[int(step) % len(texts)]


def smoothstep(t: float) -> float:
    """s(t) = 3t^2 - 2t^3 on [0, 1]: 0 at the start, 1 at the end, flat at both ends."""
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)
