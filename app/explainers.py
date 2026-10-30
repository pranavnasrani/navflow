"""
The "show" that runs when nobody is touching the exhibit: what the screens do and what they say.

The show is a loop of six steps. Each step names where the controls are (wand height, view) and where the
wind-direction knob is heading, and every object has one plain-English caption per step. All three screens
follow the same steps together, because they share one controller.

Maths link (IGCSE 0606): between two steps the knob moves along a straight line,
    knob(t) = k_from + (k_to - k_from) * s(t),     t = elapsed / seconds in [0, 1],
and s(t) = 3t^2 - 2t^3 is the "smoothstep" curve (a cubic with s(0) = 0, s(1) = 1 and zero slope at both
ends), so the knob eases in and out instead of jerking. The captions' numbers come from the solved results:
drag at 16 degrees against drag at 2 degrees is 1,238,672 / 251,895 = 4.9, "about 5 times". The push on the
tower is the TOTAL sideways force, F = sqrt(F_drag^2 + F_side^2) (Pythagoras on the two horizontal components):
it runs from 38.2 MN (60 degrees) to 40.4 MN (7.5 degrees), 40.4 / 38.2 = 1.057, "about 6 %"; and F = m g
gives 39 MN / 9.81 = 4.0 million kg, "roughly 4,000 tonnes".
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
        "The wind pushes the whole tower sideways with about 39 million newtons: roughly the weight of 4,000 tonnes.",
        "Turn the wind direction: the push changes by about 6 %, and it is strongest at 7.5°.",
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


# --------------------------------------------------------------------------------------------------
# The challenge on each screen: one question, one number, one graph
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Mission:
    question: str
    label: str      # what the big number is
    unit: str       # shown after the number ("" for a ratio)
    decimals: int
    goal: str       # "max" or "min": which end of the graph is the answer
    better: str     # one line saying which way is better
    found: str      # said once the visitor reaches the answer
    kind: str       # which solved quantity: see mission_value


MISSIONS: dict[str, Mission] = {
    "airplane": Mission(
        question="Find the angle where the plane flies most efficiently",
        label="Lift ÷ drag", unit="", decimals=1, goal="max",
        better="Higher is better: more lift for every newton of drag.",
        found="You found it! At 6° the wing gives the most lift for its drag, so the plane burns the least fuel.",
        kind="lift_to_drag",
    ),
    "f1_car": Mission(
        question="Which crosswind makes the car burn the least fuel?",
        label="Fuel burned", unit="L/h", decimals=1, goal="min",
        better="Lower is better: less drag means less fuel and less CO₂.",
        found="You found it! At 6° of crosswind the car has the least drag, so it burns the least fuel.",
        kind="fuel_burn_l_per_hr",
    ),
    "building": Mission(
        question="Which wind direction pushes hardest on the tower?",
        label="Sideways push", unit="MN", decimals=1, goal="max",
        better="Engineers look for the WORST case, so the tower is built strong enough for it.",
        found="You found it! At 7.5° the wind pushes hardest: the tower is designed for this worst case.",
        kind="total_push_mn",
    ),
}


def mission_value(kind: str, metrics: dict) -> float:
    """The quantity a challenge is about, from one solved direction's metrics.

    total_push_mn: the two horizontal force components are at right angles, so by Pythagoras the total
    sideways push is sqrt(F_drag^2 + F_side^2), in meganewtons (the same force the wind-load table uses).
    """
    def num(key: str) -> float:
        v = metrics.get(key, 0.0)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0.0

    if kind == "total_push_mn":
        return float((num("drag_force_n") ** 2 + num("side_force_n") ** 2) ** 0.5) / 1e6
    return num(kind)
