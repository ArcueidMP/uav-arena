"""Task settings: the scenario family to draw from.

`TaskSettings()` is the arena-v0.3 task, and every default below is that task's value. With the
defaults, `make_scenario` and every environment reproduce arena-v0.3 bit for bit
(`tests/test_parity.py`). Results under any other settings form a new scenario family, so they
are not comparable with arena-v0.3 tables.

    from uav_arena import TaskSettings, PayloadSettings, WindSettings, GustSettings, make_scenario

    settings = TaskSettings(payload=PayloadSettings(mass_range=(0.02, 0.06)),
                            wind=WindSettings(speed=(0.0, 2.0), gusts=GustSettings(rate=0.2)))
    scenario = make_scenario(seed=3, condition=CONDITIONS[0], settings=settings)

A layout seed fixes every draw, including the payload mass and the wind. All conditions of one seed
share them, so the effects of obstacles and latency stay paired. Settings round-trip through JSON
(`to_dict`, `from_dict`, `load_settings`), which is how `run.py`, `collect.py` and `replay.py`
take them. See docs/settings.md for the physical meaning and the measured limits of each knob.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from .wind import Gust, Turbulence, Wind


def _range(value, name, *, low=0.0, allow_equal=True):
    """A (low, high) pair from a number or a pair, checked to be finite, ordered and >= `low`."""
    pair = (float(value), float(value)) if np.isscalar(value) else tuple(float(x) for x in value)
    if len(pair) != 2 or not all(math.isfinite(x) for x in pair) or pair[0] < low or pair[1] < pair[0] \
            or (not allow_equal and pair[0] == pair[1]):
        raise ValueError(f"{name} must be a number or an ordered (low, high) pair, each >= {low}")
    return pair


def _draw(rng, pair):
    return float(rng.uniform(*pair))


# ---------- payload ----------
def hover_rotor_fraction(payload_mass):
    """The largest single-rotor thrust needed to hover the formation with this payload, as a fraction of
    the rotor limit: the static balance `initial_hover` uses (official Crazyflie 2, 0.6 m cables)."""
    from .sim.menagerie import MenagerieTransportConfig
    from .sim.plant import formation_offsets
    config = MenagerieTransportConfig()
    directions = formation_offsets(config)/config.cable_length
    tension = float(payload_mass)*config.gravity/(3*directions[0, 2])
    forces = tension*directions
    forces[:, 2] += config.drone_mass*config.gravity
    return float(np.linalg.norm(forces, axis=1).max()/4/config.rotor_max)


def max_payload_mass(hover_fraction=0.75):
    """The heaviest payload (kg) the team hovers with at most `hover_fraction` of rotor thrust."""
    if not 0 < hover_fraction <= 1:
        raise ValueError("hover_fraction must lie in (0, 1]")
    if hover_rotor_fraction(0.0) > hover_fraction:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = 0.5*(low+high)
        low, high = (middle, high) if hover_rotor_fraction(middle) <= hover_fraction else (low, middle)
    return low


@dataclass(frozen=True)
class PayloadSettings:
    """The payload mass of a scenario family.

    mass_range: each layout seed draws its payload mass uniformly from this range (kg).
    mass: one fixed payload mass for every scenario instead (kg); `mass_range` is then unused.
    nominal_mass: the publicly known mass (kg), which the onboard cable-load feedforward and the
        physics reference row assume. None: the middle of the public prior (25 g by default).
    hidden: hide the true mass from non-privileged methods (`Scenario.public()`).
    max_hover_fraction: refuse a family whose heaviest payload needs more than this fraction of rotor
        thrust to hover, so the onboard controller keeps headroom. 0.75 admits payloads up to about
        63 g. In tests, 80 g still hovered and 100 g crashed in the warm-up.
    public_prior: the (low, high) range that methods are told the mass comes from (kg). None: the
        masses the family can draw. A fixed `mass` with no wider public prior is effectively public,
        because the prior then pins it; set e.g. public_prior=(0.01, 0.06) to hide it.
    """
    mass_range: tuple = (0.010, 0.040)
    mass: float | None = None
    nominal_mass: float | None = None
    hidden: bool = True
    max_hover_fraction: float = 0.75
    public_prior: tuple | None = None

    def __post_init__(self):
        object.__setattr__(self, "mass_range", _range(self.mass_range, "PayloadSettings.mass_range", low=1e-6))
        if self.mass is not None:
            object.__setattr__(self, "mass", float(self.mass))
            if not (math.isfinite(self.mass) and self.mass > 0):
                raise ValueError("PayloadSettings.mass must be positive")
        if self.nominal_mass is not None:
            object.__setattr__(self, "nominal_mass", float(self.nominal_mass))
            if not (math.isfinite(self.nominal_mass) and self.nominal_mass > 0):
                raise ValueError("PayloadSettings.nominal_mass must be positive")
        if not 0 < self.max_hover_fraction <= 1:
            raise ValueError("PayloadSettings.max_hover_fraction must lie in (0, 1]")
        if self.public_prior is not None:
            object.__setattr__(self, "public_prior", _range(self.public_prior, "PayloadSettings.public_prior", low=1e-6))
            low, high = (self.mass, self.mass) if self.mass is not None else self.mass_range
            if not self.public_prior[0] <= low <= high <= self.public_prior[1]:
                raise ValueError("PayloadSettings.public_prior must contain every mass the family can draw")
        heaviest = max(self.prior[1], self.nominal)
        if hover_rotor_fraction(heaviest) > self.max_hover_fraction:
            raise ValueError(
                f"A {1000*heaviest:.1f} g payload needs {100*hover_rotor_fraction(heaviest):.0f}% of rotor thrust to "
                f"hover; the limit is {100*self.max_hover_fraction:.0f}% ({1000*max_payload_mass(self.max_hover_fraction):.1f} g). "
                "Raise max_hover_fraction to accept less control headroom.")

    @property
    def prior(self):
        """The public (low, high) range of the mass: `public_prior`, else the masses the family can draw."""
        if self.public_prior is not None:
            return self.public_prior
        return (self.mass, self.mass) if self.mass is not None else self.mass_range

    @property
    def nominal(self):
        low, high = self.prior
        return self.nominal_mass if self.nominal_mass is not None else (low+high)/2

    def draw(self, seed):
        if self.mass is not None:
            return self.mass
        return float(np.random.default_rng([int(seed), 13]).uniform(*self.mass_range))


# ---------- wind ----------
@dataclass(frozen=True)
class GustSettings:
    """Discrete "1-cosine" gusts with Poisson onsets over the episode.

    rate: expected gusts per second. peak: range of peak gust speeds (m/s). duration: range of gust
    durations (s). heading: range of horizontal gust directions (degrees counterclockwise from +x,
    the direction the air moves toward); None for any direction. vertical: the vertical peak
    component is uniform within +-vertical times the peak speed.
    """
    rate: float = 0.2
    peak: tuple = (1.0, 3.0)
    duration: tuple = (0.5, 2.0)
    heading: tuple | None = None
    vertical: float = 0.0

    def __post_init__(self):
        if not (math.isfinite(self.rate) and self.rate > 0):
            raise ValueError("GustSettings.rate must be positive (gusts per second)")
        object.__setattr__(self, "peak", _range(self.peak, "GustSettings.peak"))
        object.__setattr__(self, "duration", _range(self.duration, "GustSettings.duration", low=1e-3))
        if self.heading is not None:
            object.__setattr__(self, "heading", _range(self.heading, "GustSettings.heading", low=-math.inf))
        if not (math.isfinite(self.vertical) and self.vertical >= 0):
            raise ValueError("GustSettings.vertical must be >= 0")

    def draw(self, rng, time_limit):
        gusts, t = [], 0.0
        while True:
            t += float(rng.exponential(1.0/self.rate))
            if t >= time_limit:
                return tuple(gusts)
            duration, speed = _draw(rng, self.duration), _draw(rng, self.peak)
            heading = math.radians(_draw(rng, self.heading) if self.heading is not None else float(rng.uniform(0, 360)))
            vertical = self.vertical*speed*float(rng.uniform(-1.0, 1.0))
            gusts.append(Gust(onset=t, duration=duration,
                              peak=(speed*math.cos(heading), speed*math.sin(heading), vertical)))


@dataclass(frozen=True)
class TurbulenceSettings:
    """Continuous zero-mean turbulence with a first-order low-pass (Ornstein-Uhlenbeck) spectrum.

    intensity: standard deviation per world axis (m/s). time_scale: correlation time (s); the spectrum's
    corner frequency is 1/(2 pi time_scale). components: sinusoids per axis, at log-spaced random
    frequencies spanning the corner frequency /30 to x30, with powers that follow the spectrum.
    """
    intensity: tuple = (0.4, 0.4, 0.2)
    time_scale: float = 1.0
    components: int = 24

    def __post_init__(self):
        intensity = tuple(float(x) for x in self.intensity)
        if len(intensity) != 3 or not all(math.isfinite(x) and x >= 0 for x in intensity):
            raise ValueError("TurbulenceSettings.intensity must be three standard deviations >= 0 (m/s)")
        object.__setattr__(self, "intensity", intensity)
        if not (math.isfinite(self.time_scale) and self.time_scale > 0) or int(self.components) < 1:
            raise ValueError("TurbulenceSettings needs time_scale > 0 and components >= 1")
        object.__setattr__(self, "components", int(self.components))

    def draw(self, rng):
        corner = 1.0/(2*math.pi*self.time_scale)
        edges = np.geomspace(corner/30, corner*30, self.components+1)
        share = np.diff(np.arctan(edges/corner))
        share = share/share.sum()                   # each band's part of the variance
        amplitude, frequency, phase = [], [], []
        for sigma in self.intensity:
            frequency.append(edges[:-1]*(edges[1:]/edges[:-1])**rng.uniform(size=self.components))
            amplitude.append(sigma*np.sqrt(2*share))
            phase.append(rng.uniform(0, 2*math.pi, self.components))
        return Turbulence(amplitude=tuple(map(tuple, amplitude)), frequency=tuple(map(tuple, frequency)),
                          phase=tuple(map(tuple, phase)))


@dataclass(frozen=True)
class WindSettings:
    """The wind of a scenario family: a horizontal mean wind plus optional gusts and turbulence.

    speed: mean wind speed (m/s), a number or a (low, high) range drawn per layout seed.
    heading: mean wind direction (degrees counterclockwise from +x, the direction the air moves
        toward: 0 is a tailwind for the start-to-goal transfer, 180 a headwind). A number, a range,
        or None for any direction.
    hidden: hide the wind from non-privileged methods (`Scenario.public()`).
    """
    speed: float | tuple = 0.0
    heading: float | tuple | None = None
    gusts: GustSettings | None = None
    turbulence: TurbulenceSettings | None = None
    hidden: bool = True

    def __post_init__(self):
        object.__setattr__(self, "speed", _range(self.speed, "WindSettings.speed"))
        if self.heading is not None:
            object.__setattr__(self, "heading", _range(self.heading, "WindSettings.heading", low=-math.inf))

    @property
    def still(self):
        return self.speed == (0.0, 0.0) and self.gusts is None and self.turbulence is None

    def draw(self, seed, time_limit):
        """The wind of one layout seed; None for still air.

        Each part has its own random stream, [seed, 19, k] for k = 1..4, disjoint from every other
        stream in the arena: the scenario's (13, 7, 11) and the data collector's behavior policies
        ([seed, 17], [seed, 17, 2]; NumPy pads short seeds with zeros, so k starts at 1)."""
        if self.still:
            return None
        streams = [np.random.default_rng([int(seed), 19, k]) for k in range(1, 5)]
        speed = _draw(streams[0], self.speed)
        heading = math.radians(_draw(streams[1], self.heading) if self.heading is not None
                               else float(streams[1].uniform(0, 360)))
        return Wind(mean=(speed*math.cos(heading), speed*math.sin(heading), 0.0),
                    gusts=self.gusts.draw(streams[2], time_limit) if self.gusts is not None else (),
                    turbulence=self.turbulence.draw(streams[3]) if self.turbulence is not None else None)


# ---------- obstacles and the whole task ----------
PILLAR_AREA = ((-0.8, 0.8), (-1.0, 1.0))   # x and y ranges of pillar centres (m), as in arena-v0.3
PILLAR_GAP = 0.15                           # minimum free gap between two pillars (m)


def max_pillars(smallest_radius):
    """An upper bound on how many pillars of at least `smallest_radius` fit the pillar area PILLAR_GAP apart.

    Their centres lie in the area at least d = 2 * smallest_radius + PILLAR_GAP apart, so none fit
    beyond one when d exceeds the area's diagonal, and otherwise at most Oler's bound for points at
    mutual distance d in a convex region of area A and perimeter P: 2 A / (sqrt(3) d^2) + P / (2 d) + 1.
    """
    (x0, x1), (y0, y1) = PILLAR_AREA
    width, height, d = x1-x0, y1-y0, 2*smallest_radius+PILLAR_GAP
    if d > math.hypot(width, height):
        return 1
    return int(2*width*height/(math.sqrt(3)*d*d)+(width+height)/d+1)


@dataclass(frozen=True)
class ObstacleSettings:
    """Obstacle layout. Pillars appear in every condition; moving spheres in the static+moving ones.

    Pillar centres are drawn in PILLAR_AREA, at least PILLAR_GAP apart, and every layout must leave
    the team a corridor from start to goal. Settings that cannot fit `n_pillars` (`max_pillars`) are
    refused here; `make_scenario` stops with an error when a layout stays out of reach of its draws.
    """
    n_pillars: int = 4
    pillar_radius: tuple = (0.10, 0.18)
    n_movers: int = 2
    mover_radius: float = 0.15
    mover_speed: tuple = (0.25, 0.45)

    def __post_init__(self):
        if int(self.n_pillars) < 0 or int(self.n_movers) < 0:
            raise ValueError("Obstacle counts must be >= 0")
        object.__setattr__(self, "n_pillars", int(self.n_pillars))
        object.__setattr__(self, "n_movers", int(self.n_movers))
        object.__setattr__(self, "pillar_radius", _range(self.pillar_radius, "ObstacleSettings.pillar_radius", low=1e-3))
        object.__setattr__(self, "mover_speed", _range(self.mover_speed, "ObstacleSettings.mover_speed", low=1e-3))
        if not (math.isfinite(self.mover_radius) and self.mover_radius > 0):
            raise ValueError("ObstacleSettings.mover_radius must be positive")
        object.__setattr__(self, "mover_radius", float(self.mover_radius))
        if self.n_pillars > max_pillars(self.pillar_radius[0]):
            (x0, x1), (y0, y1) = PILLAR_AREA
            raise ValueError(
                f"{self.n_pillars} pillars of radius {self.pillar_radius[0]:g} m or more, {PILLAR_GAP:g} m apart, cannot "
                f"fit the pillar area (x {x0:g} to {x1:g} m, y {y0:g} to {y1:g} m); at most "
                f"{max_pillars(self.pillar_radius[0])} fit. Use fewer or thinner pillars.")


_PARTS = dict(payload=PayloadSettings, wind=WindSettings, obstacles=ObstacleSettings)
_NESTED = dict(gusts=GustSettings, turbulence=TurbulenceSettings)


@dataclass(frozen=True)
class TaskSettings:
    """Everything a scenario family varies. The defaults are the arena-v0.3 task."""
    payload: PayloadSettings = field(default_factory=PayloadSettings)
    wind: WindSettings = field(default_factory=WindSettings)
    obstacles: ObstacleSettings = field(default_factory=ObstacleSettings)
    time_limit: float = 20.0          # s
    goal_tolerance: float = 0.15      # m, payload distance to the goal that counts as arrived

    def __post_init__(self):
        if not (math.isfinite(self.time_limit) and self.time_limit > 0 and math.isfinite(self.goal_tolerance)
                and self.goal_tolerance > 0):
            raise ValueError("time_limit and goal_tolerance must be positive")

    @property
    def is_default(self):
        return self == TaskSettings()

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        unknown = set(values)-{"payload", "wind", "obstacles", "time_limit", "goal_tolerance"}
        if unknown:
            raise ValueError(f"Unknown task settings: {sorted(unknown)}")
        for name, kind in _PARTS.items():
            if name in values and not isinstance(values[name], kind):
                part = dict(values[name])
                for key, nested in _NESTED.items():
                    if part.get(key) is not None and not isinstance(part[key], nested):
                        part[key] = nested(**part[key])
                values[name] = kind(**part)
        return cls(**values)


def load_settings(path):
    """TaskSettings from a JSON file; None or an empty path gives the default (arena-v0.3) task."""
    if not path:
        return TaskSettings()
    return TaskSettings.from_dict(json.loads(Path(path).read_text()))
