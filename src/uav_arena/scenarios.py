"""Seeded arena scenarios: start, goal, payload, wind and obstacles under eight conditions.

A condition crosses two obstacle settings (static pillars; the same pillars plus
moving spheres) with four observation latencies. For one seed the layout, the
payload mass and the wind are shared by every condition, so obstacle and latency
effects are paired. The payload mass and the wind are hidden from every method
except the privileged oracle. Moving obstacles and the wind follow pure functions
of time, so any episode replays exactly.

`make_scenario(seed, condition, settings=TaskSettings(...))` draws from a tuned
scenario family (`settings.py`); the default settings give the arena-v0.3 scenarios.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace
import math

import numpy as np

from .settings import PILLAR_AREA, PILLAR_GAP, TaskSettings
from .wind import Wind


OBSTACLE_SETTINGS = ("static", "static+moving")
LATENCIES_MS = (0, 50, 100, 200)
WORKSPACE = ((-3.0, 3.0), (-2.5, 2.5), (0.0, 3.0))   # x, y, z bounds in metres
PILLAR_HEIGHT = 3.0
SUITE_SEEDS = dict(dev=tuple(range(20)), test=tuple(range(1000, 1050)))
PAYLOAD_MASS_RANGE = (0.010, 0.040)    # kg, drawn per seed; onboard feedforward assumes 25 g
NOMINAL_PAYLOAD_MASS = 0.025           # kg, cable-load feedforward shared by every method
HIDEABLE = ("payload_mass", "wind")    # scenario fields that `public()` may remove


@dataclass(frozen=True)
class Condition:
    obstacles: str = "static"
    latency_ms: int = 0

    def __post_init__(self):
        if self.obstacles not in OBSTACLE_SETTINGS or self.latency_ms not in LATENCIES_MS:
            raise ValueError(f"Unknown condition {self}")

    @property
    def name(self):
        return f"{self.obstacles}/{self.latency_ms}ms"


CONDITIONS = tuple(Condition(o, d) for o in OBSTACLE_SETTINGS for d in LATENCIES_MS)


@dataclass(frozen=True)
class Pillar:
    """Static vertical cylinder from the floor to PILLAR_HEIGHT."""
    x: float
    y: float
    radius: float


@dataclass(frozen=True)
class Mover:
    """Sphere moving back and forth on the segment a-b at constant speed."""
    a: tuple
    b: tuple
    speed: float
    phase: float
    radius: float

    def state(self, t):
        a, b = np.asarray(self.a, dtype=float), np.asarray(self.b, dtype=float)
        length = float(np.linalg.norm(b-a))
        period = 2*length/self.speed
        s = ((t/period+self.phase) % 1.0)*2.0          # 0..2 along a->b->a
        forward = s < 1.0
        fraction = s if forward else 2.0-s
        direction = (b-a)/length
        return a+fraction*(b-a), (1.0 if forward else -1.0)*self.speed*direction


@dataclass(frozen=True)
class Scenario:
    seed: int
    condition: Condition
    start: tuple                     # payload position at the hover start
    goal: tuple                      # payload goal position
    pillars: tuple
    movers: tuple = ()
    payload_mass: float = 0.03
    time_limit: float = 20.0
    goal_tolerance: float = 0.15
    meta: dict = field(default_factory=dict, compare=False, hash=False)
    # Release fields. Their defaults are the arena-v0.3 task, and `to_dict` leaves them out while they
    # hold them, so arena-v0.3 scenario dictionaries and dataset hashes stay unchanged.
    nominal_mass: float = NOMINAL_PAYLOAD_MASS      # public: onboard feedforward and the physics row's mass
    payload_prior: tuple = PAYLOAD_MASS_RANGE        # public: the range the payload mass is drawn from
    wind: Wind | None = None                         # the true wind; None is still air
    hidden: tuple = HIDEABLE                         # fields `public()` removes

    def __post_init__(self):
        object.__setattr__(self, "payload_prior", tuple(float(x) for x in self.payload_prior))
        object.__setattr__(self, "hidden", tuple(self.hidden))
        if not set(self.hidden) <= set(HIDEABLE):
            raise ValueError(f"Only {HIDEABLE} can be hidden, got {self.hidden}")
        if self.wind is not None and not isinstance(self.wind, Wind):
            raise TypeError("Scenario.wind must be a uav_arena.wind.Wind or None")

    def mover_states(self, t):
        if not self.movers:
            return np.zeros((0, 3)), np.zeros((0, 3))
        states = [m.state(t) for m in self.movers]
        return np.stack([s[0] for s in states]), np.stack([s[1] for s in states])

    def public(self):
        """The scenario without its hidden fields (by default the payload mass and the wind): what
        non-privileged methods may read."""
        return replace(self, **{name: None for name in self.hidden})

    def to_dict(self):
        values = asdict(self)
        values["condition"] = dict(obstacles=self.condition.obstacles, latency_ms=self.condition.latency_ms)
        for name, default in _RELEASE_DEFAULTS.items():
            if getattr(self, name) == default:
                del values[name]
        if self.wind is not None:
            values["wind"] = self.wind.to_dict()
        if "payload_prior" in values:
            values["payload_prior"] = list(self.payload_prior)
        if "hidden" in values:
            values["hidden"] = list(self.hidden)
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values["condition"] = Condition(**values["condition"])
        values["start"], values["goal"] = tuple(values["start"]), tuple(values["goal"])
        values["pillars"] = tuple(Pillar(**p) for p in values["pillars"])
        values["movers"] = tuple(Mover(**dict(m, a=tuple(m["a"]), b=tuple(m["b"]))) for m in values["movers"])
        if values.get("wind") is not None:
            values["wind"] = Wind.from_dict(values["wind"])
        return cls(**values)


_RELEASE_DEFAULTS = dict(nominal_mass=NOMINAL_PAYLOAD_MASS, payload_prior=PAYLOAD_MASS_RANGE, wind=None, hidden=HIDEABLE)


def _free_path(pillars, start, goal, clearance, resolution=0.05):
    """Grid search in the plane: can a disc of radius `clearance` pass the pillars?"""
    (x0, x1), (y0, y1), _ = WORKSPACE
    xs, ys = np.arange(x0, x1+1e-9, resolution), np.arange(y0, y1+1e-9, resolution)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    free = (X-x0 >= clearance) & (x1-X >= clearance) & (Y-y0 >= clearance) & (y1-Y >= clearance)
    for p in pillars:
        free &= np.hypot(X-p.x, Y-p.y) >= p.radius+clearance
    index = lambda point: (int(round((point[0]-x0)/resolution)), int(round((point[1]-y0)/resolution)))
    begin, end = index(start), index(goal)
    if not (free[begin] and free[end]):
        return False
    seen, queue = {begin}, deque([begin])
    while queue:
        i, j = queue.popleft()
        if (i, j) == end:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i+di, j+dj)
            if 0 <= n[0] < free.shape[0] and 0 <= n[1] < free.shape[1] and free[n] and n not in seen:
                seen.add(n); queue.append(n)
    return False


LAYOUT_ATTEMPTS = 1000     # pillar layouts tried until one leaves the team a corridor
PILLAR_DRAWS = 100_000     # pillar candidates per layout before the layout is abandoned
PACKING_FAILURES = 3       # consecutive abandoned layouts that end the search with an error


def team_clearance(formation_radius=0.25, uav_radius=0.06, margin=0.10):
    """Planar radius the whole team needs to pass between pillars."""
    return formation_radius+uav_radius+margin


def make_scenario(seed, condition, *, settings=None, n_pillars=None, n_movers=None, mover_radius=None,
                  pillar_radius=None, mover_speed=None, payload_mass=None, wind=None):
    """Layout, payload mass and wind depend only on the seed; movers are added only in the moving setting.

    `settings` (a `TaskSettings`) picks the scenario family, arena-v0.3's by default. Keyword
    arguments override it: obstacle sizes and counts, `payload_mass` (kg) to fix the true mass,
    and `wind` (a `Wind`) to fix the true wind.
    """
    settings = TaskSettings() if settings is None else settings
    overrides = dict(n_pillars=n_pillars, n_movers=n_movers, mover_radius=mover_radius, pillar_radius=pillar_radius,
                     mover_speed=mover_speed)
    obstacles = replace(settings.obstacles, **{k: v for k, v in overrides.items() if v is not None})  # validated
    n_pillars, n_movers, mover_radius = obstacles.n_pillars, obstacles.n_movers, obstacles.mover_radius
    pillar_radius, mover_speed = obstacles.pillar_radius, obstacles.mover_speed
    if payload_mass is None:
        payload_mass = settings.payload.draw(seed)
    if wind is None:
        wind = settings.wind.draw(seed, settings.time_limit)
    layout = np.random.default_rng([int(seed), 7])
    start = (-1.6, float(layout.uniform(-0.4, 0.4)), float(layout.uniform(1.0, 1.4)))
    goal = (1.6, float(layout.uniform(-0.4, 0.4)), float(layout.uniform(1.0, 1.4)))
    clearance = team_clearance()
    (x0, x1), (y0, y1) = PILLAR_AREA
    abandoned = 0
    for attempt in range(LAYOUT_ATTEMPTS):
        # Rejection sampling, as in arena-v0.3. With up to 16 default-size pillars no layout needs more
        # than about 25,000 draws (every standard seed needs at most 18 for its 4 pillars), so the budget
        # leaves those layouts exactly as arena-v0.3 draws them. It binds only for denser requests, which
        # arena-v0.3's sampler could take millions of draws to finish or never finish at all.
        pillars, draws = [], 0
        while len(pillars) < n_pillars and draws < PILLAR_DRAWS:
            draws += 1
            p = Pillar(float(layout.uniform(x0, x1)), float(layout.uniform(y0, y1)), float(layout.uniform(*pillar_radius)))
            if all(math.hypot(p.x-q.x, p.y-q.y) > p.radius+q.radius+PILLAR_GAP for q in pillars):
                pillars.append(p)
        if len(pillars) < n_pillars:
            abandoned += 1
            if abandoned == PACKING_FAILURES:
                raise ValueError(
                    f"Seed {seed}: could not place {n_pillars} pillars of radius {pillar_radius[0]:g} to {pillar_radius[1]:g} m, "
                    f"{PILLAR_GAP:g} m apart, in the pillar area within {PILLAR_DRAWS} draws, in {PACKING_FAILURES} "
                    "layouts in a row. Use fewer or thinner pillars.")
            continue
        abandoned = 0
        if _free_path(pillars, start, goal, clearance):
            break
    else:
        raise RuntimeError(f"No pillar layout for seed {seed} leaves the team a corridor from start to goal in "
                           f"{LAYOUT_ATTEMPTS} attempts ({n_pillars} pillar{'s' if n_pillars != 1 else ''} of radius "
                           f"{pillar_radius[0]:g} to {pillar_radius[1]:g} m). Use fewer or thinner pillars.")
    movers = ()
    if condition.obstacles == "static+moving":
        motion = np.random.default_rng([int(seed), 11])
        movers = tuple(Mover(a=(float(motion.uniform(-0.7, 0.7)), -1.3, float(motion.uniform(start[2]-0.1, start[2]+0.6))),
                             b=(float(motion.uniform(-0.7, 0.7)), 1.3, float(motion.uniform(start[2]-0.1, start[2]+0.6))),
                             speed=float(motion.uniform(*mover_speed)), phase=float(motion.uniform()),
                             radius=float(mover_radius)) for _ in range(n_movers))
    hidden = tuple(name for name, hide in (("payload_mass", settings.payload.hidden), ("wind", settings.wind.hidden))
                   if hide)
    return Scenario(seed=int(seed), condition=condition, start=start, goal=goal, pillars=tuple(pillars),
                    movers=movers, payload_mass=float(payload_mass), time_limit=float(settings.time_limit),
                    goal_tolerance=float(settings.goal_tolerance), meta=dict(layout_attempts=attempt+1),
                    nominal_mass=float(settings.payload.nominal), payload_prior=settings.payload.prior, wind=wind,
                    hidden=hidden)


def suite(name, conditions=CONDITIONS, **options):
    """Every condition on every seed of a named suite; seeds are frozen with the code."""
    return [make_scenario(seed, condition, **options) for condition in conditions for seed in SUITE_SEEDS[name]]
