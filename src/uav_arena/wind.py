"""Wind fields: a constant mean wind, discrete gusts and continuous turbulence.

A `Wind` is one scenario's realization: a world-frame air velocity that is a pure function of
episode time, like the movers' paths, so snapshots, planning forks and replays need no wind state.
MuJoCo turns the air velocity into drag through its inertia-box fluid model, which is active for
every body (air density 1.225 kg/m^3 and viscosity 1.8e-5 Pa s, as in the Menagerie airframe).

The mean wind already blows during the warm-up, so an episode starts from a hover settled in it;
gusts and turbulence act from t = 0 on. `TaskSettings.wind` (`settings.py`) draws a `Wind` for
each layout seed, and `make_scenario(..., wind=Wind(...))` sets one directly.

Model limits: drag is quadratic in airspeed on each body's equivalent inertia box. Rotor drag,
roughly linear in airspeed and dominant for small quadrotors at low speed, is not modeled, and the
cables (tendons) feel no air.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np


def _vector(values, name):
    vector = tuple(float(x) for x in values)
    if len(vector) != 3 or not all(math.isfinite(x) for x in vector):
        raise ValueError(f"{name} must be three finite numbers (m/s, world frame)")
    return vector


@dataclass(frozen=True)
class Gust:
    """One discrete "1-cosine" gust: the air velocity rises smoothly to `peak` at mid-gust and falls back."""
    onset: float          # episode time when the gust starts (s)
    duration: float       # s
    peak: tuple           # world-frame peak velocity (m/s)

    def __post_init__(self):
        object.__setattr__(self, "peak", _vector(self.peak, "Gust.peak"))
        if not (math.isfinite(self.onset) and math.isfinite(self.duration) and self.duration > 0):
            raise ValueError("A gust needs a finite onset and a positive duration")

    def velocity(self, t):
        s = (t-self.onset)/self.duration
        if s <= 0.0 or s >= 1.0:
            return np.zeros(3)
        return 0.5*(1.0-math.cos(2.0*math.pi*s))*np.asarray(self.peak)


@dataclass(frozen=True)
class Turbulence:
    """Zero-mean continuous turbulence: on each world axis a sum of sinusoids whose power follows a
    first-order low-pass (Ornstein-Uhlenbeck) spectrum; see `TurbulenceSettings`."""
    amplitude: tuple      # (3, K) m/s
    frequency: tuple      # (3, K) Hz
    phase: tuple          # (3, K) rad

    def __post_init__(self):
        rows = [tuple(tuple(float(x) for x in row) for row in getattr(self, name))
                for name in ("amplitude", "frequency", "phase")]
        shapes = {np.shape(r) for r in rows}
        if len(shapes) != 1 or len(rows[0]) != 3 or not np.isfinite(np.asarray(rows)).all():
            raise ValueError("Turbulence needs finite (3, K) amplitude, frequency and phase arrays")
        for name, value in zip(("amplitude", "frequency", "phase"), rows):
            object.__setattr__(self, name, value)

    @cached_property
    def _arrays(self):
        return (np.asarray(self.amplitude), 2.0*np.pi*np.asarray(self.frequency), np.asarray(self.phase))

    def velocity(self, t):
        amplitude, angular, phase = self._arrays
        return (amplitude*np.sin(angular*t+phase)).sum(axis=1)

    @property
    def std(self):
        """Standard deviation per axis (m/s)."""
        return np.sqrt(0.5*np.square(np.asarray(self.amplitude)).sum(axis=1))


@dataclass(frozen=True)
class Wind:
    """A scenario's air velocity: `mean` throughout, plus gusts and turbulence from t = 0."""
    mean: tuple = (0.0, 0.0, 0.0)     # world-frame mean air velocity (m/s)
    gusts: tuple = ()                 # Gust events
    turbulence: Turbulence | None = None

    def __post_init__(self):
        object.__setattr__(self, "mean", _vector(self.mean, "Wind.mean"))
        object.__setattr__(self, "gusts", tuple(sorted(self.gusts, key=lambda g: g.onset)))

    @property
    def steady(self):
        """True when the air velocity never changes (mean wind only)."""
        return not self.gusts and self.turbulence is None

    def velocity(self, t):
        """World-frame air velocity (3,) at episode time t (s)."""
        velocity = np.array(self.mean)
        for gust in self.gusts:
            if gust.onset < t < gust.onset+gust.duration:
                velocity += gust.velocity(t)
        if self.turbulence is not None:
            velocity += self.turbulence.velocity(t)
        return velocity

    def velocities(self, times):
        """Air velocities (T, 3) at episode times (T,), e.g. for plots."""
        return np.stack([self.velocity(float(t)) for t in np.asarray(times, dtype=float).reshape(-1)])

    def to_dict(self):
        values = dict(mean=list(self.mean),
                      gusts=[dict(onset=g.onset, duration=g.duration, peak=list(g.peak)) for g in self.gusts])
        if self.turbulence is not None:
            values["turbulence"] = {name: [list(row) for row in getattr(self.turbulence, name)]
                                    for name in ("amplitude", "frequency", "phase")}
        return values

    @classmethod
    def from_dict(cls, values):
        turbulence = values.get("turbulence")
        return cls(mean=tuple(values.get("mean", (0.0, 0.0, 0.0))),
                   gusts=tuple(Gust(onset=g["onset"], duration=g["duration"], peak=tuple(g["peak"]))
                               for g in values.get("gusts", ())),
                   turbulence=None if turbulence is None else Turbulence(**turbulence))
