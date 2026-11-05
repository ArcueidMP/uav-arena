# Task settings

`TaskSettings` defines a scenario family: the distribution that `make_scenario` draws a scenario from
for each layout seed. Every default is the arena-v0.3 task, so `TaskSettings()` reproduces arena-v0.3
bit for bit ([reproduction.md](reproduction.md)). Results under any other settings belong to a new
scenario family. Publish the settings file with them, and do not compare them with arena-v0.3 tables.

```python
from uav_arena import CONDITIONS, PayloadSettings, TaskSettings, WindSettings, make_scenario

settings = TaskSettings(payload=PayloadSettings(mass_range=(0.02, 0.06)), wind=WindSettings(speed=1.5))
scenario = make_scenario(seed=7, condition=CONDITIONS[2], settings=settings)
```

## How settings become a scenario

- **Seeded draws.** A layout seed fixes every draw. The payload mass comes from the random stream
  `[seed, 13]`, as in arena-v0.3. The wind comes from four separate streams `[seed, 17, k]`: one each
  for the mean speed, the mean heading, the gusts and the turbulence. Changing one wind part leaves
  the other parts unchanged. All eight conditions of a seed share the layout, the mass and the wind.
- **Overrides.** Keyword arguments of `make_scenario` win over the settings: `payload_mass=0.03`
  fixes the true mass, `wind=Wind(...)` fixes the true wind, and `n_pillars=` and the other obstacle
  arguments set the layout.
- **What a scenario carries.** `payload_mass` and `wind` are the true values. `nominal_mass` and
  `payload_prior` are public. `hidden` names the fields that `Scenario.public()` removes.
  `Scenario.to_dict()` leaves out every release field that holds its arena-v0.3 default, so
  arena-v0.3 scenario dictionaries, and the content hashes of datasets that store them, are
  unchanged.
- **JSON.** `TaskSettings.to_dict()` / `TaskSettings.from_dict()` and `load_settings(path)` read and
  write settings. Every tool takes them as `--settings file.json`. Fields left out keep their
  defaults:

```json
{
  "payload": {"mass_range": [0.02, 0.06]},
  "wind": {"speed": [0.0, 2.0], "heading": null,
           "gusts": {"rate": 0.2, "peak": [1.0, 3.0], "duration": [0.5, 2.0]},
           "turbulence": {"intensity": [0.3, 0.3, 0.1], "time_scale": 1.0}},
  "obstacles": {"n_movers": 3},
  "time_limit": 25.0
}
```

## Payload

| Field | Meaning | Default |
|---|---|---|
| `mass_range` | each layout seed draws the payload mass uniformly from this range (kg) | (0.010, 0.040) |
| `mass` | one fixed mass for every scenario instead (kg) | None |
| `nominal_mass` | the publicly known mass (kg): the onboard cable-load feedforward and the physics row assume it | None: the middle of the masses the family can draw (25 g) |
| `hidden` | hide the true mass from non-privileged methods | True |
| `max_hover_fraction` | refuse families whose heaviest payload needs more than this fraction of rotor thrust to hover | 0.75 |

**Physics.** The payload is a 2.5 cm sphere hanging from all three cables at one point, so its mass
sets the cable tensions. Each aircraft's onboard feedforward adds the cable load of
`nominal_mass`, and the Lee position integral absorbs the difference from the true mass.

**How heavy.** The table below lists the largest rotor thrust each aircraft needs to hover the
formation statically (`hover_rotor_fraction`), next to the peak measured while holding the hover for
3 s:

| Payload | Static hover thrust | Measured peak in a 3 s hover |
|---|---|---|
| 10 g | 46.6 % | 46.6 % |
| 25 g | 54.5 % | 54.5 % |
| 40 g | 62.5 % | 62.5 % |
| 60 g | 73.4 % | 73.4 % |
| 80 g | 84.4 % | 84.2 % |
| 100 g | 95.4 % | crashed during the warm-up |
| 120 g | over 100 % | refused: no hover exists |

`max_hover_fraction=0.75` admits payloads up to 62.9 g (`max_payload_mass(0.75)`), which leaves a
quarter of the thrust for control. Raise it to accept less headroom.

**What methods see.** By default the true mass is hidden, and the nominal mass and the prior range
are public. The physics row plans with the nominal mass, or with the true mass when `hidden=False`.
The adaptive row estimates the mass over a 2 g grid spanning the prior (16 candidates for
10–40 g), or uses the true mass when it is public. The oracle always uses the true mass.

## Wind

| Field | Meaning | Default |
|---|---|---|
| `speed` | mean wind speed (m/s): a number, or a (low, high) range drawn per layout seed | 0 |
| `heading` | mean wind direction in degrees counterclockwise from +x, the direction the air moves toward: 0 is a tailwind for the start-to-goal transfer, 180 a headwind; a number, a range, or None for any direction | None |
| `gusts` | `GustSettings`, or None | None |
| `turbulence` | `TurbulenceSettings`, or None | None |
| `hidden` | hide the wind from non-privileged methods | True |

**Model.** The air velocity is uniform in space and a pure function of episode time. It is written
into MuJoCo's `opt.wind` before every 2 ms physics step, and MuJoCo's inertia-box fluid model turns it
into drag on every body (air density 1.225 kg/m³, viscosity 1.8e-5 Pa·s, the Menagerie values). The
mean wind already blows during the 5 s warm-up, so episodes start from a hover settled in it. Gusts
and turbulence act from t = 0. Because the wind depends only on time, snapshots, planning forks and
replays stay exact. The oracle's rollouts in gusts and turbulence equal the episode bit for bit
(`tests/test_wind.py`).

**Gusts** (`GustSettings`) are discrete "1-cosine" gusts. A gust with peak velocity **v**, onset t₀
and duration T adds **v** · ½(1 − cos(2π(t − t₀)/T)) for t₀ < t < t₀ + T. Onsets form a Poisson
process with `rate` gusts per second. Each gust draws its peak speed from `peak` (m/s), its
duration from `duration` (s), and its horizontal direction from `heading` (degrees; None for any
direction). The vertical component of a gust is uniform within ±`vertical` × the peak speed.
Defaults: 0.2 gusts/s, 1–3 m/s, 0.5–2 s, any direction, horizontal only.

**Turbulence** (`TurbulenceSettings`) is continuous and zero-mean. On each world axis it is a sum of
`components` sinusoids whose powers follow a first-order low-pass (Ornstein–Uhlenbeck) spectrum with
correlation time `time_scale`. The frequencies are log-spaced, with a random offset inside each band,
from 1/30 to 30 times the corner frequency 1/(2π `time_scale`). The phases are random. By
construction the standard deviation on each axis equals `intensity` exactly. Defaults:
(0.4, 0.4, 0.2) m/s, 1 s, 24 components.

**Measured effects** (Menagerie Crazyflie 2 team in hover, 25 g payload):

| Wind | Effect |
|---|---|
| steady 1 / 2 / 3 m/s | 3 / 12 / 27 mN of drag per aircraft (its weight is 265 mN); the team leans into the wind |
| steady 3 m/s, after the warm-up | payload speed 0.4 cm/s at t = 0, drift under 0.5 cm over the next 3 s |
| steady 5 m/s, after the warm-up | payload speed 1.2 cm/s at t = 0, drift 1.6 cm over 3 s |
| one 1 s gust, peak 1 / 2 / 3 / 5 m/s | payload displaced by up to 0.8 / 3.1 / 6.8 / 17.9 cm |
| gusts and turbulence | 1.77 ms of computation per 100 ms decision, against 1.34 ms in still air |

**Model limits.** Drag is quadratic in airspeed on each body's equivalent inertia box. Rotor drag,
roughly linear in airspeed and dominant for small quadrotors at low speed, is not modeled. The
cables feel no air. The wind has no spatial structure: no shear, no wakes behind the pillars, and no
downwash between aircraft.

**What methods see.** By default the wind is hidden. The physics and adaptive rows then plan in
still air, and the oracle knows the wind exactly. With `hidden=False` the public scenario carries
the wind, and the physics rows plan with it.

## Obstacles

| Field | Meaning | Default |
|---|---|---|
| `n_pillars` | vertical cylinders between start and goal, in every condition | 4 |
| `pillar_radius` | radius range (m) | (0.10, 0.18) |
| `n_movers` | spheres moving back and forth across the corridor, in the `static+moving` conditions | 2 |
| `mover_radius` | sphere radius (m) | 0.15 |
| `mover_speed` | speed range (m/s) | (0.25, 0.45) |

Every layout keeps a corridor that the whole team fits through. The Gymnasium observation grows with
the counts: its length is 6 + 45 + 3 + 3·`n_pillars` + 7·`n_movers` + 36 + 2.

## Episode

`time_limit` (20 s) ends an episode as a timeout. `goal_tolerance` (0.15 m) is the payload-to-goal
distance that counts as arrived. Success also needs the payload slower than 0.2 m/s for 5
consecutive decisions.

## Reproducibility

Scenario draws use NumPy's PCG64 generator, so the same settings and seed give the same scenario
on any machine. The physics is deterministic on one platform. Across platforms (CPU architecture,
compiler, NumPy build), trajectories can differ in the last bits.
