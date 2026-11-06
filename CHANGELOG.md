# Changelog

## 0.1.0 (2026-11-06)

The first release, extracted from the frozen research code (arena-v0.3,
[docs/reproduction.md](docs/reproduction.md)). With the default settings it reproduces arena-v0.3
bit for bit.

### Added

- `TaskSettings`, a tunable scenario family, with JSON round-trips and `--settings` on `run`, `collect`,
  `record` and `replay record` (`openloop` and `harness` read the family from the episode files):
  - payload mass: a range or a fixed value, the nominal mass the controllers assume, a public prior,
    hidden or public, and a thrust-headroom check;
  - wind: a mean wind, discrete 1-cosine gusts and continuous turbulence, all pure functions of
    episode time, hidden or public;
  - obstacle counts, sizes and speeds, with a packing check and bounded layout sampling; time limit
    and goal tolerance.
- `ArenaEnv` refuses a scenario whose team loses its start hover in the warm-up (too much wind or
  payload), instead of starting an episode that is already lost.
- Virtual cameras (`uav_arena.cameras`):
  - presets for onboard, downward, chase, payload-following and overview cameras, plus custom ones;
  - RGB or depth images, drawn from the same delayed state as the observation;
  - a private render model, so the physics is unchanged.
- `ArenaGymEnv`: settings, camera observations as a `Dict` space, and `render_mode="rgb_array"`.
- Replay videos draw the wind as an arrow over the payload.
- Parity tests against a fingerprint of the frozen code and against the published dataset
  manifests (`tests/`), and `tools/fingerprint.py`. The research code's arena tests are ported
  alongside.
- Packaging: `pyproject.toml` with the extras `gym`, `plan`, `video`, `dev` and `all`.

### Changed

- The plant modules moved from `uav_transport` to `uav_arena.sim`, and the vendored files to
  `uav_arena/third_party/`. The compiled controller is cached per platform in `~/.cache/uav-arena`, and
  a finished cache can be read-only.
- The physics and adaptive rows read the nominal mass and the mass prior from the public scenario.

### Removed

- The learned participants, their training code, the analysis scripts and the research results.
  They stay in the research repository.
