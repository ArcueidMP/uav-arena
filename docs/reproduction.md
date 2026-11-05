# Reproducing arena-v0.3

arena-v0.3 is the frozen version of the research code: tag
[`arena-v0.3`](https://github.com/ArcueidMP/vincair-multi-agent-wm/tree/arena-v0.3) (commit 64134e7)
of ArcueidMP/vincair-multi-agent-wm. That tag froze the environment, the scenario distribution, the
shared cost, the planner and its settings, the data collector and the standard datasets. This
release was extracted from that repository's `uav-arena` branch at commit 0fdb022. Its core files
are identical to the tag; the Gymnasium wrapper was added after the tag.

**With the default settings, this release reproduces arena-v0.3 bit for bit.** Two test files check
this.

## The checks

- **`tests/test_parity.py`** compares the release with `tests/golden/arena_v03.npz`, a fingerprint of
  the frozen code made by `tools/fingerprint.py`. The fingerprint holds 46 arrays:
  - the scenario dictionaries (public and true) of every `dev` and `test` seed in all 8 conditions,
    and keyword layouts with 8, 12 and 16 pillars;
  - four closed-loop episodes of up to 40 decisions. For each: the full MuJoCo state, rotor commands
    and onboard controller memory after every decision; observations, rewards, clearances and
    outcomes; and a 15-decision `simulate` rollout from a planning fork;
  - 43 whole episodes that end in every kind of outcome. 40 are straight-line transfers on 5 `dev`
    layouts in all 8 conditions (24 obstacle collisions and 16 successes). Three follow simple setpoint
    rules that end in a timeout, a crash and an exit from the workspace. Each keeps its outcome, time,
    clearances and final state;
  - the Gymnasium wrapper's observations, rewards and episode flags;
  - the first four decisions of the oracle, physics and adaptive rows (shared MPPI, 32 samples),
    including the adaptive row's mass estimates, and four oracle decisions taken mid-flight among
    pillars and movers;
  - the content hashes of two recollected training episodes.
- The research code's arena tests (harness, recorder, open loop, training windows, replay, example
  participant and reference rows) are ported unchanged, minus two that test modules this release does
  not ship.
- **`tests/test_manifests.py`** recollects training and validation episodes and compares their content
  hashes with the published manifests of the standard datasets (`uav_arena/manifests/`). The research
  code collected those datasets.

Both checks have passed bit for bit on Linux x86-64 with MuJoCo 3.3.0, gcc 15.2 and glibc 2.43, in two
environments: Python 3.10 with NumPy 2.2.6 and PyTorch 2.11 (the research environment), and
Python 3.12 with NumPy 2.5.4, SciPy 1.18 and PyTorch 2.15 (CPU). So the Python, NumPy and PyTorch
versions do not matter; the platform can. A different CPU architecture, MuJoCo build, C compiler or C
library can change floating-point results in the last bits. The research code's datasets, for
example, do not reproduce on Ubuntu 22.04. On such a platform the parity test reports a mismatch as
a skip that names the difference, never as a pass; set `UAV_ARENA_STRICT_PARITY=1` to make it fail.

## Fingerprinting the frozen code yourself

```bash
git clone https://github.com/ArcueidMP/vincair-multi-agent-wm.git research
git -C research checkout 0fdb022
PYTHONPATH=research python tools/fingerprint.py --out frozen.npz
python tools/fingerprint.py --out release.npz
python tools/fingerprint.py --compare frozen.npz release.npz
```

Run both commands in one environment that has the `plan` extra installed. Without `PYTHONPATH`,
`fingerprint.py` fingerprints the installed release.

## What changed from the research code

None of these changes alter results at the default settings:

- **Layout.** `uav_transport.plant`, `menagerie` and `snapshot` became `uav_arena.sim.*`. The Lee
  rotor mixer (`rotor_mixer`) moved into `sim/plant.py` unchanged. The vendored Menagerie and
  firmware files moved under `uav_arena/third_party/` byte for byte; their pinned SHA-256 hashes are
  still checked at every model build and controller compilation.
- **Onboard controller.** The C wrapper is byte-identical, so the compiled source and its build key
  match. The build cache moved from the repository's `data/` to the user cache directory.
- **Settings.**
  - `Scenario` gained `nominal_mass`, `payload_prior`, `wind` and `hidden`, with arena-v0.3 values
    as defaults. `to_dict()` leaves them out while they hold those values.
  - The environment reads the feedforward mass from the scenario (25 g by default), and touches
    MuJoCo's air velocity only when there is wind.
  - The physics and adaptive rows read the nominal mass and the mass prior from the public scenario.
    The values are the same by default.
- **Removed.**
  - The learned participants (JEPA, PETS, W-ar, TD-MPC2, DreamerV3, CrazyMARL) and their training
    code, the CrazyMARL motor pilot in `replay.py`, and the analysis scripts.
  - The research results and archives.

  All of these stay in the research repository, with the trained models and the results tables.
- **Provenance.** `collect.py`'s manifests also record the package version.
