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
  the frozen code made by `tools/fingerprint.py`. The fingerprint holds 36 arrays:
  - the scenario dictionaries (public and true) of every `dev` and `test` seed in all 8 conditions;
  - four closed-loop episodes of 40 decisions. For each: the full MuJoCo state, rotor commands and
    onboard controller memory after every decision; observations, rewards and outcomes; and a
    15-decision `simulate` rollout from a planning fork;
  - the Gymnasium wrapper's observations, rewards and episode flags;
  - the first four decisions of the oracle, physics and adaptive rows (shared MPPI, 32 samples),
    including the adaptive row's mass estimates;
  - the content hashes of two recollected training episodes.
- **`tests/test_manifests.py`** recollects training and validation episodes and compares their content
  hashes with the published manifests of the standard datasets (`uav_arena/manifests/`). The research
  code collected those datasets.

Both checks have passed bit for bit on Linux x86-64 with MuJoCo 3.3.0 and gcc 15.2, in two
environments: Python 3.10 with NumPy 2.2.6 and PyTorch 2.11 (the research environment), and
Python 3.12 with NumPy 2.5.4, SciPy 1.18 and PyTorch 2.15 (CPU). On another platform, floating-point
results can differ in the last bits. The parity test then falls back to a tight tolerance and skips
the dataset hashes.

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
