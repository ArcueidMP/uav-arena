"""Bit-level fingerprint of the arena at its default settings, for arena-v0.3 parity.

The script uses only the API that this package shares with the frozen research code, so it runs
against either one, whichever `uav_arena` is importable:

    PYTHONPATH=/path/to/arena-v0.3-checkout python tools/fingerprint.py --out frozen.npz
    python tools/fingerprint.py --out release.npz
    python tools/fingerprint.py --compare frozen.npz release.npz

The fingerprint covers scenario generation on every dev and test seed, closed-loop environment
steps (physics, onboard Lee controller, observations, rewards, events), planning forks, the Gymnasium
wrapper, the three reference planners for a few decisions, and recollected training episodes.
tests/golden/arena_v03.npz is the fingerprint of the frozen code, and tests/test_parity.py checks
the release against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

ENV_CASES = ((0, "static/0ms"), (3, "static+moving/100ms"), (7, "static/200ms"), (12, "static+moving/50ms"))
DECISIONS = 40
PLAN_ROWS = ("physics", "adaptive", "oracle")
PLAN_DECISIONS = 4
COLLECT_SEEDS = (200_000, 200_001)


def _versions():
    import mujoco
    versions = dict(python=platform.python_version(), numpy=np.__version__, mujoco=mujoco.__version__,
                    machine=platform.machine(), system=platform.system())
    for name in ("torch", "gymnasium", "scipy"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    compiler = shutil.which("cc")
    versions["cc"] = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0] if compiler else None
    return versions


def _flat(obs):
    """An observation dict as one float64 vector, keys in sorted order."""
    return np.concatenate([np.asarray(obs[k], dtype=float).reshape(-1) for k in sorted(obs) if k != "images"])


def _state(env):
    data = env.data
    return np.concatenate([data.qpos, data.qvel, data.act, data.ctrl, [data.time],
                           np.frombuffer(env.lee.capture(), dtype=np.uint8).astype(float)])


def scenarios():
    from uav_arena.scenarios import CONDITIONS, SUITE_SEEDS, make_scenario
    items = []
    for name in ("dev", "test"):
        for condition in CONDITIONS:
            for seed in SUITE_SEEDS[name]:
                scenario = make_scenario(seed, condition)
                items.append([scenario.to_dict(), scenario.public().to_dict()])
    return {"scenarios": np.str_(json.dumps(items))}


def environment():
    from uav_arena.env import ArenaEnv
    from uav_arena.scenarios import CONDITIONS, make_scenario
    out = {}
    for seed, name in ENV_CASES:
        condition = next(c for c in CONDITIONS if c.name == name)
        env = ArenaEnv(make_scenario(seed, condition))
        obs = env.observation()
        rng = np.random.default_rng([seed, 99])
        states, observations, rewards = [_state(env)], [_flat(obs)], []
        for k in range(DECISIONS):
            target = obs["recent_setpoints"][-1]+rng.uniform(-0.1, 0.1, (3, 3))
            obs, reward, done, info = env.step(target)
            states.append(_state(env)); observations.append(_flat(obs)); rewards.append(reward)
            if k == 9:
                fork = env.physics_state()
                anchor = obs["recent_setpoints"][-1]
                sequence = anchor+np.cumsum(np.random.default_rng([seed, 98]).uniform(-0.08, 0.08, (15, 3, 3)), axis=0)
            if done:
                break
        key = f"env/{seed}/{name}"
        # Observations grow while the setpoint history fills up: store them end to end with their lengths.
        out[key+"/states"], out[key+"/observations"] = np.stack(states), np.concatenate(observations)
        out[key+"/observation_lengths"] = np.array([len(o) for o in observations])
        out[key+"/rewards"], out[key+"/outcome"] = np.array(rewards), np.str_(info["outcome"])
        env.set_physics_state(fork)
        rollout = env.simulate(sequence)
        out[key+"/simulate"] = np.concatenate([rollout[k].reshape(len(sequence), -1) for k in sorted(rollout)], axis=1)
    return out


def gym():
    from uav_arena.gym_env import ArenaGymEnv
    env = ArenaGymEnv()
    obs, info = env.reset(seed=0)
    rng = np.random.default_rng(5)
    observations, rewards, flags = [obs], [], []
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(rng.uniform(-1, 1, 9).astype(np.float32))
        observations.append(obs); rewards.append(reward); flags.append((terminated, truncated))
        if terminated or truncated:
            break
    obs, info = env.reset(options=dict(seed=1003, condition="static+moving/200ms"))
    return {"gym/observations": np.stack(observations), "gym/rewards": np.array(rewards),
            "gym/flags": np.array(flags), "gym/reset_options": obs, "gym/scenario": np.str_(info["condition"])}


def planning():
    from uav_arena.env import ArenaEnv
    from uav_arena.planning import ArenaCost, ArenaPlanner
    from uav_arena.run import make_predictor
    from uav_arena.scenarios import CONDITIONS, make_scenario
    out = {}
    for row in PLAN_ROWS:
        scenario = make_scenario(2, CONDITIONS[1])
        env = ArenaEnv(scenario)
        obs = env.observation()
        predictor = make_predictor(row, workers=2)
        planner = ArenaPlanner(ArenaCost(predictor), samples=32, horizon=15)
        planner.reset(scenario.seed)
        chosen = []
        try:
            for _ in range(PLAN_DECISIONS):
                setpoints = planner.act(obs, env)
                chosen.append(setpoints)
                obs, *_ = env.step(setpoints)
        finally:
            predictor.close()
        out[f"plan/{row}/setpoints"] = np.stack(chosen)
        if getattr(predictor, "estimates", None) is not None:
            out[f"plan/{row}/estimates"] = np.array(predictor.estimates)
    return out


def collect():
    from uav_arena.collect import collect_episode, content_sha256
    return {f"collect/{seed}": np.str_(content_sha256(collect_episode(seed))) for seed in COLLECT_SEEDS}


PARTS = dict(scenarios=scenarios, environment=environment, gym=gym, planning=planning, collect=collect)


def fingerprint(parts=tuple(PARTS)):
    arrays = {}
    for part in parts:
        arrays.update(PARTS[part]())
    return arrays


def save(path, arrays, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, meta=np.str_(json.dumps(meta)))


def load(path):
    with np.load(path) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, json.loads(str(arrays.pop("meta")))


def compare(reference, candidate, *, exact=True, rtol=1e-7, atol=1e-9):
    """Keys whose arrays differ: bit for bit when `exact`, else beyond the tolerance (strings always exactly)."""
    differ = []
    for key in sorted(set(reference) | set(candidate)):
        if key not in reference or key not in candidate:
            differ.append(f"{key}: missing")
            continue
        a, b = np.asarray(reference[key]), np.asarray(candidate[key])
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            differ.append(f"{key}: shape/dtype {a.shape}{a.dtype} vs {b.shape}{b.dtype}")
        elif a.dtype.kind in "USO" or exact:
            if a.tobytes() != b.tobytes():
                diff = "" if a.dtype.kind in "USO" else f" (max |diff| {np.nanmax(np.abs(a.astype(float)-b.astype(float))):.3g})"
                differ.append(f"{key}: not identical{diff}")
        elif not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            differ.append(f"{key}: max |diff| {np.nanmax(np.abs(a-b)):.3g}")
    return differ


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="write the fingerprint of the importable uav_arena here")
    parser.add_argument("--parts", default=",".join(PARTS), help=f"comma-separated subset of {','.join(PARTS)}")
    parser.add_argument("--label", help="name of the code being fingerprinted (default: its directory)")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("REFERENCE", "CANDIDATE"))
    args = parser.parse_args()
    if args.compare:
        (reference, meta_a), (candidate, meta_b) = (load(p) for p in args.compare)
        differ = compare(reference, candidate)
        print(json.dumps(dict(reference=meta_a, candidate=meta_b, keys=len(reference), differ=differ), indent=1))
        sys.exit(1 if differ else 0)
    if args.out is None:
        parser.error("--out or --compare is required")
    import uav_arena
    parts = tuple(p for p in args.parts.split(",") if p)
    meta = dict(source=args.label or str(Path(uav_arena.__file__).resolve().parent), parts=parts,
                version=getattr(uav_arena, "__version__", "arena-v0.3"), versions=_versions())
    save(args.out, fingerprint(parts), meta)
    print(f"fingerprint of {meta['source']} -> {args.out}")


if __name__ == "__main__":
    main()
