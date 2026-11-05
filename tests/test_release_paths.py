"""Release-only paths: tuned settings through every tool, and the guards added after review."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from uav_arena import (CONDITIONS, ArenaEnv, GustSettings, PayloadSettings, TaskSettings, TurbulenceSettings, Wind,
                       WindSettings, make_scenario)

GUSTY = TaskSettings(wind=WindSettings(speed=1.5, heading=90.0, gusts=GustSettings(rate=1.0),
                                       turbulence=TurbulenceSettings()))


# ---------- random streams ----------
def test_random_streams_are_disjoint():
    """No two parts of the arena share a random stream. NumPy pads short seeds with zeros, so compare padded."""
    seed = 200_123
    streams = {"payload mass": [seed, 13], "layout": [seed, 7], "movers": [seed, 11],
               "explore-v1": [seed, 17], "explore-v2": [seed, 17, 2],
               **{f"wind part {k}": [seed, 19, k] for k in range(1, 5)}}
    padded = {name: tuple(s+[0]*(4-len(s))) for name, s in streams.items()}
    assert len(set(padded.values())) == len(padded)
    first = {name: np.random.default_rng(s).random() for name, s in streams.items()}
    assert len(set(first.values())) == len(first)


def test_the_wind_draws_from_its_own_streams():
    """The hidden wind does not repeat the data collector's behavior stream (it once shared [seed, 17, 2])."""
    seed, settings = 200_007, GUSTY
    wind = make_scenario(seed, CONDITIONS[0], settings=settings).wind
    gusts = settings.wind.gusts
    assert wind.gusts == gusts.draw(np.random.default_rng([seed, 19, 3]), settings.time_limit)
    assert wind.gusts != gusts.draw(np.random.default_rng([seed, 17, 2]), settings.time_limit)
    turbulence = settings.wind.turbulence
    assert wind.turbulence == turbulence.draw(np.random.default_rng([seed, 19, 4]))


# ---------- payload ----------
def test_public_prior_hides_a_fixed_mass():
    hidden = TaskSettings(payload=PayloadSettings(mass=0.05, public_prior=(0.02, 0.06)))
    scenario = make_scenario(1, CONDITIONS[0], settings=hidden)
    assert scenario.payload_mass == 0.05 and scenario.payload_prior == (0.02, 0.06)
    assert scenario.nominal_mass == pytest.approx(0.04) and scenario.public().payload_mass is None
    with pytest.raises(ValueError, match="public_prior"):
        PayloadSettings(mass_range=(0.02, 0.05), public_prior=(0.03, 0.06))


def test_the_onboard_feedforward_uses_the_scenario_nominal_mass():
    def hover(nominal):
        settings = TaskSettings(payload=PayloadSettings(mass_range=(0.04, 0.06), nominal_mass=nominal))
        env = ArenaEnv(make_scenario(2, CONDITIONS[0], settings=settings))
        return env.nominal_mass, env.data.ctrl.copy()
    (light_nominal, light), (heavy_nominal, heavy) = hover(0.025), hover(0.05)
    assert (light_nominal, heavy_nominal) == (0.025, 0.05)
    assert not np.array_equal(light, heavy)          # the controller is told a different cable load


# ---------- warm-up ----------
def test_the_warmup_blows_only_the_mean_wind():
    steady = make_scenario(4, CONDITIONS[0], wind=Wind(mean=(1.0, 0.5, 0.0)))
    gusty = dataclasses.replace(steady, wind=make_scenario(4, CONDITIONS[0], settings=GUSTY).wind)
    gusty = dataclasses.replace(gusty, wind=Wind(mean=(1.0, 0.5, 0.0), gusts=gusty.wind.gusts,
                                                 turbulence=gusty.wind.turbulence))
    a, b = ArenaEnv(steady), ArenaEnv(gusty)
    assert np.array_equal(a.data.qpos, b.data.qpos) and np.array_equal(a.data.ctrl, b.data.ctrl)
    target = a.observation()["recent_setpoints"][-1]
    a.step(target)
    b.step(target)
    assert not np.array_equal(a.data.qpos, b.data.qpos)      # gusts and turbulence start at t = 0


def test_a_team_that_cannot_hold_its_hover_is_refused():
    with pytest.raises(ValueError, match="could not hold its start hover"):
        ArenaEnv(make_scenario(0, CONDITIONS[0], wind=Wind(mean=(15.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match="could not hold its start hover"):
        ArenaEnv(make_scenario(0, CONDITIONS[0], payload_mass=0.1))
    ArenaEnv(make_scenario(0, CONDITIONS[0], wind=Wind(mean=(5.0, 0.0, 0.0))))      # still fine


# ---------- reference rows ----------
@pytest.mark.plan
def test_physics_rows_read_only_public_fields_whatever_they_are_given():
    pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    from uav_arena.planning import AdaptivePhysicsPredictor, PhysicsPredictor
    scenario = make_scenario(0, CONDITIONS[0], settings=GUSTY)
    env = ArenaEnv(scenario)
    obs = env.observation()
    for given in (scenario, scenario.public()):
        physics = PhysicsPredictor(workers=1)
        try:
            physics.observe(obs, given)
        finally:
            physics.close()
        assert physics.scenario.payload_mass == scenario.nominal_mass and physics.scenario.wind is None
        adaptive = AdaptivePhysicsPredictor(workers=1)
        try:
            adaptive.observe(obs, given)
        finally:
            adaptive.close()
        assert len(adaptive.candidates) == 16


# ---------- tools on tuned scenario families ----------
def test_collected_windy_episodes_keep_the_wind_out_of_training_windows(tmp_path):
    pytest.importorskip("torch")
    from uav_arena.collect import _write
    from uav_arena.windows import load
    settings = dataclasses.replace(GUSTY, time_limit=4.0)
    _write((200_011, tmp_path, "explore-v2", settings))
    with np.load(tmp_path/"episode_200011.npz") as f:
        assert "wind" in f.files and "payload_mass" in f.files
        layout = json.loads(str(f["scenario"]))
        assert "wind" not in layout and layout["payload_mass"] is None
    (episode,) = load(tmp_path)
    assert "wind" not in episode and "payload_mass" not in episode


@pytest.mark.plan
def test_the_oracle_scores_zero_on_a_gusty_recording(tmp_path):
    """record.py stores the true wind, and the harness's truth rolls out the true, windy scenario."""
    pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    import multiprocessing as mp
    from uav_arena import harness
    from uav_arena.collect import content_sha256
    from uav_arena.planning import ArenaCost, ArenaPlanner, TrueSimPredictor
    from uav_arena.record import RecordingPlanner, episode_name, load, record_episode, true_scenario
    scenario = dataclasses.replace(make_scenario(3, CONDITIONS[1], settings=GUSTY), time_limit=1.6)
    oracle = TrueSimPredictor(workers=2)
    try:
        recorder = RecordingPlanner(ArenaPlanner(ArenaCost(oracle), samples=16, horizon=5))
        _, arrays = record_episode(scenario, recorder)
    finally:
        oracle.close()
    np.savez_compressed(tmp_path/episode_name(scenario), **arrays)
    (tmp_path/"manifest.json").write_text(json.dumps(dict(episodes=[dict(file=episode_name(scenario),
                                                                          content_sha256=content_sha256(arrays))])))
    episode = load(tmp_path/episode_name(scenario))
    assert episode["layout"].wind is None and true_scenario(episode).wind == scenario.wind
    harness.compute_truth(tmp_path, 2)
    oracle = TrueSimPredictor(workers=2)
    try:
        with np.load(tmp_path/harness.TRUTH/episode_name(scenario)) as f:
            truth = {key: f[key] for key in f.files}
        with mp.get_context("fork").Pool(2) as pool:
            rows = harness.score_episode(oracle, pool, episode, truth, float(episode["config"]["temperature"]))
    finally:
        oracle.close()
    for key in ("l1_payload", "l1_uav", "l2_error", "l3_regret"):
        assert max(abs(float(np.max(np.abs(r[key])))) for r in rows) == 0.0, key


def test_gym_wrapper_follows_the_obstacle_settings():
    pytest.importorskip("gymnasium")
    from uav_arena.gym_env import ArenaGymEnv, observation_size
    from uav_arena.settings import ObstacleSettings
    env = ArenaGymEnv(settings=TaskSettings(obstacles=ObstacleSettings(n_pillars=6, n_movers=3)))
    obs, _ = env.reset(options=dict(seed=5, condition="static+moving/50ms"))
    assert obs.shape == (observation_size(6, 3),) and env.observation_space.contains(obs)
    obs, *_ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)


def test_example_trainer_refuses_too_little_data(tmp_path):
    pytest.importorskip("torch")
    from uav_arena.collect import _write
    from uav_arena.participants.example import train
    _write((200_000, tmp_path))
    with pytest.raises(SystemExit):
        train.main(["--data", str(tmp_path), "--episodes", "1", "--batch", "100000", "--steps", "1",
                    "--workers", "0", "--out", str(tmp_path/"model.pt")])


# ---------- cameras ----------
@pytest.mark.render
def test_camera_environments_refuse_snapshots_without_scenes(can_render):
    from uav_arena.cameras import onboard_cameras
    scenario = make_scenario(1, CONDITIONS[7])
    plain, filmed = ArenaEnv(scenario), ArenaEnv(scenario, cameras=onboard_cameras(width=16, height=16))
    try:
        snapshot = plain.capture()
        before = filmed.capture()
        with pytest.raises(ValueError, match="snapshot captured with cameras"):
            filmed.restore(snapshot)
        after = filmed.capture()
        assert np.array_equal(before["plant"].state, after["plant"].state) and len(before["log"]) == len(after["log"])
        filmed.set_physics_state(plain.physics_state())          # physics snapshots stay interchangeable
    finally:
        filmed.close()


@pytest.mark.render
def test_render_keeps_a_bounded_number_of_rigs(can_render):
    from uav_arena.cameras import Camera
    from uav_arena.env import RENDER_RIGS
    env = ArenaEnv(make_scenario(0, CONDITIONS[0]))
    try:
        for k in range(RENDER_RIGS+3):
            env.render(Camera(f"orbit{k}", pos=(3.0*np.cos(k), 3.0*np.sin(k), 2.0), lookat=(0.0, 0.0, 1.0),
                              width=16, height=16))
        assert len(env._render_rigs) == RENDER_RIGS
    finally:
        env.close()


@pytest.mark.render
def test_chase_cameras_do_not_depend_on_the_hidden_mass(can_render):
    from uav_arena.cameras import CameraRig, chase_cameras
    from uav_arena.env import scene_xml
    camera = chase_cameras(width=48, height=48)[0]
    light, heavy = (ArenaEnv(make_scenario(2, CONDITIONS[0], payload_mass=m)) for m in (0.010, 0.040))
    rigs = [CameraRig(scene_xml(env.config, env.scenario), (camera,)) for env in (light, heavy)]
    try:
        images = [rig.render_scene(light.data.qpos, light.data.mocap_pos)[camera.name] for rig in rigs]
        difference = np.abs(images[0].astype(int)-images[1].astype(int))
        assert difference.max() <= 3 and (difference.max(-1) > 0).mean() < 0.01
    finally:
        for rig in rigs:
            rig.close()


# ---------- controller build cache ----------
def test_a_prebuilt_controller_cache_can_be_read_only(tmp_path):
    cache = tmp_path/"native"
    code = "from uav_arena.lowlevel import native_library; native_library(); print('ok')"
    env = dict(os.environ, UAV_ARENA_BUILD_CACHE=str(cache), PYTHONDONTWRITEBYTECODE="1")
    assert subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.strip() == "ok"
    for path in sorted(cache.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    cache.chmod(0o555)
    try:
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        assert result.stdout.strip() == "ok", result.stderr
        unwritable = dict(env, UAV_ARENA_BUILD_CACHE=str(cache/"elsewhere"))
        result = subprocess.run([sys.executable, "-c", code], env=unwritable, capture_output=True, text=True)
        assert result.returncode and "UAV_ARENA_BUILD_CACHE" in result.stderr
    finally:
        for path in [cache, *cache.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
