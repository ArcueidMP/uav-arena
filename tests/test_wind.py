"""Wind fields and their effect on the team."""
import numpy as np
import pytest

from uav_arena import (CONDITIONS, ArenaEnv, Gust, GustSettings, TaskSettings, TurbulenceSettings, Wind, WindSettings,
                       make_scenario)
from uav_arena.env import quaternion_matrices

UNSTEADY = TaskSettings(wind=WindSettings(speed=2.0, gusts=GustSettings(rate=1.0), turbulence=TurbulenceSettings()))


def hover(env, seconds):
    """Hold the settled hover setpoints; payload displacement from its start, per decision."""
    target = env.observation()["recent_setpoints"][-1].copy()
    start = env.true_state()["payload_pos"].copy()
    moved = []
    for _ in range(round(seconds/0.1)):
        _, _, done, info = env.step(target)
        moved.append(float(np.linalg.norm(env.true_state()["payload_pos"]-start)))
        assert not done, info
    return np.array(moved)


def test_gust_profile():
    gust = Gust(onset=1.0, duration=2.0, peak=(3.0, -1.0, 0.5))
    assert np.array_equal(gust.velocity(0.9), np.zeros(3)) and np.array_equal(gust.velocity(3.1), np.zeros(3))
    assert np.allclose(gust.velocity(2.0), (3.0, -1.0, 0.5))
    assert np.allclose(gust.velocity(1.5), gust.velocity(2.5))
    wind = Wind(mean=(1.0, 0.0, 0.0), gusts=(gust,))
    assert np.allclose(wind.velocity(2.0), (4.0, -1.0, 0.5)) and not wind.steady


def test_turbulence_statistics():
    turbulence = TurbulenceSettings(intensity=(0.5, 0.3, 0.1), time_scale=0.8).draw(np.random.default_rng(1))
    assert np.allclose(turbulence.std, (0.5, 0.3, 0.1))
    samples = np.stack([turbulence.velocity(t) for t in np.arange(0, 600, 0.05)])
    assert np.allclose(samples.mean(0), 0, atol=0.05)
    assert np.allclose(samples.std(0), (0.5, 0.3, 0.1), rtol=0.2)


def test_gust_draws_follow_settings():
    settings = GustSettings(rate=2.0, peak=(1.0, 1.5), duration=(0.4, 0.6), heading=(80, 100), vertical=0.2)
    gusts = settings.draw(np.random.default_rng(0), 50.0)
    assert 60 < len(gusts) < 140                              # Poisson, mean 100
    for g in gusts:
        horizontal = np.hypot(g.peak[0], g.peak[1])
        assert 1.0 <= horizontal <= 1.5 and 0.4 <= g.duration <= 0.6 and abs(g.peak[2]) <= 0.2*horizontal+1e-12
        assert 80 <= np.degrees(np.arctan2(g.peak[1], g.peak[0])) <= 100


def test_zero_wind_is_still_air():
    scenario = make_scenario(3, CONDITIONS[5])
    still, zero = ArenaEnv(scenario), ArenaEnv(make_scenario(3, CONDITIONS[5], wind=Wind()))
    rng = np.random.default_rng(0)
    for _ in range(20):
        step = rng.uniform(-0.05, 0.05, (3, 3))
        still.step(still.observation()["recent_setpoints"][-1]+step)
        zero.step(zero.observation()["recent_setpoints"][-1]+step)
    assert np.array_equal(still.data.qpos, zero.data.qpos) and np.array_equal(still.data.qvel, zero.data.qvel)


def test_steady_wind_tilts_the_team_into_it_and_settles_in_the_warmup():
    env = ArenaEnv(make_scenario(0, CONDITIONS[0], wind=Wind(mean=(3.0, 0.0, 0.0))))
    thrust_axes = quaternion_matrices(env.true_state()["uav_quat"])[:, :, 2]
    calm = quaternion_matrices(ArenaEnv(make_scenario(0, CONDITIONS[0])).true_state()["uav_quat"])[:, :, 2]
    assert thrust_axes[:, 0].mean() < calm[:, 0].mean()-0.05          # leaning upwind (-x)
    assert np.linalg.norm(env.true_state()["payload_vel"]) < 0.01      # settled at t = 0
    assert hover(env, 3.0).max() < 0.01


def test_a_gust_pushes_the_payload_and_the_team_recovers():
    gust = Gust(onset=0.5, duration=1.0, peak=(0.0, 3.0, 0.0))
    moved = hover(ArenaEnv(make_scenario(0, CONDITIONS[0], wind=Wind(gusts=(gust,)))), 8.0)
    assert moved.max() > 0.04 and moved[-1] < 0.5*moved.max()


def test_true_rollouts_stay_exact_in_unsteady_wind():
    """The oracle forks the true simulator: in gusts and turbulence its rollouts equal the episode."""
    scenario = make_scenario(4, CONDITIONS[1], settings=UNSTEADY)
    env = ArenaEnv(scenario)
    obs, rng = env.observation(), np.random.default_rng(0)
    for _ in range(5):
        obs, *_ = env.step(obs["recent_setpoints"][-1]+rng.uniform(-0.05, 0.05, (3, 3)))
    fork = env.physics_state()
    sequence = obs["recent_setpoints"][-1]+np.cumsum(rng.uniform(-0.05, 0.05, (10, 3, 3)), axis=0)
    worker = ArenaEnv(scenario)
    worker.set_physics_state(fork)
    rollout = worker.simulate(sequence)
    flown = []
    for target in sequence:
        env.step(target)
        flown.append(env.true_state()["payload_pos"].copy())
    assert np.array_equal(np.stack(flown), rollout["payload"])


def test_snapshots_replay_unsteady_wind_exactly():
    env = ArenaEnv(make_scenario(8, CONDITIONS[2], settings=UNSTEADY))
    target = env.observation()["recent_setpoints"][-1]
    for _ in range(3):
        env.step(target)
    snapshot = env.capture()
    first = [env.step(target+0.01*k)[0]["payload_pos"] for k in range(5)]
    env.restore(snapshot)
    again = [env.step(target+0.01*k)[0]["payload_pos"] for k in range(5)]
    assert np.array_equal(np.stack(first), np.stack(again))


def test_wind_never_leaks_between_environments():
    windy = ArenaEnv(make_scenario(0, CONDITIONS[0], wind=Wind(mean=(2.0, 1.0, 0.0))))
    calm = ArenaEnv(make_scenario(0, CONDITIONS[0]))
    assert np.allclose(windy.model.opt.wind, (2.0, 1.0, 0.0)) and np.array_equal(calm.model.opt.wind, np.zeros(3))


def test_unsteady_wind_tracks_episode_time():
    scenario = make_scenario(4, CONDITIONS[0], settings=UNSTEADY)
    env = ArenaEnv(scenario)
    env.step(env.observation()["recent_setpoints"][-1])
    # The wind of the last physics step was evaluated at its start: one step before the clock now reads.
    assert np.allclose(env.model.opt.wind, scenario.wind.velocity(env.data.time-env.config.dt))
    with pytest.raises(ValueError):
        Gust(onset=0.0, duration=0.0, peak=(1, 0, 0))
