"""Task settings: defaults, validation, draws, information hiding and serialization."""
import json

import numpy as np
import pytest

from uav_arena import (CONDITIONS, NOMINAL_PAYLOAD_MASS, ArenaEnv, GustSettings, ObstacleSettings, PayloadSettings,
                       Scenario, TaskSettings, TurbulenceSettings, Wind, WindSettings, hover_rotor_fraction,
                       load_settings, make_scenario, max_payload_mass)
from uav_arena.scenarios import PAYLOAD_MASS_RANGE

WINDY = TaskSettings(wind=WindSettings(speed=(0.5, 2.0), heading=(150, 210), gusts=GustSettings(rate=0.5),
                                       turbulence=TurbulenceSettings(intensity=(0.3, 0.3, 0.1))))


def test_defaults_are_arena_v03():
    settings = TaskSettings()
    assert settings.is_default
    assert settings.payload.prior == PAYLOAD_MASS_RANGE and settings.payload.nominal == NOMINAL_PAYLOAD_MASS
    assert settings.wind.still and settings.wind.draw(0, 20.0) is None
    for seed in (0, 7, 1003):
        for condition in CONDITIONS[::3]:
            default, explicit = make_scenario(seed, condition), make_scenario(seed, condition, settings=settings)
            assert default == explicit and default.to_dict() == explicit.to_dict()
            assert set(default.to_dict()) == {"seed", "condition", "start", "goal", "pillars", "movers",
                                              "payload_mass", "time_limit", "goal_tolerance", "meta"}


def test_payload_mass_draws_fixed_values_and_nominal():
    family = PayloadSettings(mass_range=(0.02, 0.05))
    masses = [make_scenario(s, CONDITIONS[0], settings=TaskSettings(payload=family)).payload_mass for s in range(40)]
    assert 0.02 <= min(masses) and max(masses) <= 0.05 and np.ptp(masses) > 0.02
    assert family.nominal == pytest.approx(0.035)
    fixed = make_scenario(3, CONDITIONS[0], settings=TaskSettings(payload=PayloadSettings(mass=0.05)))
    assert fixed.payload_mass == 0.05 and fixed.nominal_mass == 0.05 and fixed.payload_prior == (0.05, 0.05)
    assert make_scenario(3, CONDITIONS[0], payload_mass=0.012).payload_mass == 0.012


def test_payload_limits_follow_rotor_thrust():
    assert hover_rotor_fraction(0.025) == pytest.approx(0.545, abs=2e-3)      # measured hover: 54.5 %
    assert hover_rotor_fraction(0.060) == pytest.approx(0.734, abs=2e-3)
    assert 0.060 < max_payload_mass(0.75) < 0.065
    with pytest.raises(ValueError, match="rotor thrust"):
        PayloadSettings(mass_range=(0.02, 0.08))
    PayloadSettings(mass_range=(0.02, 0.08), max_hover_fraction=0.9)       # accepted with less headroom
    with pytest.raises(ValueError):
        PayloadSettings(mass_range=(0.04, 0.02))


def test_settings_validation():
    for bad in (dict(speed=-1.0), dict(speed=(2.0, 1.0)), dict(heading=(10, 0))):
        with pytest.raises(ValueError):
            WindSettings(**bad)
    with pytest.raises(ValueError):
        GustSettings(rate=0)
    with pytest.raises(ValueError):
        TurbulenceSettings(intensity=(0.1, 0.1))
    with pytest.raises(ValueError):
        TaskSettings(time_limit=0)
    with pytest.raises(ValueError):
        TaskSettings.from_dict(dict(windspeed=3))


def test_draws_are_paired_across_conditions_and_seeded():
    a = [make_scenario(5, c, settings=WINDY) for c in CONDITIONS]
    assert len({(s.payload_mass, s.wind) for s in a}) == 1
    assert make_scenario(5, CONDITIONS[0], settings=WINDY).wind == a[0].wind
    assert make_scenario(6, CONDITIONS[0], settings=WINDY).wind != a[0].wind
    speed = np.linalg.norm(a[0].wind.mean)
    heading = np.degrees(np.arctan2(a[0].wind.mean[1], a[0].wind.mean[0])) % 360
    assert 0.5 <= speed <= 2.0 and 150 <= heading <= 210 and a[0].wind.mean[2] == 0.0


def test_changing_one_wind_part_keeps_the_others():
    base = make_scenario(9, CONDITIONS[0], settings=WINDY).wind
    no_gusts = TaskSettings(wind=WindSettings(speed=(0.5, 2.0), heading=(150, 210),
                                              turbulence=TurbulenceSettings(intensity=(0.3, 0.3, 0.1))))
    other = make_scenario(9, CONDITIONS[0], settings=no_gusts).wind
    assert other.mean == base.mean and other.turbulence == base.turbulence and not other.gusts


def test_public_hides_mass_and_wind_unless_settings_publish_them():
    hidden = make_scenario(2, CONDITIONS[0], settings=WINDY)
    assert hidden.public().payload_mass is None and hidden.public().wind is None
    known = TaskSettings(payload=PayloadSettings(hidden=False), wind=WindSettings(speed=1.0, hidden=False))
    public = make_scenario(2, CONDITIONS[0], settings=known).public()
    assert public.payload_mass is not None and public.wind is not None


def test_json_round_trips(tmp_path):
    path = tmp_path/"settings.json"
    path.write_text(json.dumps(WINDY.to_dict()))
    assert load_settings(path) == WINDY and load_settings(None) == TaskSettings()
    scenario = make_scenario(4, CONDITIONS[6], settings=WINDY)
    again = Scenario.from_dict(json.loads(json.dumps(scenario.to_dict())))
    assert again == scenario
    t = np.linspace(0, 20, 57)
    assert np.array_equal(again.wind.velocities(t), scenario.wind.velocities(t))
    public = Scenario.from_dict(json.loads(json.dumps(scenario.public().to_dict())))
    assert public == scenario.public()


def test_obstacle_settings():
    settings = TaskSettings(obstacles=ObstacleSettings(n_pillars=6, n_movers=3, mover_speed=(0.5, 0.6)))
    scenario = make_scenario(1, CONDITIONS[4], settings=settings)
    assert len(scenario.pillars) == 6 and len(scenario.movers) == 3
    assert all(0.5 <= m.speed <= 0.6 for m in scenario.movers)
    assert len(make_scenario(1, CONDITIONS[0], settings=settings).movers) == 0


def test_infeasible_obstacles_fail_fast_with_a_clear_error():
    with pytest.raises(ValueError, match="cannot fit"):
        ObstacleSettings(n_pillars=2, pillar_radius=(2, 2))
    with pytest.raises(ValueError, match="cannot fit"):             # keyword overrides are validated too
        make_scenario(0, CONDITIONS[0], n_pillars=2, pillar_radius=(2, 2))
    dense = TaskSettings(obstacles=ObstacleSettings(n_pillars=41, pillar_radius=(0.1, 0.1)))
    with pytest.raises(ValueError, match="could not place"):       # within the bound, beyond rejection sampling
        make_scenario(0, CONDITIONS[0], settings=dense)
    with pytest.raises(RuntimeError, match="corridor"):            # fits, but blocks the way
        make_scenario(0, CONDITIONS[0], n_pillars=1, pillar_radius=(1.5, 1.5))


def test_episode_settings_reach_the_environment():
    settings = TaskSettings(time_limit=3.0, goal_tolerance=0.3)
    env = ArenaEnv(make_scenario(0, CONDITIONS[0], settings=settings))
    obs = env.observation()
    done = False
    while not done:
        obs, _, done, info = env.step(obs["recent_setpoints"][-1])
    assert info["outcome"] == "timeout" and info["time"] == pytest.approx(3.0)


def test_wind_values_are_validated():
    with pytest.raises(ValueError):
        Wind(mean=(1.0, 0.0))
    with pytest.raises(TypeError):
        Scenario(seed=0, condition=CONDITIONS[0], start=(0, 0, 1), goal=(1, 0, 1), pillars=(), wind=(1, 0, 0))
