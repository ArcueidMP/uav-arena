"""UAV Arena: a world-model arena for three-UAV cooperative transport of a cable-suspended payload.

    from uav_arena import ArenaEnv, CONDITIONS, TaskSettings, WindSettings, make_scenario

    scenario = make_scenario(0, CONDITIONS[0], settings=TaskSettings(wind=WindSettings(speed=1.5)))
    env = ArenaEnv(scenario)
    obs = env.reset()
    obs, reward, done, info = env.step(obs["recent_setpoints"][-1])

The default settings reproduce the arena-v0.3 task bit for bit. See README.md and docs/settings.md.
"""
from .env import ACTION_LIMIT, DECISION_DT, N_UAV, NOMINAL_PAYLOAD_MASS, ArenaEnv
from .scenarios import CONDITIONS, LATENCIES_MS, OBSTACLE_SETTINGS, SUITE_SEEDS, Condition, Scenario, make_scenario, suite
from .settings import (GustSettings, ObstacleSettings, PayloadSettings, TaskSettings, TurbulenceSettings, WindSettings,
                       hover_rotor_fraction, load_settings, max_payload_mass)
from .wind import Gust, Turbulence, Wind

__version__ = "0.1.0"

__all__ = ["ACTION_LIMIT", "CONDITIONS", "DECISION_DT", "LATENCIES_MS", "N_UAV", "NOMINAL_PAYLOAD_MASS",
           "OBSTACLE_SETTINGS", "SUITE_SEEDS", "ArenaEnv", "Condition", "Gust", "GustSettings", "ObstacleSettings",
           "PayloadSettings", "Scenario", "TaskSettings", "Turbulence", "TurbulenceSettings", "Wind", "WindSettings",
           "hover_rotor_fraction", "load_settings", "make_scenario", "max_payload_mass", "suite", "__version__"]
