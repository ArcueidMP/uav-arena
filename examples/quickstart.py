"""Fly the arena by hand: the default (arena-v0.3) task, then a windy family with a heavier payload.

    python examples/quickstart.py

The "policy" here only drifts the formation toward the goal along a minimum-jerk path and ignores
obstacles; it shows the interface, not a solution. Plug a planner or a learned policy in its place.
"""
import numpy as np

from uav_arena import (CONDITIONS, ArenaEnv, GustSettings, PayloadSettings, TaskSettings, WindSettings, make_scenario)


def fly(scenario, duration=6.0):
    env = ArenaEnv(scenario)
    obs = env.reset()
    start, shift = obs["recent_setpoints"][-1], obs["goal"]-obs["payload_pos"]
    while not env.done:
        s = min(1.0, (obs["decision_time"]+0.1)/duration)
        obs, reward, done, info = env.step(start+shift*(10*s**3-15*s**4+6*s**5))
    return info


def describe(scenario):
    wind = "still air" if scenario.wind is None else (
        f"mean wind {np.linalg.norm(scenario.wind.mean):.1f} m/s, {len(scenario.wind.gusts)} gusts")
    return f"seed {scenario.seed}, {scenario.condition.name}, payload {1000*scenario.payload_mass:.1f} g, {wind}"


if __name__ == "__main__":
    print("arena-v0.3 task:")
    for seed in range(3):
        scenario = make_scenario(seed, CONDITIONS[0])
        info = fly(scenario)
        print(f"  {describe(scenario)} -> {info['outcome']} at {info['time']:.1f} s")

    windy = TaskSettings(payload=PayloadSettings(mass_range=(0.03, 0.06)),
                         wind=WindSettings(speed=(1.0, 2.5), heading=180.0, gusts=GustSettings(rate=0.3)))
    print("A windy family with heavier payloads (headwind, gusts):")
    for seed in range(3):
        scenario = make_scenario(seed, CONDITIONS[0], settings=windy)
        info = fly(scenario)
        print(f"  {describe(scenario)} -> {info['outcome']} at {info['time']:.1f} s")
