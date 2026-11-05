"""The Gymnasium wrapper with onboard camera observations, driven by random actions.

    MUJOCO_GL=egl python examples/gym_random.py
"""
from uav_arena import TaskSettings, WindSettings
from uav_arena.cameras import onboard_cameras
from uav_arena.gym_env import ArenaGymEnv

if __name__ == "__main__":
    env = ArenaGymEnv(settings=TaskSettings(wind=WindSettings(speed=(0.0, 2.0))),
                      cameras=onboard_cameras(width=64, height=64))
    print("observation space:", env.observation_space)
    obs, info = env.reset(seed=0)
    total, steps = 0.0, 0
    terminated = truncated = False
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total, steps = total+reward, steps+1
    print(f"layout {info['seed']} {info['condition']}: {info['outcome']} after {steps} steps, return {total:.1f}")
    print({key: value.shape for key, value in obs.items()})
    env.close()
