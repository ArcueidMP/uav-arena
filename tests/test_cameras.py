"""Virtual cameras: images, delays, presets, and no effect on the physics."""
import numpy as np
import pytest

from uav_arena import CONDITIONS, ArenaEnv, make_scenario
from uav_arena.cameras import (Camera, CameraRig, chase_cameras, downward_cameras, onboard_cameras, overview_camera,
                               payload_camera)

pytestmark = pytest.mark.render


def assert_same_image(a, b):
    """GPU rasterization can change a few pixels by one level between renders of the same scene."""
    difference = np.abs(a.astype(int)-b.astype(int))
    assert difference.max() <= 3 and (difference.max(-1) > 0).mean() < 0.01


def presets():
    return (*onboard_cameras(), *downward_cameras(), *chase_cameras(), payload_camera(), overview_camera())


def test_presets_are_valid_and_distinct():
    cameras = presets()
    assert len({c.name for c in cameras}) == len(cameras)
    for camera in cameras:
        if camera.model_camera is None:
            right, up = np.array(camera.xyaxes()[:3]), np.array(camera.xyaxes()[3:])
            assert np.isclose(right@up, 0) and np.isclose(np.linalg.norm(right), 1) and np.isclose(np.linalg.norm(up), 1)
    with pytest.raises(ValueError):
        Camera("bad", body="uav7")
    with pytest.raises(ValueError):
        Camera("bad", pos=(0, 0, 1), lookat=(0, 0, 0))        # looking along `up`


def test_observation_images(can_render):
    cameras = (*onboard_cameras(width=32, height=24), Camera("depth", body="uav1", lookat=(0.0, 0.0, -1.0),
                                                             up=(1.0, 0.0, 0.0), depth=True, width=16, height=16))
    env = ArenaEnv(make_scenario(0, CONDITIONS[5]), cameras=cameras)
    try:
        images = env.observation()["images"]
        assert set(images) == {c.name for c in cameras}
        assert images["uav0_fpv"].shape == (24, 32, 3) and images["uav0_fpv"].dtype == np.uint8
        assert images["uav0_fpv"].std() > 5                     # not a blank frame
        depth = images["depth"]
        assert depth.shape == (16, 16) and depth.dtype == np.float32
        # The camera looks straight down from about 1.3 m: the payload hangs 0.55 m below, the floor further.
        assert 0.3 < depth.min() < 1.0 and depth.max() < 3.0
    finally:
        env.close()


def test_cameras_leave_the_physics_unchanged(can_render):
    scenario = make_scenario(2, CONDITIONS[6])
    plain, filmed = ArenaEnv(scenario), ArenaEnv(scenario, cameras=onboard_cameras(width=16, height=16))
    try:
        rng = np.random.default_rng(1)
        for _ in range(10):
            step = rng.uniform(-0.05, 0.05, (3, 3))
            plain.step(plain.observation()["recent_setpoints"][-1]+step)
            filmed.step(filmed.observation()["recent_setpoints"][-1]+step)
        assert np.array_equal(plain.data.qpos, filmed.data.qpos) and plain.plant.model_hash == filmed.plant.model_hash
    finally:
        filmed.close()


def test_delayed_images_show_the_delayed_scene(can_render):
    """At 200 ms latency a delayed camera sees the scene of the state observation, not the current one."""
    camera = overview_camera(width=96, height=72)
    current = Camera("now", body="world", pos=camera.pos, lookat=camera.lookat, fovy=camera.fovy, width=96, height=72,
                     delayed=False)
    env = ArenaEnv(make_scenario(1, CONDITIONS[7]), cameras=(camera, current))
    try:
        obs = env.observation()
        for _ in range(6):
            obs, *_ = env.step(obs["recent_setpoints"][-1]+np.array([0.1, 0.0, 0.0]))
        assert not np.array_equal(obs["images"]["overview"], obs["images"]["now"])
        record = env.log[-1-env.delay_records]
        assert record["time"] == pytest.approx(obs["time"])
        rig = CameraRig(_scene(env), (camera,))
        try:
            assert_same_image(rig.render_scene(record["qpos"], record["mocap_pos"])["overview"], obs["images"]["overview"])
        finally:
            rig.close()
        assert_same_image(env.render(current), obs["images"]["now"])
    finally:
        env.close()


def _scene(env):
    from uav_arena.env import scene_xml
    return scene_xml(env.config, env.scenario)


def test_gym_wrapper_with_cameras_and_render(can_render):
    pytest.importorskip("gymnasium")
    from uav_arena.gym_env import ArenaGymEnv
    env = ArenaGymEnv(cameras=onboard_cameras(width=32, height=32), render_mode="rgb_array", render_size=(120, 160))
    try:
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        frame = env.render()
        assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
    finally:
        env.close()
