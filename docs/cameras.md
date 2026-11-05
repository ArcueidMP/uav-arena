# Virtual cameras

Cameras turn the arena into an input for vision-based world models and policies, and they film
episodes. They never touch the physics: a `CameraRig` compiles a private copy of the scene that adds
the cameras, a textured floor and a sky, and draws it at the positions the simulator reports. An
environment with cameras flies exactly as one without them, with the same model and the same
snapshots.

```python
from uav_arena import ArenaEnv, CONDITIONS, make_scenario
from uav_arena.cameras import Camera, downward_cameras, onboard_cameras, overview_camera

env = ArenaEnv(make_scenario(0, CONDITIONS[5]), cameras=(*onboard_cameras(), *downward_cameras()))
obs = env.reset()
obs["images"]["uav0_fpv"]       # (64, 64, 3) uint8
env.close()                     # releases the rendering contexts
```

## Presets

| Preset | Names | View | Default size |
|---|---|---|---|
| `onboard_cameras(pitch=20, fovy=90)` | `uav0_fpv` … `uav2_fpv` | forward from each aircraft, pitched down by `pitch` degrees | 64 × 64 |
| `downward_cameras(fovy=90)` | `uav0_down` … `uav2_down` | straight down from each aircraft, at its cable and the payload | 64 × 64 |
| `chase_cameras()` | `uav0_track` … `uav2_track` | Menagerie's chase camera of each aircraft: a fixed offset behind it, level | 64 × 64 |
| `payload_camera()` | `payload_chase` | behind the payload (−x side) and above it, following it with a level orientation | 128 × 128 |
| `overview_camera()` | `overview` | a fixed camera that sees the whole workspace | 640 × 480 |

Every preset takes `width`, `height`, `depth` and `delayed`. At zero yaw, the aircraft's x axis points
from start to goal, so "forward" is the direction of travel.

## Your own cameras

```python
Camera("left_wing", body="uav1", pos=(0.0, 0.05, 0.0), lookat=(1.0, 0.6, -0.3), fovy=70, width=96, height=96)
Camera("tower", body="world", pos=(3.0, -3.0, 2.5), lookat=(0.0, 0.0, 1.0), fovy=45)
Camera("follow", body="payload", mode="trackcom", pos=(0.0, -2.0, 1.0), lookat=(0.0, 0.0, 0.0))
Camera("aim", body="world", pos=(0.0, 3.0, 2.0), mode="targetbody", target="payload")
Camera("down_depth", body="uav0", lookat=(0.0, 0.0, -1.0), up=(1.0, 0.0, 0.0), depth=True)
Camera("chase0", model_camera="uav0_track")
```

- **Mount** (`body`): `world`, `payload`, or `uav0` to `uav2`. `pos`, `lookat` and `up` are given in
  the mount's frame (an aircraft frame has x forward and z up). Without `lookat` the camera looks
  along the mount's +x axis.
- **Mode:** a MuJoCo camera mode. `fixed` moves and turns with the mount. `trackcom` follows the
  mount with a world-fixed orientation. `targetbody` keeps aiming at `target`.
- **Depth:** `depth=True` returns float32 distances in metres, of shape H × W.
- **Model cameras:** `model_camera` names a camera that is already in the model.

## Delays

An observation's images show the same instant as the rest of the observation, the state recorded
`latency` before the decision. A camera with `delayed=False` shows the current scene instead. That
is privileged information, like the oracle's state, so it is useful for diagnostics and videos only.
For delayed images, an environment with cameras keeps the MuJoCo positions of every 10 ms record.

`env.render(camera)` draws any camera on demand, for example for a video. It draws the current
scene, or the delayed one with `delayed=True`; the latter works only in an environment built with
cameras.

## Gymnasium

With `cameras`, `ArenaGymEnv`'s observation is a `Dict`: `state`, the flat state vector, plus one
`Box` per camera keyed by its name (uint8 H × W × 3, or float32 H × W for depth). With
`render_mode="rgb_array"`, `render()` returns an overview image of the current scene, of size
`render_size` (480 × 640 by default).

## Rendering backend and cost

MuJoCo renders with OpenGL. On a machine without a display, set `MUJOCO_GL=egl` (GPU) or
`MUJOCO_GL=osmesa` (CPU, slower) before `mujoco` is first imported. With EGL on a laptop RTX 5070, a
frame takes about 0.3 ms at 64 × 64 and 0.6 ms at 640 × 480. Eleven cameras, including a 640 × 480
overview, take 3.8 ms per observation, against 1.3 ms of physics per decision. On NVIDIA's EGL,
MuJoCo may print `OpenGL error 0x502 in or before mjr_makeContext` when it creates a context; the
images are unaffected. Call `env.close()`, or `close()` on the Gymnasium wrapper, to release the
contexts.

Images are drawn on the GPU and are not bit-reproducible: two renders of the same scene can differ
by one intensity level in a few pixels (6 of 6,912 in a 96 × 72 test). The physics, observations and
rewards stay exact.
