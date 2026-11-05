"""Virtual cameras: images of the arena scene for vision-based methods and for videos.

A `Camera` is either one of the model's cameras (each aircraft carries Menagerie's `uav{i}_track`
chase camera) or a new one mounted on the world, the payload or an aircraft. Cameras never touch
the physics. A `CameraRig` compiles its own copy of the scene that adds the cameras and a textured
floor, and draws it at given positions. The physics model, its hash and its snapshots are the same
with or without cameras.

In `ArenaEnv(scenario, cameras=...)`, every observation carries `images`, one per camera, drawn
from the same delayed state as the rest of the observation. A camera with `delayed=False` draws
the current scene instead, which is privileged information.

Rendering needs an OpenGL backend. On a machine without a display, set MUJOCO_GL=egl (GPU) or
MUJOCO_GL=osmesa (CPU) before importing mujoco.

    from uav_arena import ArenaEnv, make_scenario, CONDITIONS
    from uav_arena.cameras import onboard_cameras, overview_camera

    env = ArenaEnv(make_scenario(0, CONDITIONS[5]), cameras=(*onboard_cameras(), overview_camera()))
    obs = env.reset()
    obs["images"]["uav0_fpv"].shape     # (64, 64, 3) uint8
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .sim.menagerie import vendor_assets


MODES = ("fixed", "track", "trackcom", "targetbody", "targetbodycom")
MOUNTS = ("world", "payload", "uav0", "uav1", "uav2")


def _triple(values, name):
    vector = tuple(float(x) for x in values)
    if len(vector) != 3 or not all(math.isfinite(x) for x in vector):
        raise ValueError(f"Camera.{name} must be three finite numbers")
    return vector


@dataclass(frozen=True)
class Camera:
    """One virtual camera.

    name: its key in `obs["images"]`. width, height: image size (pixels). depth: render depth in
    metres (float32, H x W) instead of colour (uint8, H x W x 3). delayed: draw the scene the
    decision maker sees (t - latency); False draws the current, privileged scene.
    model_camera: draw an existing model camera (e.g. "uav0_track"); the placement fields are then unused.
    body: the mount ("world", "payload", "uav0".."uav2"). pos: position in the mount's frame (m).
    lookat: a point to look at in the mount's frame; None looks along the mount's +x axis. up: the
    image's up direction in the mount's frame. fovy: vertical field of view (degrees). mode: MuJoCo
    camera mode; "trackcom" follows the mount with a world-fixed orientation, "targetbody" aims at
    `target`.
    """
    name: str
    width: int = 64
    height: int = 64
    depth: bool = False
    delayed: bool = True
    model_camera: str | None = None
    body: str = "world"
    pos: tuple = (0.0, 0.0, 0.0)
    lookat: tuple | None = None
    up: tuple = (0.0, 0.0, 1.0)
    fovy: float = 60.0
    mode: str = "fixed"
    target: str | None = None

    def __post_init__(self):
        if not self.name or self.name == "state":
            raise ValueError("A camera needs a name other than 'state'")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("Camera width and height must be positive")
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "pos", _triple(self.pos, "pos"))
        object.__setattr__(self, "up", _triple(self.up, "up"))
        if self.lookat is not None:
            object.__setattr__(self, "lookat", _triple(self.lookat, "lookat"))
        if self.model_camera is None:
            if self.body not in MOUNTS:
                raise ValueError(f"Camera.body must be one of {MOUNTS}")
            if self.mode not in MODES:
                raise ValueError(f"Camera.mode must be one of {MODES}")
            if self.mode.startswith("targetbody") and not self.target:
                raise ValueError("targetbody cameras need a target body")
            if not 0 < self.fovy < 180:
                raise ValueError("Camera.fovy must lie in (0, 180) degrees")
            self.xyaxes()                   # fail early on a degenerate orientation

    def xyaxes(self):
        """MuJoCo's `xyaxes`: the image's right and up directions in the mount's frame."""
        forward = np.subtract(self.lookat, self.pos) if self.lookat is not None else np.array([1.0, 0.0, 0.0])
        if np.linalg.norm(forward) < 1e-9:
            raise ValueError(f"Camera {self.name!r} looks at its own position")
        forward = forward/np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        if np.linalg.norm(right) < 1e-9:
            raise ValueError(f"Camera {self.name!r}: `up` is parallel to the viewing direction; choose another")
        right = right/np.linalg.norm(right)
        up = np.cross(-forward, right)
        return (*right, *up)

    def space(self):
        """The Gymnasium space of this camera's images."""
        from gymnasium.spaces import Box
        if self.depth:
            return Box(0.0, np.inf, (self.height, self.width), np.float32)
        return Box(0, 255, (self.height, self.width, 3), np.uint8)


# ---------- presets ----------
def onboard_cameras(*, pitch=20.0, fovy=90.0, width=64, height=64, **options):
    """A forward-looking camera on each aircraft, pitched down by `pitch` degrees.

    Aircraft frame: x forward (the start-to-goal direction at zero yaw), z up. Names `uav{i}_fpv`.
    """
    p = math.radians(pitch)
    pos = (0.03, 0.0, -0.01)
    lookat = (pos[0]+math.cos(p), 0.0, pos[2]-math.sin(p))
    return tuple(Camera(name=f"uav{i}_fpv", body=f"uav{i}", pos=pos, lookat=lookat, fovy=fovy,
                        width=width, height=height, **options) for i in range(3))


def downward_cameras(*, fovy=90.0, width=64, height=64, **options):
    """A camera under each aircraft looking straight down at its cable and the payload. Names `uav{i}_down`."""
    return tuple(Camera(name=f"uav{i}_down", body=f"uav{i}", pos=(0.0, 0.0, -0.015), lookat=(0.0, 0.0, -1.0),
                        up=(1.0, 0.0, 0.0), fovy=fovy, width=width, height=height, **options) for i in range(3))


def chase_cameras(*, width=64, height=64, **options):
    """Menagerie's chase camera of each aircraft (`uav{i}_track`): a fixed offset behind it, level."""
    return tuple(Camera(name=f"uav{i}_track", model_camera=f"uav{i}_track", width=width, height=height, **options)
                 for i in range(3))


def payload_camera(*, distance=1.6, height_offset=0.9, fovy=60.0, width=128, height=128, **options):
    """A third-person camera that follows the payload from behind (the -x side) with a level orientation."""
    return Camera(name="payload_chase", body="payload", mode="trackcom", pos=(-distance, 0.0, height_offset),
                  lookat=(0.0, 0.0, 0.3), fovy=fovy, width=width, height=height, **options)


def overview_camera(*, width=640, height=480, fovy=45.0, **options):
    """A fixed world camera that sees the whole workspace (x from -3 to 3 m) from its -y side."""
    return Camera(name="overview", body="world", pos=(0.0, -5.6, 2.8), lookat=(0.0, 0.0, 1.0), fovy=fovy,
                  width=width, height=height, **options)


# ---------- rendering ----------
def render_xml(scene_xml, cameras):
    """The scene XML plus the cameras, a textured floor and a sky: a model for drawing only."""
    root = ET.fromstring(scene_xml)
    world = root.find("worldbody")
    # Track and trackcom cameras freeze their world frame at the model's reference pose, where each
    # aircraft hangs at a hover tilt set by the true payload mass. Level aircraft there keep those frames
    # level and independent of the hidden mass; drawn poses always come from the recorded positions.
    for index in range(3):
        aircraft = world.find(f"body[@name='uav{index}']")
        if aircraft is not None:
            aircraft.set("quat", "1 0 0 0")
    for camera in cameras:
        if camera.model_camera is not None:
            continue
        parent = world if camera.body == "world" else world.find(f".//body[@name='{camera.body}']")
        attributes = dict(name=camera.name, pos=" ".join(format(x, ".17g") for x in camera.pos),
                          fovy=format(camera.fovy, ".17g"), mode=camera.mode)
        if camera.mode.startswith("targetbody"):
            attributes["target"] = camera.target
        else:
            attributes["xyaxes"] = " ".join(format(float(x), ".17g") for x in camera.xyaxes())
        ET.SubElement(parent, "camera", **attributes)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="arena_grid", type="2d", builtin="checker", rgb1=".20 .22 .25",
                  rgb2=".27 .29 .33", width="512", height="512", mark="edge", markrgb=".36 .38 .42")
    ET.SubElement(asset, "material", name="arena_floor", texture="arena_grid", texrepeat="10 10", texuniform="true")
    ET.SubElement(asset, "texture", name="arena_sky", type="skybox", builtin="gradient", rgb1=".55 .68 .80",
                  rgb2=".12 .14 .18", width="256", height="1536")
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.attrib.pop("rgba", None)
        floor.set("material", "arena_floor")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    size = visual.find("global")
    if size is None:
        size = ET.SubElement(visual, "global")
    size.set("offwidth", str(max([640, int(size.get("offwidth", 0)), *(c.width for c in cameras)])))
    size.set("offheight", str(max([480, int(size.get("offheight", 0)), *(c.height for c in cameras)])))
    if visual.find("headlight") is None:       # brighter than the physics scene's defaults, as replays use
        ET.SubElement(visual, "headlight", ambient=".35 .35 .35", diffuse=".6 .6 .6", specular=".1 .1 .1")
    return ET.tostring(root, encoding="unicode")


class CameraRig:
    """Draws cameras on a private copy of the scene at given MuJoCo positions."""

    def __init__(self, scene_xml, cameras):
        self.cameras = tuple(cameras)
        names = [c.name for c in self.cameras]
        if len(set(names)) != len(names):
            raise ValueError("Camera names must be unique")
        self.model = mujoco.MjModel.from_xml_string(render_xml(scene_xml, self.cameras), assets=vendor_assets())
        self.data = mujoco.MjData(self.model)
        self.camera_ids = {}
        for camera in self.cameras:
            name = camera.model_camera or camera.name
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if index < 0:
                raise ValueError(f"The model has no camera {name!r}")
            self.camera_ids[camera.name] = index
        self._renderers = {}

    def _renderer(self, camera):
        key = (camera.height, camera.width, camera.depth)
        if key not in self._renderers:
            renderer = mujoco.Renderer(self.model, camera.height, camera.width)
            if camera.depth:
                renderer.enable_depth_rendering()
            self._renderers[key] = renderer
        return self._renderers[key]

    def _set_scene(self, qpos, mocap_pos):
        self.data.qpos[:] = qpos
        if self.model.nmocap:
            self.data.mocap_pos[:] = mocap_pos
        mujoco.mj_fwdPosition(self.model, self.data)

    def _draw(self, camera):
        renderer = self._renderer(camera)
        renderer.update_scene(self.data, camera=self.camera_ids[camera.name])
        image = renderer.render()
        return image.astype(np.float32) if camera.depth else image

    def render_scene(self, qpos, mocap_pos, names=None):
        """Every camera (or those in `names`) at one scene: MuJoCo positions and mocap positions."""
        self._set_scene(qpos, mocap_pos)
        return {c.name: self._draw(c) for c in self.cameras if names is None or c.name in names}

    def render(self, *, delayed, current):
        """Images for an observation: delayed cameras from the `delayed` record, the others from `current`.
        Records are `ArenaEnv` log entries kept with `qpos` and `mocap_pos`."""
        images = {}
        for record, group in ((delayed, [c for c in self.cameras if c.delayed]),
                              (current, [c for c in self.cameras if not c.delayed])):
            if group:
                self._set_scene(record["qpos"], record["mocap_pos"])
                images.update({c.name: self._draw(c) for c in group})
        return {c.name: images[c.name] for c in self.cameras}

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers = {}
