import os
import sys

# Headless rendering: MuJoCo picks its OpenGL backend when it is first imported.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def can_render():
    """Skip a test when no OpenGL backend can create a rendering context."""
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom size='.1'/></worldbody></mujoco>")
        with mujoco.Renderer(model, 8, 8) as renderer:
            renderer.render()
    except Exception as error:      # no display, no EGL/OSMesa
        pytest.skip(f"no OpenGL backend for rendering ({error}); set MUJOCO_GL=egl or osmesa")
    return True
