import ctypes.util
import os
import sys

# Headless rendering: MuJoCo picks its OpenGL backend when it is first imported. Choose EGL or OSMesa only
# when its library is present; otherwise leave MUJOCO_GL alone, so that MuJoCo still imports and only the
# rendering tests skip.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
    for backend, library in (("egl", "EGL"), ("osmesa", "OSMesa")):
        if ctypes.util.find_library(library):
            os.environ["MUJOCO_GL"] = backend
            break

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
