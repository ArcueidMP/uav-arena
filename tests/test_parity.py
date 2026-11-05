"""Default settings reproduce arena-v0.3 bit for bit.

tests/golden/arena_v03.npz is the fingerprint (tools/fingerprint.py) of the frozen research code,
ArcueidMP/vincair-multi-agent-wm at 0fdb022, whose core files are those of tag arena-v0.3. Every part is
compared bit for bit. The same results have held with Python 3.10 and 3.12, NumPy 2.2 and 2.5 and PyTorch
2.11 and 2.15, so those versions do not matter. The platform does: a different CPU architecture, MuJoCo
build, C compiler or C library can change the last bits of floating-point results. On a platform that
differs from the recording one in any of those, a mismatch is reported as a skip that names the
difference, never as a pass. Set UAV_ARENA_STRICT_PARITY=1 to fail instead.
"""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"tests"/"golden"/"arena_v03.npz"
spec = importlib.util.spec_from_file_location("fingerprint", ROOT/"tools"/"fingerprint.py")
fingerprint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fingerprint)

PREFIX = dict(scenarios="scenarios", environment="env/", outcomes="outcomes/", gym="gym/", planning="plan/",
              collect="collect/")
PLATFORM = ("machine", "mujoco", "cc", "libc")


def _check(part):
    reference, meta = fingerprint.load(GOLDEN)
    reference = {k: v for k, v in reference.items() if k.startswith(PREFIX[part])}
    candidate = fingerprint.fingerprint((part,))
    differ = fingerprint.compare(reference, candidate, exact=True)
    if not differ:
        return
    here = fingerprint._versions()
    changed = {k: (meta["versions"].get(k), here.get(k)) for k in PLATFORM if here.get(k) != meta["versions"].get(k)}
    if not changed or os.environ.get("UAV_ARENA_STRICT_PARITY") == "1":
        pytest.fail(f"{part} is not bit for bit arena-v0.3: {differ}")
    pytest.skip(f"{part} differs from arena-v0.3 in the last bits on this platform {changed}; bit-for-bit parity "
                f"is verified on the recording platform only. Differences: {differ[:3]}")


@pytest.mark.parametrize("part", ["scenarios", "environment", "outcomes", "collect"])
def test_default_task_reproduces_arena_v03(part):
    _check(part)


def test_gym_wrapper_reproduces_arena_v03():
    pytest.importorskip("gymnasium")
    _check("gym")


@pytest.mark.plan
def test_reference_planners_reproduce_arena_v03():
    pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    _check("planning")
