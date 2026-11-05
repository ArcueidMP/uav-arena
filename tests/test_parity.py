"""Default settings reproduce arena-v0.3.

tests/golden/arena_v03.npz is the fingerprint (tools/fingerprint.py) of the frozen research code,
ArcueidMP/vincair-multi-agent-wm at 0fdb022, whose core files are those of tag arena-v0.3. Every part
is first compared bit for bit, which has held with Python 3.10 and 3.12, NumPy 2.2 and 2.5 and
PyTorch 2.11 and 2.15 on Linux x86-64. Only on a platform that differs from the recording one (NumPy,
MuJoCo, compiler or CPU architecture) may floating-point arrays fall back to a tight tolerance;
scenarios and outcomes must still match exactly, and dataset hashes are then not checked.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT/"tests"/"golden"/"arena_v03.npz"
spec = importlib.util.spec_from_file_location("fingerprint", ROOT/"tools"/"fingerprint.py")
fingerprint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fingerprint)

PREFIX = dict(scenarios="scenarios", environment="env/", gym="gym/", planning="plan/", collect="collect/")
# Tolerances off the recording platform. Planning goes through float32 torch costs, hence looser.
TOLERANCE = dict(environment=(1e-6, 1e-8), gym=(1e-5, 1e-6), planning=(1e-4, 1e-6))


def _check(part, keys=("numpy", "mujoco", "machine", "cc")):
    reference, meta = fingerprint.load(GOLDEN)
    reference = {k: v for k, v in reference.items() if k.startswith(PREFIX[part])}
    candidate = fingerprint.fingerprint((part,))
    differ = fingerprint.compare(reference, candidate, exact=True)
    if not differ:
        return
    here = fingerprint._versions()
    if all(here.get(k) == meta["versions"].get(k) for k in keys):
        pytest.fail(f"{part} differs from arena-v0.3 on the recording platform: {differ}")
    if part == "collect":
        pytest.skip("dataset hashes are bitwise, and this platform differs from the recording one")
    rtol, atol = TOLERANCE.get(part, (0.0, 0.0))
    differ = fingerprint.compare(reference, candidate, exact=False, rtol=rtol, atol=atol)
    assert not differ, f"{part} differs from arena-v0.3 beyond the cross-platform tolerance: {differ}"


@pytest.mark.parametrize("part", ["scenarios", "environment", "collect"])
def test_default_task_reproduces_arena_v03(part):
    _check(part)


def test_gym_wrapper_reproduces_arena_v03():
    pytest.importorskip("gymnasium")
    _check("gym")


@pytest.mark.plan
def test_reference_planners_reproduce_arena_v03():
    pytest.importorskip("torch")
    pytest.importorskip("stable_worldmodel")
    _check("planning", keys=("numpy", "mujoco", "machine", "cc", "torch"))
