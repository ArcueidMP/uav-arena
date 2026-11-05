"""Recollected training episodes match the published dataset manifests (uav_arena/manifests/).

The manifests list each episode's content hash for the standard arena-v0.3 datasets, collected with
the frozen research code. Matching them shows that scenario draws, physics, onboard control, the
behavior policies and the episode format are all unchanged at the default settings.
"""
import json
from pathlib import Path

import pytest

from uav_arena.collect import REFERENCES, collect_episode, content_sha256, verify


def _reference(name):
    manifest = json.loads((REFERENCES/f"{name}.json").read_text())
    return manifest["policy"], {e["seed"]: e["content_sha256"] for e in manifest["episodes"]}


@pytest.mark.parametrize("name,seed", [("explore_v2_train", 200_000), ("explore_v2_train", 201_999),
                                       ("explore_v2_val", 299_000), ("explore_v1_train", 200_000)])
def test_episode_matches_published_manifest(name, seed):
    import platform
    if platform.machine() not in ("x86_64", "AMD64"):
        pytest.skip("the published hashes were recorded on x86-64")
    policy, reference = _reference(name)
    assert content_sha256(collect_episode(seed, policy)) == reference[seed]


def test_verify_reports_matches_and_mismatches(tmp_path):
    import numpy as np
    episode = collect_episode(200_000)
    np.savez_compressed(Path(tmp_path)/"episode_200000.npz", **episode)
    episode["payload_mass"] = np.float64(episode["payload_mass"]+1e-12)
    np.savez_compressed(Path(tmp_path)/"episode_200001.npz", **episode)
    result = verify(tmp_path, processes=1)
    assert result["match"] == [200_000] and result["differ"] == [200_001]
