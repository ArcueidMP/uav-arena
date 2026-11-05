"""The reference rows read the payload prior, the nominal mass and public fields from the scenario."""
import dataclasses

import numpy as np
import pytest

from uav_arena import CONDITIONS, ArenaEnv, PayloadSettings, TaskSettings, WindSettings, make_scenario

pytestmark = pytest.mark.plan
pytest.importorskip("torch")
pytest.importorskip("stable_worldmodel")

from uav_arena.planning import MASS_CANDIDATES, AdaptivePhysicsPredictor, PhysicsPredictor, mass_candidates  # noqa: E402


def observe(predictor, scenario):
    env = ArenaEnv(scenario)
    try:
        predictor.observe(env.observation(), scenario.public())
    finally:
        predictor.close()
    return predictor


def test_default_candidates_are_arena_v03():
    assert np.array_equal(mass_candidates((0.010, 0.040)), MASS_CANDIDATES) and len(MASS_CANDIDATES) == 16


def test_physics_row_assumes_the_public_nominal_mass():
    settings = TaskSettings(payload=PayloadSettings(mass_range=(0.02, 0.05), nominal_mass=0.03))
    predictor = observe(PhysicsPredictor(workers=1), make_scenario(0, CONDITIONS[0], settings=settings))
    assert predictor.scenario.payload_mass == 0.03 and predictor.scenario.wind is None


def test_physics_row_uses_what_the_settings_publish():
    settings = TaskSettings(payload=PayloadSettings(hidden=False), wind=WindSettings(speed=1.5, hidden=False))
    scenario = make_scenario(0, CONDITIONS[0], settings=settings)
    predictor = observe(PhysicsPredictor(workers=1), scenario)
    assert predictor.scenario.payload_mass == scenario.payload_mass and predictor.scenario.wind == scenario.wind


def test_adaptive_row_searches_the_public_prior():
    settings = TaskSettings(payload=PayloadSettings(mass_range=(0.02, 0.05)))
    predictor = observe(AdaptivePhysicsPredictor(workers=1), make_scenario(0, CONDITIONS[0], settings=settings))
    assert predictor.candidates[0] == 0.02 and predictor.candidates[-1] == 0.05 and len(predictor.candidates) == 16
    known = TaskSettings(payload=PayloadSettings(hidden=False))
    scenario = make_scenario(0, CONDITIONS[0], settings=known)
    predictor = observe(AdaptivePhysicsPredictor(workers=1), scenario)
    assert list(predictor.candidates) == [scenario.payload_mass]


def test_explicit_predictor_options_still_win():
    scenario = make_scenario(0, CONDITIONS[0])
    assert observe(PhysicsPredictor(workers=1, nominal_mass=0.02), scenario).scenario.payload_mass == 0.02
    custom = observe(AdaptivePhysicsPredictor(workers=1, candidates=(0.01, 0.02)), dataclasses.replace(scenario))
    assert list(custom.candidates) == [0.01, 0.02]
