# Adding a world model

A participant is a predictor: given candidate setpoint sequences, it predicts the team's motion.
The shared planner (MPPI from stable-worldmodel 0.1.1, with 512 samples, a 15-decision horizon and
the shared cost in `costs.py`) turns its predictions into decisions. The predictor is the only thing
that differs between rows. The reference rows (the oracle U, the physics row and the adaptive row)
are predictors under the same rules.

## The predictor interface

```python
class MyWorldModel:
    def reset(self): ...                      # optional: a new episode starts; clear episode state
    def observe(self, obs, layout): ...       # optional: every decision, before planning
    def __call__(self, info, setpoints):      # required
        # setpoints: torch (..., H, 3, 3), world-frame segment end points for the next H decisions,
        #            the first one starting at obs["decision_time"]
        # returns {"uav": (..., H, 3, 3), "payload": (..., H, 3), "payload_vel": (..., H, 3)}:
        #            the team state at the end of each of those decisions, on the device and in the
        #            dtype of `setpoints`
        ...
    def close(self): ...                      # optional: release workers or GPU memory
```

- `obs` is `ArenaEnv.observation()`, recorded `obs["latency"]` before the decision. The predictor
  bridges the delay from `obs["time"]` to `obs["decision_time"]` itself, with
  `obs["recent_setpoints"]` and `obs["recent_setpoint_times"]`: the last four setpoints sent and
  their start times. Each setpoint ramps linearly from the previous one over 100 ms.
- `layout` is `Scenario.public()`: start, goal, pillars, movers, condition, the nominal payload
  mass and the mass prior. Hidden fields (by default the payload mass and the wind) are None.
- `info` holds the current observation as batched tensors (`planning.info_from_observation`).
  MPPI calls the predictor with leading shape (1, 512), and the evaluation harness with other
  shapes, so a predictor must accept any leading shape.
- To use the augmented-Lagrangian solver (`--solver lagrangian`), the output must be differentiable
  with respect to `setpoints`.
- **Stochastic predictors**, whose output is random for a fixed input, set `stochastic = True`.
  They draw all their randomness from torch's global generator and reseed it in `reset()` from a
  fixed seed. Each episode then replays exactly, whatever ran before it. A predictor without
  `stochastic = True` must return identical outputs for identical calls.

`uav_arena/participants/example/` is a complete small participant: an MLP trained on collected data
(`train.py`) with its predictor (`model.py`). Run any predictor in closed loop, or on collected
episodes:

```bash
python -m uav_arena.run --predictor my_package.my_model:make \
  --predictor-options '{"checkpoint": "runs/my_model/ckpt.pt"}' --suite dev --seeds 0:5 \
  --out runs/results/my_model_dev.json
python -m uav_arena.openloop --predictor my_package.my_model:make \
  --predictor-options '{"checkpoint": "runs/my_model/ckpt.pt"}' --data runs/datasets/explore_v2_val \
  --out runs/results/my_model_openloop_val.json
```

Both take `--settings file.json` to evaluate on a tuned scenario family ([settings.md](settings.md)).

## Information rules

At decision time a predictor may use:
- `obs`, `layout` and `info`;
- anything it computed from earlier observations of the same episode.

At decision time a predictor may not use:
- hidden fields (by default the payload mass and the wind), by any route: seed lookups, data files,
  or the environment object;
- the true undelayed state, the onboard motor and controller states, or simulator internals;
- future obstacle paths. The shared cost scores obstacles with constant-velocity forecasts, so a
  predictor only predicts the team;
- anything carried over from another episode.

In training:
- Use only training seeds, through `collect.py` or your own simulation of those seeds' scenarios.
- The payload mass and the wind stored in data files are for diagnostics. They are neither inputs
  nor training targets.
- Report an ablation that breaks a rule as privileged, separately from the rows.

Cameras follow the same rule: by default they show the delayed scene. A camera with
`delayed=False` is privileged.

## Seeds and splits

| Use | Seeds |
|---|---|
| `dev` suite: closed-loop development and debugging | 0–19 |
| `test` suite: final closed-loop results, once per trained model | 1000–1049 |
| World-model training data, and the layouts that online methods train on (`ArenaGymEnv`) | 200000–298999 |
| World-model validation data (model selection) | 299000–299999 |
| Reserved by the research code (CrazyMARL training layouts) | 100000–102047 |

## Data

- **Collector:** `python -m uav_arena.collect --seeds 200000:202000 --out runs/datasets/explore_v2`.
  The behavior policy `explore-v2` flies the formation between random waypoints at random speeds
  and formation sizes, with the shared planner's action statistics. Each aircraft's increment is
  the step to its place in the formation plus Gaussian noise at MPPI's sampling scale.
- **Episodes** are open-field versions of the seeds' scenarios (start, payload mass, wind; no
  obstacles), up to the time limit. A file holds the true state every 10 ms (aircraft position,
  velocity, rotation columns, body rates; payload position and velocity) and every setpoint flown
  with its time, so any latency is applied offline (`openloop.observation`).
- **Standard datasets:** with the default settings, seeds 200000–201999 (2,000 training episodes)
  and 299000–299099 (100 validation episodes) are arena-v0.3's `explore-v2` data. To check a
  regenerated dataset against the published content hashes, run
  `python -m uav_arena.collect --verify DIR`.
- **Tuned families:** `--settings file.json` collects from another scenario family. Its manifest
  records the settings, and its files store the true wind next to the true mass.
- **Training windows:** `windows.py` turns episodes into training examples aligned exactly as
  `openloop.py` scores predictions, with the same padding at the episode start.
- **Older policy:** `explore-v1` (`--policy explore-v1`) stays available with its manifests;
  results on it are not comparable with `explore-v2` results.
