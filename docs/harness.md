# Evaluation harness: layers 1–3

The harness scores a predictor on decision states recorded from the oracle's flights, against the
true simulator. Three layers ask increasingly decision-relevant questions:
1. How well does it predict what happens under a plan?
2. Does it get the effect on the other agents right when one aircraft changes its actions?
3. Would it pick good plans?

Layer 4, the closed loop, is `run.py`.

```bash
python -m uav_arena.record --suite dev --seeds 0:5 --out runs/states/dev \
  --results runs/results/oracle_dev_recorded.json
python -m uav_arena.harness truth --states runs/states/dev
python -m uav_arena.harness score --predictor physics --states runs/states/dev \
  --out runs/results/harness/physics_dev.json
```

`score` takes the same `--predictor` and `--predictor-options` as `run.py`, so a participant's
`module:factory` runs unchanged. `record` takes `--settings` to record a tuned scenario family. The
episode files keep the true wind next to the true mass, and the harness rolls out the true
scenario from them.

## Questions: decision states flown by the oracle

- **Source.** The oracle U (`TrueSimPredictor` with the shared MPPI) flies a suite in all 8
  conditions. Every model is scored on the same questions: every 5th decision of every recorded
  episode. A stateful predictor observes every decision of the episode up to a question.
- **Recorder** (`record.py`). It flies U through `run.run_episode` and the shared planner, unchanged;
  thin wrappers only read what each solve uses and returns. A file per episode extends
  `collect.py`'s format. Besides the true 10 ms state log and every setpoint sent, it holds U's plan
  at every decision. At each question it also holds the physics snapshot U planned from, the solver's
  generator state (so the 512 candidates U scored can be regenerated exactly), and U's cost of each
  candidate, which is their true cost.

## Calling a predictor

The harness uses only the predictor interface ([participants.md](participants.md)). Each episode
starts with `reset()`. At every decision the harness calls `observe(obs, layout)` with the
observation the planner saw. At a question it calls the predictor once, on U's plan, the 42
layer-2 changes and the 512-candidate pool: a leading shape of (1, 555). Predictors never see
snapshots, true states or hidden fields. A stochastic predictor (`stochastic = True`) is called once
per sequence for layer 2, each call after the same seed, and 5 times with different seeds for layers
1 and 3, whose metrics are averaged. A predictor that does not declare itself stochastic but returns
different outputs for the same input stops the run.

## Layer 1: factual prediction

The truth is `simulate` of U's 15-decision plan from the snapshot: an open-loop rollout. The
harness reports payload and aircraft position errors per step and averaged over the 15 steps. **L1**
is the payload error averaged over the steps.

## Layer 2: cross-agent counterfactuals

Each aircraft in turn gets changed increments, while the other two keep theirs:
- **A, 8 per aircraft:** Gaussian noise at MPPI's own sampling scale, 0.04 m per decision and axis,
  seeded by layout, obstacle setting and decision.
- **B, 6 per aircraft:** a 0.10 m offset along ±x, ±y or ±z, ramped in over 3 decisions and then held.

The predicted effect f(changed) − f(base) is compared with the true effect, both rolled out from
the snapshot. The error is averaged over the 15 steps for the payload, the two unchanged aircraft,
and the changed aircraft (a control for models that ignore actions). **L2**, the cross-agent
counterfactual error, is ½ payload + ½ other aircraft, over the A changes. The harness also reports
a relative version, the error divided by the mean true effect, where that effect is at least 0.5 cm.

## Layer 3: decision level

- **Pool and costs.** The pool is the 512 candidates U's MPPI scored at that decision. The model's
  predicted costs use the shared cost on its predictions. The true costs are U's own.
- **Executed-plan regret.** The model's executed plan is the MPPI update on its own costs: softmax
  weights at temperature 0.05, the weighted mean of the candidates, clipped. Its regret is its true
  cost minus that of U's executed plan.
- **L3.** The share of costly decisions, those with regret above 1 cost unit, among questions with a
  reliable reference, where U's plan costs at most 1 unit more than the best candidate.
- **Also reported.** Elite regret (top-1, top-8), mean and median regret, and the Spearman correlation
  and pairwise accuracy of predicted against true costs over the pool.

The harness scores the executed plan rather than a ranking of the whole pool, because MPPI's
weights concentrate on a few candidates and most of the pool sits far above the best.

## Outputs

`score` writes a JSON summary, overall and per condition, with the source version, the question
set's manifest hash, the checkpoint hash, and whether the predictor was stochastic or privileged.
Beside it, an `.npz` holds every question's metrics with its layout seed and condition, for
bootstrap analyses over layouts. U scored as a predictor gets exactly zero on every layer.
