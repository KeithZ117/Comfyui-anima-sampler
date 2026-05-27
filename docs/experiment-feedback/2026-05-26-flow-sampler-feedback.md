# 2026-05-26 Flow Sampler Feedback

## Observations

- `flow_heun` produced noticeably better prompt following and detail handling
  than `flow_euler`.
- `flow_cosmos` produced strong results and became the preferred schedule for
  this sampler. Shifted and shifted-tail schedule options were removed after
  comparison.
- The previous noise kick behavior could make later `flow_er` generations turn
  black. Random perturbations crossing multistep history boundaries were treated
  as unsafe unless history is reset or the step is downgraded.
- `flow_heun + flow_cosmos` was reported as the best tested combination so
  far for semantic following and fine detail handling.

## Settings Mentioned

- Model: Anima / Cosmos Predict2 derivative.
- Preferred schedule: `flow_cosmos`.
- Strong solver result: `flow_heun`.
- ER baseline under investigation: `flow_er` with configurable order.
- Noise kick: useful in some early tests, but unsafe when it contaminates
  multistep history.
- Prompt, seed, exact step count, and exact CFG for the reported image
  comparisons: not provided.

## Interpretation

- RF x0 exponential Heun appears to match the local flow formulation better
  than plain velocity Euler for instruction following and detail.
- `flow_cosmos` likely helps because its external sigma logspace schedule maps
  naturally to a smoother logSNR-like coordinate for flow integration.
- Random jumps need to be flow-aware. If kick or stochastic refresh is used, the
  solver must avoid reusing stale multistep history from before the jump.

## Actions Taken

- Kept `flow_cosmos` as the preferred/default schedule.
- Added `flow_heun` as an RF x0 exponential Heun solver.
- Reworked `flow_er` toward RF x0 exponential LMS behavior with order control.
- Reworked noise kick into an RF endpoint kick and reset/downshifted solver
  history across stochastic boundaries.

## Follow-Up Comparisons

- Compare `flow_heun + flow_cosmos` against `flow_er_order=2` and
  `flow_er_order=3` on the same prompts and seeds.
- Test RF endpoint kick only in early steps with small gamma values.
- Record whether kick improves semantic exploration, detail, composition, or
  only introduces more variation.
- Track black/gray image failures with the exact solver, schedule, seed, and
  kick/stochastic settings.
