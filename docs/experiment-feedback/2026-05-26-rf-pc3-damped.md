# RF-PC3-Damped Implementation

## Context

Source analysis: `docs/rf_flow_matching,analysis.md`.

The proposed solver keeps `flow_heun` as the stable RF x0 exponential
baseline, then adds a variable-step AM3 correction when the recent accepted
state history is valid.

## Implemented Behavior

- New solver option: `flow_pc3_damped`.
- Predictor: RF exponential AB2 endpoint predictor when one accepted previous
  x0 prediction is available; otherwise constant-x0 exact prediction.
- Corrector: RF exponential Heun baseline plus AM3 correction from previous,
  current, and endpoint x0 predictions.
- Damping:
  - embedded error from `rms(x_AM3 - x_H2) / (rms(x_H2) + eps)`;
  - `flow_pc3_tolerance` controls how much AM3 error is accepted;
  - `flow_pc3_gamma` caps total correction strength;
  - lambda gate suppresses very early high-noise and very late low-noise
    correction.
- History rule: only x0 predictions evaluated on accepted actual sampler
  states are stored. Endpoint predictions from predictor states are never stored
  as history.
- Noise kick and stochastic refresh reset PC3 history conservatively.

## Controls To Sweep

```text
flow_solver = flow_pc3_damped
flow_schedule = flow_cosmos
flow_pc3_gamma = 0.25, 0.5, 1.0
flow_pc3_tolerance = 0.002, 0.005, 0.01
steps = same as flow_heun baseline
```

## Current Status

Implementation is unit-tested for formula behavior and node wiring. Image
quality is not yet validated. Compare directly against the reported strong
baseline:

```text
flow_solver = flow_heun
flow_schedule = flow_cosmos
```

