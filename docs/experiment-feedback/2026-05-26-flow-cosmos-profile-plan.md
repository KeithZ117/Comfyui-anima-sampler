# 2026-05-26 Flow Cosmos Profile Plan

## Source Facts

- The user supplied `user_input/2511.00062v2.pdf`, the Cosmos-Predict2.5
  technical report.
- The report defines Flow Matching with:

```text
x_t = (1 - t) x + t eps
v_t = eps - x
```

- The report samples training timesteps from a logit-normal distribution and
  applies:

```text
t_s = beta * t / (1 + (beta - 1) * t)
```

- The report says beta grows from `1` at 256p to `5` at 720p.
- The report also states that 5% of training samples are drawn from the highest
  2% of the noise distribution.
- The archived Predict2 rectified-flow scheduler uses `sigma_min=0.002`,
  `sigma_max=80.0`, and `order=7.0`.

## Implemented Profiles

- `flow_cosmos`: current logspace external-sigma profile, equivalent to roughly
  uniform RF logSNR steps.
- `flow_cosmos_lambda_biased_light`, `flow_cosmos_lambda_biased`, and
  `flow_cosmos_lambda_biased_strong`: fixed-endpoint RF lambda schedules with
  light/default/strong density bumps around high, middle, and low noise
  regions. These are intended to test density shaping without globally shifting
  the whole path toward higher noise.
- `flow_cosmos_beta5`: report-aligned beta=5 timestep shift. Since
  `sigma_ext = t / (1 - t)`, the beta shift is implemented as
  `sigma_ext_shifted = 5 * sigma_ext`, then mapped back to RF time.
- `flow_cosmos_rho7`: Predict2-style rho/order=7 external sigma grid, mapped
  back to RF time and followed by the sampler's terminal zero.

## Hypothesis

- `flow_cosmos` may remain strongest with `flow_heun` because both are
  aligned around RF logSNR integration.
- `flow_cosmos_beta5` may improve early semantic reconstruction or structure
  because it biases the schedule toward noisier states.
- `flow_cosmos_rho7` may reveal whether Anima benefits from the original
  Predict2 inference grid rather than the current logSNR-uniform grid.
- The lambda-biased profiles may improve shape/detail refinement without the
  low-noise endpoint loss of full beta5 shifting.

## Suggested A/B

Use the same prompt, seed, steps, CFG, and solver:

```text
flow_solver = flow_heun
flow_schedule = flow_cosmos, flow_cosmos_lambda_biased_light,
                flow_cosmos_lambda_biased,
                flow_cosmos_lambda_biased_strong,
                flow_cosmos_beta5, flow_cosmos_rho7
steps = 30, 35
cfg = 6.0
denoise_legacy_progress = false
cfg_legacy_progress = false
noise_kick_enabled = false
rf_endpoint_noise_refresh_enabled = false
```

Record failures separately for black/gray images, washed-out details, semantic
drift, or improved prompt binding.
