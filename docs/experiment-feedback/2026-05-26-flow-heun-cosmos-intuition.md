# 2026-05-26 Flow Exp Heun + Flow Cosmos Intuition

## Observation

`flow_heun + flow_cosmos` was reported as the best tested combination so far
for semantic following and detail handling.

## Mathematical Intuition

Anima/Cosmos sampling follows a rectified-flow path:

```text
x_t = (1 - t) * x0 + t * eps
```

The model wrapper returns a denoised/data prediction:

```text
D_theta(x_t, t) = x0_pred
```

If `x0_pred` were constant over one step, the exact RF update is:

```text
x_next = (t_next / t) * x + (1 - t_next / t) * x0_pred
```

So the main error is not the homogeneous transport term. The error comes from
`x0_pred` changing as the latent moves to the next noise level.

`flow_heun` addresses that specific error. It first predicts the next state
with the constant-`x0` exact update, then evaluates the model again at the
predicted endpoint. It integrates the changing `x0_pred` in RF logSNR space:

```text
lambda = log((1 - t) / t)
```

This makes the method a flow-aware predictor-corrector rather than a generic
velocity trapezoid.

`flow_cosmos` helps because its external sigmas are log-spaced and then mapped
to RF time:

```text
t = sigma_external / (1 + sigma_external)
lambda = -log(sigma_external)
```

Therefore `flow_cosmos` is close to uniform stepping in RF logSNR space. That
is the same coordinate used by the exponential Heun correction, so the schedule
and solver are aligned.

## Why Quality Improves

- The solver corrects changes in `x0_pred` across a step, which matters for
  semantic binding and fine details.
- The schedule spends steps in a coordinate where RF integration is smoother,
  reducing uneven jumps between noise levels.
- The endpoint model call lets the sampler re-check the prompt-conditioned
  prediction after moving to the next denoise level.
- The final step returns the denoised prediction directly, avoiding unstable
  high-order extrapolation near `t = 0`.

## Hypothesis

The strong result is likely not just because Heun is second order. It is because
the implemented solver, model output type, and schedule all agree on the same
RF structure:

```text
RF path + x0 prediction + exponential lambda integral + logspace Cosmos schedule
```

Further quality work should preserve this alignment.
