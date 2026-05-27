# Anima Flow Sampler Improvement Plan

This document tracks planned algorithm-level experiments for the Anima Flow
sampler after the first stable baseline.

## Current Baseline

```text
steps                 = 35
cfg                   = 6.0
flow_solver           = flow_pc3_damped
flow_er_order         = 2
flow_schedule         = flow_cosmos
cosmos_sigma_max      = 80.0
cosmos_sigma_min      = 0.002
denoise_legacy_progress = false
cfg_legacy_progress   = false
cfg_schedule_mode     = beta_bump
cfg_early_scale       = 0.98
cfg_early_ramp_end    = 0.10
cfg_peak_boost        = 0.60
cfg_bump_start        = 0.08
cfg_bump_end          = 0.68
cfg_beta_alpha        = 4.0
cfg_beta_beta         = 7.0
cfg_interval_start    = 0.12
cfg_interval_rise_end = 0.24
cfg_interval_fall_start = 0.36
cfg_interval_end      = 0.58
early_cfg_boost       = 0.5
early_cfg_until       = 0.30
late_cfg_scale        = 0.92
late_cfg_start        = 0.76
rf_endpoint_noise_refresh_enabled = false
rf_endpoint_noise_refresh_strength = 0.15
rf_endpoint_noise_refresh_until = 0.20
```

Observed behavior:

- Good prompt robustness across studio, sports, market, and library prompts.
- `cfg = 6.0` is the best current balance point.
- `flow_pc3_damped` is the current formal-quality solver. `flow_er` remains
  the low-cost one-call option.
- `flow_cosmos` is the stable baseline schedule; rho7 RF-tail schedules are the
  current dynamic-composition experiments.
- Noise kick has been removed; RF endpoint-noise refresh remains the only
  stochastic RF-side perturbation control.

## Node Layout

The sampler now separates reusable settings from execution:

```text
Anima Flow Settings
  -> ANIMA_FLOW_SETTINGS
  -> Anima Flow Corrective Sampler
```

`Anima Flow Settings` carries the tuned baseline and exposes the algorithmic
controls:

```text
steps, cfg, flow_solver, flow_er_order, flow_schedule,
cosmos_sigma_max, cosmos_sigma_min,
denoise_legacy_progress,
cfg_legacy_progress,
cfg_schedule_mode,
cfg_early_scale, cfg_early_ramp_end, cfg_peak_boost,
cfg_bump_start, cfg_bump_end, cfg_beta_alpha, cfg_beta_beta,
cfg_interval_start, cfg_interval_rise_end,
cfg_interval_fall_start, cfg_interval_end,
early_cfg_boost, early_cfg_until,
late_cfg_scale, late_cfg_start,
rf_endpoint_noise_refresh_enabled, rf_endpoint_noise_refresh_strength,
rf_endpoint_noise_refresh_until
```

The sampler requires `ANIMA_FLOW_SETTINGS` and keeps runtime inputs local:

```text
model, conditioning, latent, flow_settings, seed, denoise, add_noise, disable_pbar
```

The sampler no longer exposes duplicate algorithm widgets. Workflows should
connect `Anima Flow Settings` before the sampler.

For experiment grids, use:

```text
Anima Flow Matrix Sweep
```

Default matrix:

```text
primary_sweep_parameter   = flow_schedule
primary_sweep_values      = flow_cosmos_lambda_biased_strong,
                            flow_cosmos_rho7_rf_tail_auto
secondary_sweep_parameter = flow_solver
secondary_sweep_values    = flow_pc3_damped,
                            flow_pc3_fsal_gated,
                            flow_3m_sparse_pc3_fsal,
                            flow_3m_damped,
                            flow_heun
include_comfy_er_sde_simple = false
columns                   = 5
max_runs                  = 10
```

This is intended for same-seed side-by-side testing of PC3 speedups on the two
current schedule contenders: strongest lambda bias and the kept
`flow_cosmos_rho7_rf_tail_auto` schedule. The default matrix compares full PC3,
FSAL-gated PC3, sparse PC3+FSAL, one-eval 3M, and `flow_heun`.
`flow_er`, `flow_cosmos`, and `flow_cosmos_beta5` remain available manually,
but are not part of the default speedup preset. The old
`flow_rho7_euler` ablation was removed because the sigma-exact version is
equivalent to `flow_euler`. Use
`Anima Flow Parameter Sweep` for smaller one-parameter checks such as
`steps = 24, 30, 36` or `denoise = 0.25, 0.5, 0.75`.
Its default `flow_schedule` sweep uses `max_runs = 2` so both focused profiles
above are included.

The matrix appends one stock ComfyUI baseline tile when
`include_comfy_er_sde_simple = true`:

```text
sampler_name = er_sde
scheduler    = simple
```

This path bypasses the custom RF solver, RF schedule, lambda CFG schedule,
RF denoise mapping, and endpoint refresh controls. It is only a native ComfyUI
reference point.

The dated evening plan for the first dynamic CFG validation pass is recorded in
`docs/experiment-feedback/2026-05-27-dynamic-cfg-evening-plan.md`.

## Current Core Algorithm

1. Build the selected sigma schedule.
2. Run ComfyUI `sample_custom` with a custom `KSAMPLER`.
3. Apply dynamic CFG before each model call.
4. Convert the denoised prediction to flow velocity and integrate with the
   selected solver.

```text
v = (x - x0) / sigma
x_next = x + (sigma_next - sigma) * v
```

## Experiment 1: RF-Domain Control Windows

Problem:

- `early_cfg_until` and `late_cfg_start` were originally step-progress controls.
- Step progress is simple, but it is not the same as actual noise level.

Implemented for CFG:

- `cfg_legacy_progress = false` uses the RF-native default:
  `lambda = log((1 - t) / t)` normalized over the finite schedule.
- `cfg_legacy_progress = true` restores the old step-count behavior for
  ablations.

Test:

- Compare progress-domain baseline against lambda-domain equivalents at 25, 35,
  and 40 steps.

## Experiment 2: RF-Domain Denoise Strength

Implemented:

- `denoise_legacy_progress = false` maps partial denoise to an RF lambda start.
- For `flow_cosmos*` schedules, `denoise` now selects the starting external
  sigma by geometric interpolation between the finite low-noise and high-noise
  endpoints:

```text
sigma_start = sigma_min * (sigma_max / sigma_min) ** denoise
```

- `denoise_legacy_progress = true` restores the old Comfy-style behavior:
  build a longer schedule and keep the final `steps + 1` entries.
- The native-table `simple` baseline keeps the legacy truncation behavior.

Test:

- Compare `denoise = 0.25, 0.5, 0.75` with `denoise_legacy_progress` on/off.
- Watch image preservation, redraw strength, and whether structure changes
  scale more predictably across `flow_cosmos`, `rho7`, and
  `flow_cosmos_lambda_biased_strong`.

## Removed: Noise Kick

Noise kick was removed after multiple solver/scheduler iterations. The original
goal was to force the model to re-check the trajectory after several early
steps, but the current RF solvers already re-evaluate the model at every step
and the extra stochastic jump made solver history and RF time semantics harder
to reason about.

## Experiment 4: RF Exponential Heun / Predictor-Corrector

Implemented solvers:

```text
flow_euler  = one model call per step
flow_heun = RF x0 exponential Heun, usually two model calls per step
flow_pc3_damped = RF x0 exponential PC3 with damped AM3 correction,
                      usually two model calls per step
flow_pc3_fsal_gated = same PC3 correction, but endpoint x0 predictions can
                          be reused as the next step's current x0 when gated
flow_3m_damped = one-eval lambda-native 2M/3M with damped 3M extrapolation
flow_3m_sparse_pc3_fsal = one-eval 3M backbone with budgeted PC3 endpoint
                              correction and the same FSAL cache gate
flow_er     = deterministic RF x0 exponential LMS solver, order 1-3
```

Notes:

- This doubles model calls.
- It should be optional, not a replacement for Euler.
- It may improve fine details, hand-object contact, and background geometry.
- The terminal step still returns the denoised estimate directly instead of
  making a second call at zero sigma.
- `flow_heun` is the formal RF x0 exponential Heun path. The older
  velocity-trapezoid Heun implementation was removed from the dropdown and is
  not accepted.
- `flow_er` keeps one model call per step and uses an RF x0 LMS update in
  `lambda = log((1 - t) / t)` space. `flow_er_order = 2` is the default;
  order 3 is available for 35-40 step experiments. Endpoint-noise refresh
  resets its multistep history.
- `flow_rho7_euler` is no longer accepted as a solver option.
- `flow_pc3_damped` keeps `flow_heun` as the safety baseline, then
  applies a variable-step AM3 correction only when accepted-state x0 history is
  available. The correction is damped by an embedded AM3-vs-Heun error estimate
  and a lambda-space gate.
- `flow_pc3_fsal_gated` is the first PC3 speedup path. It keeps the current
  PC3 algebra and caches `D_pred_next = model(x_pred, t_next)` for the next
  step when `rms(x_next - x_pred)` and AM3-vs-Heun disagreement are small.
  The cache is force-refreshed near the final steps, after rho7 RF-tail
  boundaries, and after endpoint-noise refresh.
- `flow_3m_damped` is a fast one-call baseline. It computes lambda-native
  2M and 3M predictions and blends toward 3M only when the 2M-vs-3M disagreement
  is below the PC3 tolerance budget.
- `flow_3m_sparse_pc3_fsal` is the experimental final target: 3M damped by
  default, with budgeted endpoint PC3 in body/tail risk regions and FSAL cache
  on accepted endpoint calls.

Test:

```text
flow_solver = flow_pc3_damped,
              flow_pc3_fsal_gated,
              flow_3m_damped,
              flow_3m_sparse_pc3_fsal,
              flow_heun
flow_pc3_gamma = 0.25, 0.5, 1.0
flow_pc3_tolerance = 0.002, 0.005, 0.01
steps = 25, 30, 35
```

If Heun at 25-30 steps matches Euler at 35, it may become a quality-efficient
option despite extra calls.

## Experiment 5: Schedule Variants

Current schedule dropdown:

```text
flow_schedule = flow_cosmos
              | flow_cosmos_lambda_biased_strong
              | flow_cosmos_beta5
              | flow_cosmos_rho7_rf_tail_auto
              | simple
```

Plan:

- Keep `flow_cosmos` as the default and current logSNR-uniform baseline. It
  uses exact external-sigma endpoints rather than a subsampled lookup table.
- Keep only `flow_cosmos_lambda_biased_strong` as the biased schedule profile;
  the light/default variants were removed from the formal dropdown.
- Use `flow_cosmos_beta5` to test the Cosmos-Predict2.5 report's 720p
  timestep-shift idea. In sigma-ratio coordinates this is equivalent to
  multiplying the external sigma ratio by beta before mapping back to RF time.
- Use `flow_cosmos_rho7_rf_tail_auto` as the single retained rho7-derived
  dynamic schedule.
- Keep `simple` only as a basic ComfyUI-style baseline.

## Experiment 6: Stress Prompts

The baseline passed broad prompt robustness. Further tests should focus on known
weak areas:

- left/right binding
- hand-object contact
- crowded small objects
- multiple characters with distinct actions
- high-denoise img2img redraw

Use the same seed set across tests:

```text
67, 68, 69, 70
```

## Experiment 7: Adaptive RF Step Refinement

Status: recorded idea, not implemented.

The sampler can eventually support dynamic steps by treating `steps` as the base
RF integration grid and allowing a solver to spend a bounded number of extra
model calls on hard intervals.

Best candidate:

```text
base solver = flow_heun
error signal = rms(x_heun - x_euler) / (rms(x_heun) + eps)
split point = lambda midpoint between the current and next RF endpoint
```

When the local predictor/corrector disagreement is above a tolerance, the
sampler can split:

```text
[lambda_i, lambda_{i+1}]
  -> [lambda_i, lambda_mid] + [lambda_mid, lambda_{i+1}]
```

Expected controls:

```text
adaptive_refine_enabled = false
adaptive_refine_tolerance = 0.003 to 0.01
adaptive_refine_max_splits = 1
adaptive_refine_max_extra_steps = 4 to 12
```

Rationale:

- Complex prompts may have higher local vector-field curvature in only part of
  the RF trajectory.
- Heun already asks the model twice, so its predictor/corrector disagreement is
  a natural local error estimate.
- Extra steps should be bounded and logged as extra model calls, otherwise
  comparisons become hard to reproduce.

Do not make this the default until fixed-grid scheduler/solver/denoise tests are
finished. It is better suited as a reference-quality mode than a baseline.

## Experiment 8: CFG Curve For High-Noise Structure

Status: implemented as selectable CFG schedule modes.

The original `early_cfg_boost` was a simple decaying boost: highest at the
first step, then back to base. The settings node now exposes
`cfg_schedule_mode` so that this old behavior is only one option:

```text
beta_bump        = default; mild start, bounded beta-shaped peak, late softening
limited_interval = trapezoid/window bump for controlled ablations
legacy_boost     = original early-high decaying boost
constant         = fixed CFG
```

Current default intent:

```text
very early high noise: slightly lower CFG
readable coarse stage: CFG peak
mid denoise: return near base CFG
late denoise: optional slight softening
```

Use the existing lambda-domain progress as the default curve domain. Step-count
progress is only for ablations.

Rationale:

- At the first pure-noise steps, strong CFG can over-constrain an unstable
  direction.
- Once the blurry preview already shows composition, a moderate CFG peak may
  improve structure, framing, and object count.
- After the coarse structure forms, returning to base CFG may preserve detail
  and reduce overcooked artifacts.

First dynamic two-character action prompt result:

- `flow_er` and the removed velocity-Heun ablation did not fully solve the
  prompt; Heun was better than `flow_er`.
- Strong schedule bias was the best variant for both.
- `flow_pc3_damped` was broadly usable, with the strongest biased schedule
  performing best.
- `flow_cosmos_rho7` with normal RF solvers preserved good detail.
- The old `flow_rho7_euler` path had strong motion/dynamic action behavior but
  unacceptable detail quality, and was later removed because its sigma-exact
  rewrite is equivalent to `flow_euler`.
- Native `er_sde + simple` was usable but weaker, with body-structure errors.

Next priority: test the rho7 RF-tail schedules with
`flow_pc3_damped + strong bias` as the quality baseline. Detailed notes are in
`docs/experiment-feedback/2026-05-27-dynamic-cfg-rho7-results.md`.

Implemented rho7 RF-tail schedules:

```text
flow_cosmos_rho7_rf_tail_auto     latest rho prefix with RF tail gap <= 0.5
```

This schedule keeps a rho7 high/mid-noise prefix, then replaces the low-noise
tail with nodes uniform in `ell = -log(t)`. `auto` chooses the latest feasible
switch point whose resulting uniform-ell tail stays within the configured
`tail_delta_ell_max` budget. The intent is to keep rho7's dynamic composition
bias while giving
`flow_heun` and `flow_pc3_damped` a more RF-native refinement tail.

For hybrid schedules, the sampler resets multistep solver history at the
detected prefix/tail boundary. This prevents `flow_er` and
`flow_pc3_damped` from carrying rho-prefix state into the uniform-ell tail.

## Experiment 9: High-Noise Evaluate-And-Rescue

Status: recorded idea, not implemented.

The user observed that a very blurry high-noise preview can already reveal
whether the final image is likely going in the wrong direction. A future
reference-quality mode could use this checkpoint for adaptive rescue:

```text
run to checkpoint -> evaluate coarse x0 preview -> accept or perturb/retry
```

Candidate rescue strategies:

- branch from the checkpoint with several endpoint-noise refresh strengths;
- jump back to an earlier high-noise RF time and resample;
- split the interval with extra Heun/PC evaluation before deciding;
- expose a manual preview gate before building any automatic scorer.

Guardrails:

- retry count must be bounded;
- all retry attempts must be logged with seed offsets and RF time;
- the default sampler must remain deterministic and fixed-cost;
- do not add external VLM/cloud scoring to the MVP.

This is better treated as a diagnostic or reference-quality mode than as the
default path.

Manual preview gate variant:

```text
run to high-noise checkpoint -> show coarse preview -> accept or retry
```

This is useful for studying the decisive high-noise interval, but it is deferred
because it needs interactive checkpoint state and UI support.

## Implementation Order

1. Add `schedule_domain` and sigma-based progress helpers.
2. Add optional RF solver variants.
3. Retest defaults only after each change.
4. Keep ComfyUI standard preview/progress behavior intact.
