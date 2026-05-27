# 2026-05-27 Dynamic CFG Evening Plan

## Goal

Tonight's experiment should answer one narrow question:

```text
Does a mild-start / early-mid peak CFG curve reduce high-noise trajectory
lock-in compared with constant CFG and the old first-step-heavy CFG boost?
```

Do not try to solve prompt staging, LoRA loading, manual preview gates, node
layout, or inpaint behavior tonight.

## Locked Baseline

Keep these fixed unless a section explicitly says to sweep them:

```text
steps                 = 35
cfg                   = 6.0
flow_solver           = flow_pc3_damped
flow_er_order         = 2
flow_pc3_gamma        = 1.0
flow_pc3_tolerance    = 0.005
flow_schedule         = flow_cosmos
cosmos_sigma_max      = 80.0
cosmos_sigma_min      = 0.002
denoise_legacy_progress = false
cfg_legacy_progress   = false
rf_endpoint_noise_refresh_enabled = false
```

Current dynamic CFG baseline:

```text
cfg_schedule_mode     = beta_bump
cfg_early_scale       = 0.98
cfg_early_ramp_end    = 0.10
cfg_peak_boost        = 0.60
cfg_bump_start        = 0.08
cfg_bump_end          = 0.68
cfg_beta_alpha        = 4.0
cfg_beta_beta         = 7.0
late_cfg_scale        = 0.92
late_cfg_start        = 0.76
```

## Prompt Set

Use a small fixed prompt set. Prefer 4-6 prompts that have previously shown
early structure failures:

- full-body / feet visible / wide framing;
- multi-object count or prop placement;
- unusual camera angle or pose;
- dense background layout where composition matters;
- one normal portrait/control prompt to catch quality regressions.

For each prompt, use the same seed across all variants. Use 2-3 seeds per
prompt only after the first pass shows a meaningful difference.

## Pass 1: Mode Sanity Check

Use `Anima Flow Parameter Sweep`:

```text
sweep_parameter = cfg_schedule_mode
sweep_values    = constant, legacy_boost, beta_bump, limited_interval
max_runs        = 4
```

Purpose:

- `constant`: static CFG reference.
- `legacy_boost`: old behavior, high at the first step then decay.
- `beta_bump`: current hypothesis.
- `limited_interval`: checks whether a simpler bounded window behaves similarly.

Do not tune any curve parameters in this pass. Only record which mode gives
better high-noise structure without worse final artifacts.

## Pass 2: Beta Bump Strength

Only run this if `beta_bump` is competitive in Pass 1.

Sweep:

```text
sweep_parameter = cfg_peak_boost
sweep_values    = 0.30, 0.60, 0.90
```

Readout:

- `0.30`: should be conservative and close to constant CFG.
- `0.60`: current balanced default.
- `0.90`: tests whether stronger structure binding overcooks details.

Reject higher peak if it improves prompt binding but causes stiff pose,
oversaturation, duplicated limbs, melted hands, or excessive outfit/style
over-constraint.

## Pass 3: Very-Early CFG

Only run this after choosing a tentative `cfg_peak_boost`.

Sweep:

```text
sweep_parameter = cfg_early_scale
sweep_values    = 0.95, 0.98, 1.00
```

Purpose:

- `0.95`: tests whether very early high-noise CFG should be lower.
- `0.98`: current mild reduction.
- `1.00`: no early reduction; isolates whether the bump alone matters.

If `1.00` wins, the problem is probably not "early CFG too strong"; the useful
part may be the later bump shape.

## Pass 4: Late Softening

Only run this if the previous passes produce a clear candidate.

Sweep:

```text
sweep_parameter = late_cfg_scale
sweep_values    = 0.90, 0.92, 1.00
```

Purpose:

- `0.90`: stronger late de-emphasis.
- `0.92`: current default.
- `1.00`: no late softening.

Judge mostly on final texture, line quality, local anatomy, and whether prompt
critical structure remains intact.

## Optional Pass 5: Solver Interaction

Only run if there is still time and the CFG curve result is clear.

Use `Anima Flow Matrix Sweep` with the selected CFG curve settings:

```text
primary_sweep_parameter   = flow_solver
primary_sweep_values      = flow_pc3_damped, flow_heun, flow_er
secondary_sweep_parameter = <none>
max_runs                  = 3
```

Purpose:

- Check whether dynamic CFG prefers the formal `flow_pc3_damped` path, the
  safer two-call `flow_heun`, or the older `flow_er` ablation.
- Do not change schedule and solver at the same time tonight.

## What To Record

For each grid, save the output image and log. Add short notes with this shape:

```text
prompt:
seed:
pass:
winner:
failures:
early structure:
final quality:
notes:
```

Use these criteria:

- coarse framing / body extent established correctly;
- object count and major props present;
- pose and camera angle obeyed;
- background layout not collapsed;
- no worse hands/limbs/duplicates;
- no overcooked color, texture, or style.

## Stop Rules

Stop early if:

- `beta_bump` loses clearly to `constant` on most prompts;
- any setting only improves one prompt type while damaging the control prompt;
- differences are not visible on first-pass grids.

If the result is ambiguous, keep the current `beta_bump` default and collect
more matched-seed examples later. Do not add new mechanisms tonight.

## Notes For Future Work

While watching previews, record the approximate stage where a bad composition
becomes obvious. This is only data collection for the future manual preview
gate idea. Do not implement the gate tonight.
