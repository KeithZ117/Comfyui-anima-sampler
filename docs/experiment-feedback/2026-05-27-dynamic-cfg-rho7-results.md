# 2026-05-27 Dynamic CFG / Rho7 Matrix Results

## Observation

The matrix used a complex two-character YuruYuri prompt with dynamic motion,
large bouquet detail, depth of field, bokeh, chromatic aberration, festival
lighting, wet pavement reflections, and full-body action framing.

User-visible result summary:

- `flow_er`: almost never fully correct. It can produce usable image structure,
  but the dynamic two-person action and object relationships are not reliable.
- `flow_heun`: also failed across the set, but generally looked better than
  `flow_er`.
- Strong schedule bias was the best variant for both `flow_er` and `flow_heun`.
- `flow_pc3_damped`: all variants were usable. The strongest biased
  schedule was the best result in this group.
- `flow_cosmos_rho7` with normal RF solvers: detail quality is good. The rho7
  schedule itself is not the detail problem.
- `flow_cosmos_rho7 + flow_rho7_euler`: strongest dynamic motion and action
  energy, but detail quality was poor enough that the output was not
  acceptable. This solver has now been removed because the sigma-exact rewrite
  is equivalent to `flow_euler`.
- Native `er_sde + simple`: weak compared with custom RF paths. It can make an
  image, but body structure is wrong and the dynamic pose fails more often.

## Interpretation

The useful signal is not simply "more solver order is better."

Current read:

```text
flow_pc3_damped is the current best quality/structure path.
strong schedule bias helps dynamic composition.
flow_cosmos_rho7 is useful as a schedule.
flow_rho7_euler was not worth keeping as a separate solver.
```

This suggests the next work should prioritize the rho7 schedule with normal RF
solvers rather than continuing to tune `flow_rho7_euler`.

## Rho7 Hypothesis

`flow_cosmos_rho7` may be placing useful integration density or coordinate
behavior around the part of the trajectory where motion and body arrangement
are established. The detail failure was isolated to `flow_rho7_euler`, not to
the rho7 schedule.

Resolution:

- The old `flow_rho7_euler` integrated in linear external-sigma coordinates.
- The first attempted repair used raw `sigma_ext ** (1 / 7)` Euler, but that
  coordinate is only for schedule placement, not RF ODE integration.
- The sigma-exact RF x0 Euler coefficient written in sigma form is:
  `A = (sigma_next - sigma) / (sigma * (1 + sigma_next))`.
- This directly fixes the old systematic under-denoise caused by using
  `1 + sigma` in the denominator.
- That sigma-exact form is algebraically the same as `flow_euler`, so the
  separate `flow_rho7_euler` option was removed.

Remaining checks:

- The old matrix pairing hid the fact that rho7-like schedule shaping works
  with normal RF solvers.
- Dynamic CFG defaults may interact differently with the rho7 schedule domain.
- Keep only `flow_cosmos_rho7_rf_tail_auto` as the retained rho7-derived
  dynamic schedule, then compare it against `flow_cosmos` and the strongest
  lambda-biased schedule with `flow_pc3_damped`.

## Next Actions

Prioritize these in order:

1. Use `flow_cosmos_rho7_rf_tail_auto` with normal RF solvers in default matrix
   tests.
2. Compare `flow_cosmos_rho7_rf_tail_auto + flow_pc3_damped` against the
   strongest lambda-biased PC3 result.
3. Keep `flow_pc3_damped + strong bias` as the current quality baseline.

## Follow-Up Grid

Use the same prompt and seed set. The default `Anima Flow Matrix Sweep` now
matches this focused hybrid-rho7 pass:

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
max_runs                  = 10
```

Record dynamic pose, body correctness, bouquet detail, background detail, and
line/texture quality separately. Do not judge rho7 only by motion energy.
Also record `actual_model_calls`, `cache_accept_rate`, and whether the FSAL or
3M variants visibly preserve the full PC3 result at lower call count.

## Updated Stress Prompt

Use `Anima Flow Test Prompt` with:

```text
prompt_case = yuruyuri_4girls_dynamic_festival
```

Positive prompt:

```text
masterpiece, best quality, score_7, safe, very aesthetic, official art, yuru yuri, yuruyuri, 4girls, akaza akari, toshinou kyouko, funami yui, yoshikawa chinatsu, nanamori school uniform, school uniform, pleated skirt, long sleeves, sailor collar, red ribbon, brown cardigan, full body, dynamic angle, wide shot, dutch angle, foreshortening, complex group pose, synchronized motion, festival street, rainy night, wet pavement, mirror-like reflections, backlighting, rim light, high contrast, depth of field, blurry background, bokeh, chromatic aberration, lens flare, light particles, falling petals, floating ribbons, confetti, motion blur, huge bouquet, mixed flowers, rose, lily, daisy, baby's-breath, flower petals, leaf, intricate floral pattern, detailed fabric pattern, embroidered ribbon, lace trim, Kyouko leaping backward while pulling Akari by the wrist, Akari stumbling forward with one foot off the ground, Yui catching the oversized bouquet with both hands, Chinatsu spinning under a long ribbon with her skirt and hair swirling, interlocked arms, crossed legs, expressive faces, happy, surprised, laughing, wind, ultra-detailed, huge filesize
```

Negative prompt:

```text
fused fingers, mutated hands, bad hands, extra fingers, missing fingers, fused arms, extra arms, missing arms, extra legs, missing legs, extra toes, bad feet, bad anatomy, wrong body count, duplicate character, merged bodies, cropped head, simple background, nipples, cleavage, nsfw, worst quality, low quality, score_1, score_2, score_3, lowres, bad, text, error, jpeg artifacts, watermark, unfinished, displeasing, oldest, early, signature, artist name, username, scan, abstract, english text, shiny hair
```

This prompt is intentionally harder than the prior two-character bouquet test:
four named characters, crossing actions, one oversized detailed object, wet
reflections, depth of field, bokeh, chromatic aberration, ribbons, petals, and
full-body dynamic framing.

## PC3 Speedup First Result

Observation:

- Broad PC3 speedup variants showed visible quality loss.
- The standout exception was:

```text
flow_schedule = flow_cosmos_rho7_rf_tail_auto
flow_solver   = flow_pc3_fsal_gated
```

- This combination looked good to the user, with little visible quality loss.
- Runtime was about 16 seconds shorter than the full PC3 comparison.

Interpretation:

- FSAL cache should not be treated as universally free. The quality drop in
  other combinations suggests stale endpoint reuse can damage detail or
  structure depending on schedule/solver interaction.
- `rho7_rf_tail_auto` may be a better match for gated endpoint reuse because
  its RF-native tail keeps late refinement steps more regular, making
  `D_pred_next` closer to a valid next-step current denoised estimate.
- Keep `flow_pc3_damped` as the quality reference, but promote
  `flow_cosmos_rho7_rf_tail_auto + flow_pc3_fsal_gated` to the current
  practical speed/quality candidate.

Follow-up:

- Rerun the standout pair on 2-3 matched seeds with the same prompt.
- Compare against full PC3 only, not the whole matrix, to confirm whether the
  16-second speedup repeats.
- Watch bouquet microdetail, hand contact, and face consistency specifically;
  these are likely places where stale endpoint cache would fail first.

Implementation fix after the first run:

- A likely FSAL quality bug was found and fixed: cached endpoint denoised values
  were being written into PC3/3M multistep history as if they were evaluated on
  accepted sampler states.
- Cached `D_pred_next` is now allowed to drive the current step, but it is not
  persisted into PC3/3M history. The next step is downgraded conservatively
  where relevant.
- Rerun `flow_cosmos_rho7_rf_tail_auto + flow_pc3_fsal_gated` after this
  fix before judging the final cache threshold.

Post-fix result:

- The rerun looked much better.
- The user could not see obvious quality loss compared with full PC3.
- Motion/action changed somewhat, but the changes looked like acceptable
  stochastic trajectory variation and still matched the prompt.

Current practical read:

```text
flow_cosmos_rho7_rf_tail_auto + flow_pc3_fsal_gated
```

is now the leading speed/quality candidate. Treat full PC3 as the reference
quality path, and use the gated version as the practical fast path unless a
later prompt exposes a repeatable detail or anatomy regression.
