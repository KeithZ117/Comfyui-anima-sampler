# AGENT.md

This repository is for researching and building a ComfyUI custom-node plugin
for inference-time control of `circlestone-labs/Anima`.

The project goal is not model training. The goal is a small, testable,
non-invasive ComfyUI control stack that makes Anima sampling easier to inspect,
branch, and correct.

## Working Principles

Use these principles before touching code:

1. Think before coding.
   State assumptions, inspect the local code that matters, and surface unknowns.
   Do not silently assume SDXL, SD 1.5, CLIP, or UNet behavior.

2. Simplicity first.
   Prefer the smallest node or helper that proves a useful inference-time
   behavior. Do not add speculative abstractions, model weights, network
   services, or training pipelines.

3. Surgical changes.
   Touch only files required for the current task. Do not rewrite unrelated
   project structure, formatting, or comments.

4. Goal-driven execution.
   Convert implementation requests into verifiable outcomes: loadable node,
   visible node mapping, sampler parity check, preview output, branch output, or
   masked latent edit. Finish with the checks that were run.

These principles are adapted from the Karpathy-inspired CLAUDE.md guidance:
think before coding, keep changes simple, make surgical edits, and verify the
goal rather than only following imperative steps.

## Project Facts

- Target model: `circlestone-labs/Anima`.
- Base family: Anima is a derivative of
  `nvidia/Cosmos-Predict2-2B-Text2Image`.
- Runtime target: ComfyUI custom node package.
- ComfyUI model files for Anima:
  - `models/diffusion_models/anima-base-v1.0.safetensors`
  - `models/text_encoders/qwen_3_06b_base.safetensors`
  - `models/vae/qwen_image_vae.safetensors`
- Anima is a 2B anime / illustration-oriented text-to-image model. It is not a
  realism model.
- Cosmos-Predict2-2B-Text2Image is a latent-space diffusion transformer with
  self-attention, cross-attention, and feedforward blocks. Text conditioning is
  injected through cross-attention during denoising, and timestep information is
  embedded through adaptive layer normalization.

Read these project notes before making architecture decisions:

- `docs/anima-model-card-notes.md`
- `docs/cosmos-predict2-2b-text2image-notes.md`
- `docs/sources.md`

## Design Bias

Start with conditioning-time and sampler-scheduling controls. Do not modify
self-attention in the MVP. Only investigate cross-attention routing after the
local ComfyUI model wrapper exposes a clear and reversible hook.

The first useful abstraction is phase-aware sampling:

- Early phase: structure, pose, count, camera, composition.
- Mid phase: identity, character attributes, objects, props.
- Late phase: style, lineart, shading, quality, detail.

For high-denoise redraw and img2img, prefer tools that preserve or expose the
early latent trajectory: segmented sampling, early previews, branch-and-prune,
and local re-noise.

## Non-Goals

- Do not train or fine-tune Anima.
- Do not train ControlNet or BrushNet-like weights for the MVP.
- Do not modify Qwen text encoder weights, LLM adapter weights, or VAE weights.
- Do not add cloud APIs, VLM scoring, external services, or network
  dependencies.
- Do not assume CLIP token offsets, SDXL prompt internals, or UNet attention
  module names.
- Do not silently change neutral sampling behavior.
- Do not bundle model weights in this repository.

## Implementation Sequence

When implementation begins, use this order unless the user changes scope:

1. Inspect only the relevant local ComfyUI API available to the project.
2. Create a minimal ComfyUI custom-node package skeleton.
3. Implement `AnimaPromptPhaseBuilder` as pure data/config.
4. Implement `AnimaSegmentedSampler` using verified sampler APIs.
5. Add intermediate latent checkpoint outputs.
6. Add optional VAE-based preview decoding.
7. Add manual branch-and-prune sampling.
8. Add masked latent region re-noise.
9. Investigate cross-attention hooks only after the safe hook surface is known.

## Required MVP Nodes

- `AnimaPromptPhaseBuilder`
- `AnimaSegmentedSampler`
- `AnimaEarlyCheckpointPreview`
- `AnimaBranchSampler`
- `AnimaRegionRenoise`

All advanced hooks must degrade gracefully with clear warnings when unsupported.

## Verification Rules

For documentation-only work:

- Verify the files exist.
- Keep source links in `docs/sources.md`.
- Mark source facts separately from project hypotheses.

For code work:

- Verify import/load behavior where possible.
- If ComfyUI is not available in the current project, do not fake a passing
  runtime test. State the limitation.
- For neutral segmented settings, compare behavior against the standard sampler
  path as closely as the local API allows.
- Log phase boundaries and active conditioning for segmented sampling.

## Licensing And Safety

Anima is under the CircleStone Labs non-commercial license and is also subject
to NVIDIA Open Model License terms as a derivative of Cosmos-Predict2. Do not
represent this project as commercial-ready unless licensing is reviewed.

Do not bypass, disable, or reduce model guardrails. Do not add code that changes
safety behavior outside the user's explicit, lawful local workflow.

