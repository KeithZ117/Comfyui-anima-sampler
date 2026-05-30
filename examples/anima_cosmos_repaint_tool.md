# Anima Cosmos Repaint Tool

This branch adds a first-pass repaint toolkit for Anima/Cosmos-style image edits.

## Nodes

### Anima Cosmos Repaint Prepare

Inputs:

- `image`: source image.
- `mask`: white is repaint area, black is keep area.
- `vae`: encodes the source or filled image into a sampler latent.
- `mode`: `upscale clean`, `edge repair`, or `structure repaint`.
- `mask_grow` / `mask_feather`: expand and soften the repaint region.
- `latent_fill`: how the sampler latent is initialized inside the mask.
- `control_fill`: how the control/reference image is initialized inside the mask.

Outputs:

- `latent`: encoded latent with `noise_mask` already attached.
- `control_image`: masked image for Anima LLLite or reference workflows.
- `mask`: processed soft mask.
- `mask_preview`: red overlay preview.
- `log`: recommended sampler settings for the selected mode.

Recommended starting settings:

```text
upscale clean:      steps 16, cfg 4.5, denoise 0.22
edge repair:        steps 16, cfg 4.5, denoise 0.32
structure repaint:  steps 22, cfg 4.2, denoise 0.52
```

### Anima Cosmos Reference Latent

This is the no-ControlNet path inspired by the Cosmos reference-latent workflow.
It patches the model wrapper so reference latents are appended on the Cosmos time
axis during `apply_model`, then the output is cropped back to the original frame
count.

Typical workflow:

```text
source image + mask
-> Anima Cosmos Repaint Prepare
-> latent -> Anima Flow Corrective Sampler

source/reference latent
-> Anima Cosmos Reference Latent
-> patched model -> Anima Flow Corrective Sampler
```

This should be treated as an edit/reference constraint, not a trained inpaint
model. It is best for preserving identity, color, and broad context while the
mask/noise_mask controls where the sampler is allowed to change pixels.

## LLLite / ControlNet Route

kohya-ss published Anima LLLite inpaint weights using 4-channel conditioning
(`RGB + mask`) and a ComfyUI node that loads the weights:

- https://huggingface.co/kohya-ss/Anima-LLLite
- https://github.com/kohya-ss/ComfyUI-Anima-LLLite

Use `control_image` and the processed `mask` from `Anima Cosmos Repaint Prepare`
as inputs to that node. This route is the better candidate when the masked
region must be regenerated structurally, because the model has learned an
explicit inpaint condition.

## Decision Rule

Use no-ControlNet reference latent first for small edits and local cleanup:

```text
target: preserve original identity, color, style, pose
mask: small to medium
denoise: 0.18-0.35
```

Use LLLite/ControlNet inpaint when structure is actually wrong:

```text
target: rebuild hands, fingers, clothing, local objects
mask: whole anatomical/object region
denoise: 0.35-0.60
```

For upscale cleanup, run a paired test before promoting a default:

```text
A: Repaint Prepare + noise_mask only
B: Repaint Prepare + Reference Latent
C: Repaint Prepare + LLLite inpaint
```

Keep seed, prompt, sampler, steps, cfg, resolution, and mask fixed. Score
structure correctness, edge ghosting, detail density, and color/style drift
separately.
