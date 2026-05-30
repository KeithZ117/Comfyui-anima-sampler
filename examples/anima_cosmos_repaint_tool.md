# Anima Cosmos Repaint Tool

This branch adds a first-pass repaint toolkit for Anima/Cosmos-style image edits.

## Nodes

### Route A: Anima T-Reference Repaint Route

Use this route when you do not want ControlNet/LLLite. It masks the repaint
area out of the reference image, appends that reference latent on the Cosmos
time axis during model apply, then prepares a masked latent for
`Anima Flow Corrective Sampler`.

Workflow:

```text
Checkpoint MODEL -> Anima T-Reference Repaint Route.model
Load Image.image -> Anima T-Reference Repaint Route.image
Load Image.mask  -> Anima T-Reference Repaint Route.mask
VAE              -> Anima T-Reference Repaint Route.vae

Anima T-Reference Repaint Route.model
-> Anima Flow Corrective Sampler.model

Anima T-Reference Repaint Route.latent
-> Anima Flow Corrective Sampler.latent_image
```

Starting settings:

```text
mode: structure repaint
latent_fill: latent noise
mask_grow: 24-48
mask_feather: 32-64
denoise: 0.55-0.85
cfg: 4.0-5.5
```

This is the route inspired by the discussion workflow that uses the model/UNet
forward path directly instead of training an inpaint ControlNet.

### Route B: Anima T-Reference Control Repaint Route

Use this route when you want to combine the time-axis reference trick with
Anima LLLite or another ControlNet-style node. This node intentionally has no
`MODEL` input so it can prepare the shared assets without creating a connection
cycle around external ControlNet nodes. The emitted `reference_latent` is
encoded from the same masked control image, so the old pixels inside the repaint
area are not copied back through the time-axis reference.

Workflow for LLLite or a model-patching ControlNet:

```text
Load Image.image -> Anima T-Reference Control Repaint Route.image
Load Image.mask  -> Anima T-Reference Control Repaint Route.mask
VAE              -> Anima T-Reference Control Repaint Route.vae

Route.control_image + Route.mask
-> Anima LLLite / ControlNet apply

Control-patched MODEL
-> Anima Cosmos Reference Latent.model

Route.reference_latent
-> Anima Cosmos Reference Latent.latent

Anima Cosmos Reference Latent.model
-> Anima Flow Corrective Sampler.model

Route.latent
-> Anima Flow Corrective Sampler.latent_image
```

Workflow for a conditioning-only ControlNet:

```text
Route.control_image + Route.mask
-> ControlNet apply -> positive/negative conditioning

Checkpoint MODEL + Route.reference_latent
-> Anima Cosmos Reference Latent
-> Anima Flow Corrective Sampler.model

Route.latent
-> Anima Flow Corrective Sampler.latent_image
```

Keep `control_fill: masked black` for Anima LLLite inpaint weights. The LLLite
inpaint model was trained with masked RGB plus a binary mask as 4-channel
conditioning, so this route outputs both `control_image` and `mask`.

### Anima Inpaint Latent Prepare

Use this lower-level node when you only want a sampler-ready masked latent.
For the full no-ControlNet time-reference workflow, prefer Route A.

Typical workflow:

```text
Load Image image
Load Image mask
VAE
-> Anima Inpaint Latent Prepare
-> latent -> Anima Flow Corrective Sampler latent_image
-> Save Image from sampler image output
```

The node only prepares the inpaint latent:

```text
image + mask
-> VAE encode
-> replace masked latent area with noise
-> attach latent noise_mask
```

White mask pixels are repainted. Black mask pixels are kept.

Starting settings:

```text
steps: 18
cfg: 4.5
denoise: 0.45
flow_solver: flow_unipc2_x0
mask_grow: 32
mask_feather: 16
latent_fill: latent noise
noise_seed: same seed you want for the masked latent start
```

For light edge cleanup, reduce `denoise` to `0.25-0.35`. For structural
redraws such as hands, use `0.45-0.60` and mask the whole local structure.
Use `latent_fill: original` only when you want masked img2img behavior. Avoid
`masked black` and `neutral gray` for normal inpaint; those are diagnostic
fills and can leave color remnants because Anima treats them as image content.

### Anima Cosmos Repaint Prepare

Inputs:

- `image`: source image.
- `mask`: white is repaint area, black is keep area.
- `vae`: encodes the source or filled image into a sampler latent.
- `mode`: `upscale clean`, `edge repair`, or `structure repaint`.
- `mask_threshold`: `0` selects any nonzero mask pixel; higher values require
  mask opacity greater than or equal to the threshold.
- `mask_grow` / `mask_feather`: expand and soften the repaint region.
- `latent_fill`: how the sampler latent is initialized inside the mask.
- `noise_seed`: deterministic seed for `latent_fill: latent noise`.
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

Use `control_image` and the processed `mask` from
`Anima T-Reference Control Repaint Route` or `Anima Cosmos Repaint Prepare` as
inputs to that node. This route is the better candidate when the masked region
must be regenerated structurally, because the model has learned an explicit
inpaint condition.

## Decision Rule

Use Route A first for small edits and local cleanup:

```text
target: preserve original identity, color, style, pose
mask: small to medium
denoise: 0.18-0.35
```

Use Route B with LLLite/ControlNet inpaint when structure is actually wrong:

```text
target: rebuild hands, fingers, clothing, local objects
mask: whole anatomical/object region
denoise: 0.35-0.60
```

For upscale cleanup, run a paired test before promoting a default:

```text
A: Repaint Prepare + noise_mask only
B: Route A, time-axis reference only
C: Route B, time-axis reference + LLLite inpaint
```

Keep seed, prompt, sampler, steps, cfg, resolution, and mask fixed. Score
structure correctness, edge ghosting, detail density, and color/style drift
separately.
