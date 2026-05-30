# Anima Cosmos Repaint Tool

This branch adds a first-pass repaint toolkit for Anima/Cosmos-style image edits.

## Nodes

### Anima T-Reference Edit Route

Use this route for the workflow from the Anima discussion: full-image edit,
AnimaEdit LoRA, no inpaint mask. It encodes the input image once and uses that
same latent as both the KSampler `latent_image` and the Cosmos time-axis
reference.

Workflow:

```text
Checkpoint/UNET MODEL
-> AnimaEdit LoRA
-> Anima T-Reference Edit Route.model

Load Image.image -> Anima T-Reference Edit Route.image
VAE              -> Anima T-Reference Edit Route.vae

Anima T-Reference Edit Route.model
-> native KSampler.model

Anima T-Reference Edit Route.latent
-> native KSampler.latent_image
```

Starting settings from the embedded discussion workflow:

```text
sampler: er_sde
scheduler: simple
steps: 22
cfg: 3.4
denoise: 1.0
prompt: describe the target edit
```

This route needs an AnimaEdit LoRA. Without the edit LoRA, the reference latent
mostly acts like a strong image-preservation constraint.

### Route A: Anima T-Reference Repaint Route

Use this route when you do not want ControlNet/LLLite. It fills the repaint
area with `reference_fill` before encoding the time-axis reference latent, then
prepares a masked latent for `Anima Flow Corrective Sampler`. The default
`reference_fill: neutral gray` avoids feeding either old masked content or a
black patch back through the time-axis reference.

Workflow:

```text
Checkpoint MODEL -> Anima T-Reference Repaint Route.model
Load Image.image -> Anima T-Reference Repaint Route.image
Load Image.mask  -> Anima T-Reference Repaint Route.mask
VAE              -> Anima T-Reference Repaint Route.vae

Anima T-Reference Repaint Route.model
-> Anima Flow Corrective Sampler.model

Anima T-Reference Repaint Route.latent
-> native KSampler.latent_image or Anima Flow Corrective Sampler.latent_image
```

Starting settings:

```text
mode: structure repaint
latent_fill: neutral gray
reference_fill: neutral gray
mask_grow: 24-48
mask_feather: 16-48
denoise: 0.85-1.00
cfg: 3.5-5.0
```

This is the route inspired by the discussion workflow that uses the model/UNet
forward path directly instead of training an inpaint ControlNet.

If you use native KSampler + VAEDecode, feed the decoded image, original source
image, and route `mask` output into `Anima Repaint Composite`. This keeps the
unmasked area from being softened by a full-image VAE roundtrip.

### Route B: Anima T-Reference Control Repaint Route

Use this route when you want to combine the time-axis reference trick with
Anima LLLite or another ControlNet-style node. This node intentionally has no
`MODEL` input so it can prepare the shared assets without creating a connection
cycle around external ControlNet nodes. The emitted `reference_latent` is
encoded from `reference_fill`, while `control_image` still follows
`control_fill`. This keeps LLLite's masked-black control input separate from
the time-axis reference, so black fill is not copied back through the model.

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

For native KSampler, replace the final sampler with:

```text
Route.model + Route.latent
-> native KSampler er_sde/simple
-> VAEDecode
-> Anima Repaint Composite.repaint_image

Load Image.image -> Anima Repaint Composite.source_image
Route.mask       -> Anima Repaint Composite.mask
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

For the final image output, connect the same VAE to
`Anima Flow Corrective Sampler.vae` and use the sampler's `image` output. Repaint
latents carry the original image and a feathered composite mask so the sampler
composites the decoded repaint back over the source image; this avoids showing
the full VAE-decoded source and reduces visible seams outside the mask.

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
-> fill masked area before encode
-> attach latent noise_mask
```

White mask pixels are repainted. Black mask pixels are kept.

Starting settings:

```text
steps: 18
cfg: 4.5
denoise: 0.55
flow_solver: flow_unipc2_x0
mask_grow: 32
mask_feather: 16
latent_fill: neutral gray
noise_seed: only used by latent_fill: latent noise
```

For light edge cleanup, reduce `denoise` to `0.35-0.55`. For structural
redraws such as hands, use `0.85-1.00` and mask the whole local structure.
Use `latent_fill: original` only when you want masked img2img behavior. Avoid
`masked black` unless you are explicitly testing black-fill artifacts.

### Anima Cosmos Repaint Prepare

Inputs:

- `image`: source image.
- `mask`: white is repaint area, black is keep area.
- `vae`: encodes the source or filled image into a sampler latent.
- `mode`: `upscale clean`, `edge repair`, or `structure repaint`.
- `mask_threshold`: `0` selects any nonzero mask pixel; higher values require
  mask opacity greater than or equal to the threshold.
- `mask_grow` / `mask_feather`: expand the sampler repaint region, then soften
  only the preview/final composite edge. The output `mask` remains hard for
  ControlNet/LLLite mask inputs.
- `latent_fill`: how the sampler latent is initialized inside the mask.
- `noise_seed`: deterministic seed for `latent_fill: latent noise`.
- `control_fill`: how the control/reference image is initialized inside the mask.

Outputs:

- `latent`: encoded latent with a hard grown `noise_mask` already attached.
- `control_image`: masked image for Anima LLLite or reference workflows.
- `mask`: hard grown mask for ControlNet/LLLite mask inputs.
- `mask_preview`: red overlay preview.
- `log`: recommended sampler settings for the selected mode.

### Anima Repaint Composite

Use this after native KSampler + VAEDecode repaint routes. Native VAE decode
returns a full decoded image, so even unchanged areas can look softer than the
source. This node composites only the masked region over the original image.

Inputs:

- `source_image`: original image.
- `repaint_image`: image decoded from the sampler output.
- `mask`: hard processed mask from an Anima repaint route, or the original mask.
- `mask_grow`: leave at `0` when using a route output mask, since it is already
  grown.
- `mask_feather`: softens only the final composite edge.

Recommended starting settings:

```text
upscale clean:      steps 16, cfg 4.5, denoise 0.30
edge repair:        steps 18, cfg 4.3, denoise 0.55
structure repaint:  steps 22, cfg 4.0, denoise 0.90
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
denoise: 0.35-0.55
```

Use Route B with LLLite/ControlNet inpaint when structure is actually wrong:

```text
target: rebuild hands, fingers, clothing, local objects
mask: whole anatomical/object region
denoise: 0.85-1.00
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
