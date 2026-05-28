# ComfyUI Anima Flow Corrective Sampler

Custom ComfyUI sampler nodes for Anima / Cosmos-style rectified-flow image
models. The default profile packages the current tested Anima workflow:

```text
solver        = flow_pc3_damped
schedule      = flow_cosmos
flow_shift    = 5.0
steps         = 35
cfg           = 6.0
cfg_mode      = bump cfg
```

The goal is to improve prompt structure, spatial relationships, and detail
stability while keeping the node surface small enough for daily use.

## Nodes

- `Anima Flow Corrective Sampler`: the main sampler node.
- `Anima Flow Settings`: optional advanced controls for solver and CFG tuning.

The sampler works without connecting `Anima Flow Settings`; the tested defaults
are built in. Connect the settings node only when you want to tune advanced
parameters.

The sampler outputs both `LATENT` and `IMAGE`. Connect a `VAE` to the optional
`vae` input when you want the image output decoded directly from the sampler.
Leave `vae` disconnected when you only need the latent output.

## Install

Clone or copy this repository into ComfyUI's `custom_nodes` directory:

```text
ComfyUI/custom_nodes/Comfyui-anima-sampler
```

Then restart ComfyUI.

This repository does not include model weights. Install Anima and its required
text encoder / VAE files according to the official Anima model card:

- https://huggingface.co/circlestone-labs/Anima

## Recommended Use

Use `Anima Flow Corrective Sampler` in place of a normal sampler node.

Everyday controls:

- `steps`: default `35`
- `cfg`: default `6.0`
- `cfg_mode`: `bump cfg` or `const`
- `flow_solver`: default `flow_pc3_damped`
- `flow_schedule`: default `flow_cosmos`
- `flow_shift`: default `5.0`
- `denoise`
- `add_noise`
- optional `vae` for direct image output

`flow_shift` affects `flow_cosmos`. Set it to `1.0` for no extra shift. The
`flow_cosmos_rho7_rf_tail_auto` schedule ignores `flow_shift` by design.

## Current Default

The packaged default is based on matched-seed testing where
`flow_pc3_damped + flow_cosmos + flow_shift 5.0 + bump cfg` gave the best
balance of semantic adherence, complex relationship handling, and low-noise
detail stability.

`flow_cosmos_rho7_rf_tail_auto` remains available as a quality baseline.
`simple` remains available for comparison with native ComfyUI-style schedules.

## Development

Run the local test suite from the repository root:

```text
python -m unittest discover -s tests
```

## License

This sampler code is released under the MIT License. See `LICENSE`.

Model weights are not included and are not relicensed by this project. The
official Anima model card currently lists Anima under the CircleStone Labs
Non-Commercial License and notes that it is also subject to the NVIDIA Open
Model License Agreement for derivative Cosmos models.
