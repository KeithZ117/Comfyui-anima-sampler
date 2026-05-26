# Cosmos-Predict2-2B-Text2Image Notes

Checked: 2026-05-26

Primary sources:

- https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image
- https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2/
- https://github.com/nvidia-cosmos/cosmos-predict2

NVIDIA's archived Cosmos-Predict2 repository links "Paper (coming soon!)" rather
than a separate PDF. Until a standalone paper is available, treat the Hugging
Face model card, NVIDIA project page, and GitHub README as the public technical
reference set for this project.

## Model Family

- Cosmos-Predict2 is a family of world foundation models for Physical AI.
- The family includes text-to-image and video-to-world variants at 2B and 14B
  scales.
- Target base for this project: `nvidia/Cosmos-Predict2-2B-Text2Image`.
- NVIDIA later released Cosmos-Predict2.5 and recommends migration for general
  Cosmos users, but Anima remains a derivative of Cosmos-Predict2-2B-Text2Image.

## Architecture Facts

The `nvidia/Cosmos-Predict2-2B-Text2Image` model card describes the model as:

- a diffusion transformer for latent-space image denoising;
- built from interleaved self-attention, cross-attention, and feedforward
  layers;
- conditioned on text through cross-attention during denoising;
- using adaptive layer normalization before each layer to inject denoising time
  information.

Project implication: do not write UNet block hooks, SDXL attention assumptions,
or CLIP token offset logic unless verified from the local ComfyUI runtime.

## Input And Output

- Input type: text prompt.
- The model card recommends prompts under 300 words and expects descriptive
  scene content.
- Default output is listed as an RGB image at 1280x704.
- Official examples use Diffusers `Cosmos2TextToImagePipeline`.

Project implication: prompt phase scheduling should operate on complete prompt
strings or ComfyUI conditioning objects, not assumed CLIP tokens.

## Runtime Notes

- Officially listed runtime integrations include the Cosmos codebase and
  Diffusers.
- The model card lists PyTorch and Transformer Engine for inference.
- Only BF16 precision is called out as tested.
- Linux is listed as the tested OS.
- The 2B text-to-image card lists about 26 GB GPU VRAM for a single generation.

Project implication: ComfyUI custom nodes should avoid adding extra memory-heavy
decodes unless optional. Preview decoding should require an explicit VAE input
and should be clearly documented as memory-costly.

## Benchmarks And Limits

- The NVIDIA project page and model card report strong GenEval performance for
  Cosmos-Predict2 text-to-image compared with SDXL, DALL-E 3, and Flux 1-Dev.
- Reported strengths include object count, colors, position, and color
  attribution.
- The model card also states limits: high-resolution artifacts, unstable camera
  or object motion, imprecise interactions, and imperfect physical or spatial
  representation.
- The explainability subcard notes that text following can still fail.

Project implication: benchmark strength does not remove the need for early
trajectory inspection. The plugin should expose failure points instead of hiding
them behind a single monolithic final sample.

## Safe Hooking Policy

Preferred MVP controls:

- staged conditioning;
- segmented sampler calls;
- callbacks or intermediate latent checkpoints when supported;
- manual branch selection;
- mask-local latent re-noise.

Avoid in the MVP:

- self-attention modification;
- tokenizer-specific token weighting without inspected Qwen/Cosmos token maps;
- global monkey patches that affect other ComfyUI models;
- guardrail bypassing or safety filter changes.

