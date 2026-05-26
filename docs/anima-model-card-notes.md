# Anima Model Card Notes

Checked: 2026-05-26

Primary source: https://huggingface.co/circlestone-labs/Anima

This file is a project-local summary of the public Anima model card. It is not a
copy of the model card.

## Identity

- Hugging Face ID: `circlestone-labs/Anima`
- Developer: CircleStone Labs, in collaboration with Comfy Org.
- Model type: 2B text-to-image model for anime, illustration, and other
  non-photorealistic art.
- Anima is described by its card as a derivative of
  `nvidia/Cosmos-Predict2-2B-Text2Image`.
- License: CircleStone Labs non-commercial license. The card also notes that
  NVIDIA Open Model License terms apply to the Cosmos-derived model lineage.

## Training And Intended Use

- Training data is described as several million anime images plus roughly
  800k non-anime artistic images.
- The model card says no synthetic data was used for training.
- Anime training data knowledge cutoff: September 2025.
- Intended output style: illustrations and artistic images.
- Known weak area: realism.

## ComfyUI Files

The model card says Anima is natively supported in ComfyUI and lists these model
placements:

- `ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors`
- `ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors`
- `ComfyUI/models/vae/qwen_image_vae.safetensors`

Project implication: custom nodes must treat the model as an Anima/Cosmos
workflow, not as SDXL, SD 1.5, CLIP, or a UNet workflow.

## Generation Settings From The Card

- Resolution range: roughly 512^2 to 1536^2 pixels.
- Suggested steps: 30-50.
- Suggested CFG: 4-5.
- Samplers called out by the card include `er_sde`, `euler_a`, and
  `dpmpp_2m_sde_gpu`, with different style tendencies.

Project implication: neutral segmented sampling should default to Anima-like
step and CFG ranges and should not assume SDXL default strengths.

## Prompting Notes

- The card says Anima was trained on Danbooru-style tags, natural-language
  captions, and mixtures of the two.
- Tags should generally be lowercase and use spaces instead of underscores,
  except score tags.
- The card recommends a positive prefix using quality and safety tags, and a
  negative prompt using low-quality score tags.
- Suggested tag order:
  quality/meta/year/safety, subject count, character, series, artist, general
  tags.
- Artist tags should use an `@` prefix.
- Prompt weighting works, but the card says higher weights than typical SDXL
  usage may be needed.
- The model was trained with random tag dropout, so prompts do not need to
  enumerate every relevant tag.
- For multiple characters, the card recommends naming the character and also
  describing basic appearance, because names alone can confuse the model.

Project implication: `AnimaPromptPhaseBuilder` should be prompt-text oriented
first. Token-level routing is future work unless the Qwen/Cosmos text pathway is
verified locally.

## Limitations From The Card

- Realism is not the target.
- Short or under-specified prompts can produce undesired content.
- Text rendering is limited.
- The base version is intentionally broad and not heavily aesthetic-tuned.

Project implication: the sampler stack should avoid pretending it can solve
semantic failures after the trajectory has already committed. It should expose
early checkpoints, branching, and local re-noise tools so users can intervene
earlier.

## Relevance To Error-Corrective Sampling

Useful first controls:

- Structure-heavy early phase: count, pose, body framing, camera, composition.
- Identity/object mid phase: character, hair, eyes, outfit, key props.
- Detail/style late phase: lineart, shading, quality, artist/style tags.
- Early previews for 10-35 percent progress points.
- Branch-and-prune before full denoising commits to a wrong layout.
- Masked local re-noise for late repair of faces, hands, props, or wrong body
  parts.

