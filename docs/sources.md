# Sources

Checked: 2026-05-26

## Anima

- Anima model card:
  https://huggingface.co/circlestone-labs/Anima

Key facts used:

- 2B anime / illustration-focused text-to-image model.
- Native ComfyUI file placement.
- Prompting guidance, tag order, sampler settings, limitations, and license.
- Anima is described as a derivative of
  `nvidia/Cosmos-Predict2-2B-Text2Image`.

## Cosmos-Predict2

- NVIDIA Cosmos-Predict2-2B-Text2Image model card:
  https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image
- NVIDIA Cosmos-Predict2 project page:
  https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2/
- NVIDIA Cosmos-Predict2 GitHub repository:
  https://github.com/nvidia-cosmos/cosmos-predict2

Key facts used:

- Cosmos-Predict2 model family and Text2Image variants.
- Latent diffusion transformer architecture: self-attention, cross-attention,
  feedforward layers, and adaptive layer normalization.
- Text conditioning through cross-attention.
- Input/output expectations, runtime notes, BF16/Linux/VRAM caveats.
- Reported GenEval benchmark summary and public limitations.
- Repository status: archived, with a pointer to Cosmos-Predict2.5.

## Cosmos-Predict2.5

- User-provided technical report PDF:
  `user_input/2511.00062v2.pdf`
- arXiv abstract page:
  https://arxiv.org/abs/2511.00062
- NVIDIA Cosmos-Predict2.5 repository:
  https://github.com/nvidia-cosmos/cosmos-predict2.5

Key facts used:

- Cosmos-Predict2.5 is described as a flow-matching model.
- The report defines the latent path as
  `x_t = (1 - t) x + t eps` and the velocity target as `v_t = eps - x`.
- Training timesteps are sampled from a logit-normal distribution.
- The report applies the shift
  `t_s = beta * t / (1 + (beta - 1) * t)`, with beta increasing from `1` at
  256p to `5` at 720p.
- The report states that 5% of training samples are explicitly drawn from the
  highest 2% of the noise distribution to reduce transition artifacts.
- The Predict2 scheduler code uses a rectified-flow rho/order schedule with
  `sigma_min=0.002`, `sigma_max=80.0`, and `order=7.0`.

## Karpathy-Inspired Agent Guidance

- Karpathy-inspired Claude Code guidelines repository:
  https://github.com/multica-ai/andrej-karpathy-skills

Key ideas used:

- Think before coding.
- Simplicity first.
- Surgical changes.
- Goal-driven execution with verification.
