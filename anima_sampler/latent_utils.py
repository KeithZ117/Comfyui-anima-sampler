"""Latent shape, channel, and model-inspection helpers."""

from __future__ import annotations

from typing import Any

def _normalize_cosmos_latent(
    latent: dict[str, Any],
    *,
    expected_latent_channels: int | None = None,
) -> tuple[dict[str, Any], bool, str, str]:
    """Make image latents acceptable to Cosmos/Predict2 transformer code.

    Standard image latents are usually ``[B, C, H, W]``. Cosmos/Predict2 model
    code expects a temporal axis: ``[B, C, T, H, W]``. For text-to-image Anima,
    ``T=1`` is the natural image case.
    """

    samples = latent["samples"]
    ndim = int(getattr(samples, "ndim", len(samples.shape)))
    latent_in_shape = _shape_text(samples)
    channel_adapter = ""

    if ndim == 5:
        out = latent.copy()
        added_temporal_dim = False
    elif ndim == 4:
        out = latent.copy()
        out["samples"] = samples.unsqueeze(2)
        added_temporal_dim = True
    else:
        raise ValueError(
            "AnimaFlowCorrectiveSampler expected a 4D image latent or 5D "
            f"Cosmos latent, got shape {latent_in_shape}"
        )

    if expected_latent_channels is not None:
        out_samples = out["samples"]
        current_channels = int(out_samples.shape[1])
        if current_channels < expected_latent_channels:
            pad_channels = expected_latent_channels - current_channels
            padding_shape = list(out_samples.shape)
            padding_shape[1] = pad_channels
            padding = out_samples.new_zeros(padding_shape)
            out["samples"] = _torch_cat_like(out_samples, padding, dim=1)
            channel_adapter = (
                f"padded latent channels {current_channels}->{expected_latent_channels} "
                "before sampling"
            )
        elif current_channels > expected_latent_channels:
            raise ValueError(
                "AnimaFlowCorrectiveSampler received too many latent channels: "
                f"got {current_channels}, expected {expected_latent_channels}. "
                "Use the Anima/Qwen image latent path, not an incompatible VAE latent."
            )

    return out, added_temporal_dim, latent_in_shape, channel_adapter
def _infer_cosmos_latent_channels(
    x_embedder_features: int | None,
    current_channels: int | None,
) -> int | None:
    """Infer latent channel count from Cosmos patch embedding width.

    Anima's current T2I path uses a padding-mask channel before 2x2 patch
    embedding. The observed model has ``68 = (16 latent + 1 mask) * 2 * 2``.
    """

    if x_embedder_features is None:
        if current_channels == 4:
            return 16
        return None

    preferred_patch_volumes = (4, 1, 2, 8, 16)
    candidates: list[int] = []
    for patch_volume in preferred_patch_volumes:
        if x_embedder_features % patch_volume == 0:
            channels = x_embedder_features // patch_volume - 1
            if channels > 0:
                candidates.append(channels)

    if current_channels in candidates:
        return current_channels
    if 16 in candidates:
        return 16
    if candidates:
        return candidates[0]
    return None
def _latent_channel_count(samples: Any) -> int | None:
    shape = getattr(samples, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])
def _torch_cat_like(left, right, *, dim: int):
    import torch

    return torch.cat([left, right], dim=dim)
def _restore_image_latent(samples):
    if int(getattr(samples, "ndim", len(samples.shape))) == 5 and samples.shape[2] == 1:
        return samples.squeeze(2)
    return samples
def _shape_text(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "<unknown>"
    return "[" + ", ".join(str(int(dim)) for dim in shape) + "]"
def _restore_sampler_channels(denoised, sampler_state):
    """Return a denoised tensor compatible with the sampler state shape."""

    if tuple(denoised.shape) == tuple(sampler_state.shape):
        return denoised

    if (
        int(getattr(denoised, "ndim", len(denoised.shape)))
        == int(getattr(sampler_state, "ndim", len(sampler_state.shape)))
        and denoised.shape[0] == sampler_state.shape[0]
        and denoised.shape[2:] == sampler_state.shape[2:]
        and denoised.shape[1] >= sampler_state.shape[1]
    ):
        return denoised[:, : sampler_state.shape[1], ...]

    raise RuntimeError(
        "Anima sampler model output shape is incompatible with sampler state: "
        f"output={_shape_text(denoised)}, state={_shape_text(sampler_state)}"
    )
def _find_x_embedder_in_features(torch, model: Any) -> int | None:
    """Best-effort lookup of Cosmos/Predict2 x_embedder input width."""

    seen: set[int] = set()
    stack = [model]
    attr_names = (
        "inner_model",
        "diffusion_model",
        "model",
        "model_patcher",
        "patcher",
    )

    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if hasattr(obj, "x_embedder"):
            try:
                in_features = _first_linear_in_features(torch, getattr(obj, "x_embedder"))
            except Exception:
                in_features = None
            if in_features is not None:
                return in_features

        if isinstance(obj, (list, tuple, torch.nn.Sequential, torch.nn.ModuleList)):
            stack.extend(reversed(list(obj)))
            continue

        for attr_name in attr_names:
            if hasattr(obj, attr_name):
                try:
                    stack.append(getattr(obj, attr_name))
                except Exception:
                    pass

        if hasattr(obj, "_modules"):
            try:
                stack.extend(reversed(list(obj._modules.values())))
            except Exception:
                pass

    return None
def _first_linear_in_features(torch, root: Any) -> int | None:
    stack = [root]
    seen: set[int] = set()
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if isinstance(obj, torch.nn.Linear):
            return int(obj.in_features)

        in_features = getattr(obj, "in_features", None)
        if isinstance(in_features, int):
            return int(in_features)

        if isinstance(obj, (list, tuple, torch.nn.Sequential, torch.nn.ModuleList)):
            stack.extend(reversed(list(obj)))
            continue

        if hasattr(obj, "proj"):
            try:
                stack.append(getattr(obj, "proj"))
            except Exception:
                pass

        if hasattr(obj, "_modules"):
            try:
                stack.extend(reversed(list(obj._modules.values())))
            except Exception:
                pass

    return None
