"""Cosmos-style repaint preparation nodes for Anima."""

from __future__ import annotations

from .flow_math import _make_generator, _randn_like
from .latent_utils import _shape_text
from .node_constants import NODE_CATEGORY

REPAINT_MODES = ["upscale clean", "edge repair", "structure repaint"]
LATENT_FILL_MODES = ["latent noise", "original", "masked black", "neutral gray"]
CONTROL_FILL_MODES = ["masked black", "neutral gray", "blurred reference"]


class AnimaInpaintLatentPrepare:
    """Prepare an inpaint latent for Anima Flow Corrective Sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_grow": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 1},
                ),
                "mask_feather": (
                    "INT",
                    {"default": 16, "min": 0, "max": 512, "step": 1},
                ),
                "latent_fill": (LATENT_FILL_MODES, {"default": "latent noise"}),
                "noise_seed": (
                    "INT",
                    {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "mask", "mask_preview", "log")
    FUNCTION = "prepare"
    CATEGORY = NODE_CATEGORY

    def prepare(
        self,
        image,
        mask,
        vae,
        mask_threshold,
        mask_grow,
        mask_feather,
        latent_fill,
        noise_seed,
        invert_mask,
    ):
        latent, _control_image, processed_mask, mask_preview, prep_log = (
            AnimaCosmosRepaintPrepare().prepare(
                image=image,
                mask=mask,
                vae=vae,
                mode="structure repaint",
                mask_threshold=mask_threshold,
                mask_grow=mask_grow,
                mask_feather=mask_feather,
                latent_fill=latent_fill,
                noise_seed=noise_seed,
                control_fill="masked black",
                invert_mask=invert_mask,
            )
        )
        log = "\n\n".join(
            [
                "AnimaInpaintLatentPrepare",
                "workflow: image+mask -> VAE encode -> masked latent noise -> noise_mask",
                "next_node: connect latent to Anima Flow Corrective Sampler latent_image",
                "controlnet: disabled",
                "prepare_log:",
                prep_log,
            ]
        )
        return latent, processed_mask, mask_preview, log


class AnimaCosmosRepaintPrepare:
    """Prepare a Cosmos-style masked repaint latent and optional control image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "mode": (REPAINT_MODES, {"default": "upscale clean"}),
                "mask_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "mask_grow": (
                    "INT",
                    {"default": 32, "min": 0, "max": 512, "step": 1},
                ),
                "mask_feather": (
                    "INT",
                    {"default": 16, "min": 0, "max": 512, "step": 1},
                ),
                "latent_fill": (LATENT_FILL_MODES, {"default": "latent noise"}),
                "noise_seed": (
                    "INT",
                    {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "control_fill": (CONTROL_FILL_MODES, {"default": "masked black"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "control_image", "mask", "mask_preview", "log")
    FUNCTION = "prepare"
    CATEGORY = NODE_CATEGORY

    def prepare(
        self,
        image,
        mask,
        vae,
        mode,
        mask_threshold,
        mask_grow,
        mask_feather,
        latent_fill,
        noise_seed,
        control_fill,
        invert_mask,
    ):
        image = _normalize_image_tensor(image)
        latent_fill_mode = str(latent_fill)
        hard_mask = _normalize_mask_tensor(
            mask,
            image,
            threshold=mask_threshold,
            invert=invert_mask,
        )
        grown_mask = _grow_mask(hard_mask, int(mask_grow))
        soft_mask = _feather_mask(grown_mask, int(mask_feather))

        if _normalized_fill_mode(latent_fill_mode) == "latent noise":
            latent_pixels = image
        else:
            latent_pixels = _fill_masked_image(image, soft_mask, latent_fill_mode)
        control_image = _fill_masked_image(image, soft_mask, str(control_fill))
        samples = _encode_latent_image(vae, latent_pixels)
        noise_mask = _resize_mask_to_latent(soft_mask, samples)
        samples = _fill_masked_latent(samples, noise_mask, latent_fill_mode, int(noise_seed))
        mask_preview = _build_mask_preview(image, soft_mask)

        latent = {"samples": samples, "noise_mask": noise_mask}
        profile = _repaint_profile(mode)
        log = "\n".join(
            [
                "AnimaCosmosRepaintPrepare",
                f"mode: {mode}",
                f"recommended_steps: {profile['steps']}",
                f"recommended_cfg: {profile['cfg']:.2f}",
                f"recommended_denoise: {profile['denoise']:.2f}",
                f"mask_threshold: {float(mask_threshold):.3f}",
                f"mask_grow: {int(mask_grow)}",
                f"mask_feather: {int(mask_feather)}",
                f"latent_fill: {latent_fill}",
                f"noise_seed: {int(noise_seed)}",
                f"control_fill: {control_fill}",
                f"invert_mask: {bool(invert_mask)}",
                f"image_shape: {_shape_text(image)}",
                f"latent_shape: {_shape_text(samples)}",
                f"noise_mask_shape: {_shape_text(noise_mask)}",
                "noise_mask: white/1.0 is repaint area; black/0.0 is keep area",
                "mask_threshold: 0 selects any nonzero mask; higher values require mask >= threshold",
                (
                    "control_image: feed to Anima LLLite inpaint/any-test nodes "
                    "or to a reference-latent edit path"
                ),
            ]
        )
        return latent, control_image, soft_mask, mask_preview, log


def _encode_latent_image(vae, image):
    if vae is None or not hasattr(vae, "encode"):
        raise ValueError("vae must provide an encode(image) method")
    if image.shape[-1] > 3:
        image = image[..., :3]
    encoded = vae.encode(image)
    if isinstance(encoded, dict):
        if "samples" not in encoded:
            raise ValueError("VAE encode dict must contain a 'samples' tensor")
        encoded = encoded["samples"]
    ndim = int(getattr(encoded, "ndim", len(encoded.shape)))
    if ndim not in {4, 5}:
        raise ValueError("VAE encode must return a 4D or 5D latent tensor")
    return encoded


def _fill_masked_latent(samples, noise_mask, mode: str, seed: int):
    mode = _normalized_fill_mode(mode)
    if mode != "latent noise":
        return samples
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("noise_seed must be in the range [0, 2^64 - 1]")

    torch, _ = _torch_modules()
    generator = _make_generator(torch, samples.device, int(seed))
    noise = _randn_like(torch, samples, generator)
    blend_mask = _latent_blend_mask(noise_mask, samples)
    return samples * (1.0 - blend_mask) + noise * blend_mask


def _latent_blend_mask(noise_mask, samples):
    sample_ndim = int(getattr(samples, "ndim", len(samples.shape)))
    mask = noise_mask.to(device=samples.device, dtype=samples.dtype)
    if sample_ndim == 4:
        return mask
    if sample_ndim == 5:
        return mask.unsqueeze(2)
    raise ValueError("latent samples must be 4D or 5D")


def _normalize_image_tensor(image):
    ndim = int(getattr(image, "ndim", len(image.shape)))
    if ndim != 4:
        raise ValueError("image must be a 4D ComfyUI IMAGE tensor [B,H,W,C]")
    if int(image.shape[-1]) < 3:
        raise ValueError("image must have at least 3 channels")
    return image.clamp(0.0, 1.0)


def _normalize_mask_tensor(mask, image, *, threshold, invert):
    torch, _ = _torch_modules()
    ndim = int(getattr(mask, "ndim", len(mask.shape)))
    if ndim == 2:
        mask = mask.unsqueeze(0)
    elif ndim == 4:
        if int(mask.shape[-1]) == 1:
            mask = mask[..., 0]
        elif int(mask.shape[1]) == 1:
            mask = mask[:, 0]
        else:
            raise ValueError("4D mask must have a singleton channel dimension")
    elif ndim != 3:
        raise ValueError("mask must be a 2D, 3D, or single-channel 4D tensor")

    image_b, image_h, image_w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    if int(mask.shape[0]) == 1 and image_b > 1:
        mask = mask.repeat(image_b, 1, 1)
    if int(mask.shape[0]) != image_b:
        raise ValueError("mask batch must match image batch or be a single mask")
    if int(mask.shape[-2]) != image_h or int(mask.shape[-1]) != image_w:
        _, F = _torch_modules()
        mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=(image_h, image_w),
            mode="nearest",
        ).squeeze(1)

    mask = mask.to(device=image.device, dtype=image.dtype).clamp(0.0, 1.0)
    if invert:
        mask = 1.0 - mask
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("mask_threshold must be in the range [0, 1]")
    if threshold <= 0.0:
        return torch.where(mask > 0.0, torch.ones_like(mask), torch.zeros_like(mask))
    return torch.where(mask >= threshold, torch.ones_like(mask), torch.zeros_like(mask))


def _grow_mask(mask, radius: int):
    if radius <= 0:
        return mask
    _, F = _torch_modules()
    kernel = int(radius) * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel, stride=1, padding=int(radius)).squeeze(1)


def _feather_mask(mask, radius: int):
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    torch, F = _torch_modules()
    kernel = int(radius) * 2 + 1
    soft = F.avg_pool2d(
        mask.unsqueeze(1),
        kernel,
        stride=1,
        padding=int(radius),
        count_include_pad=False,
    ).squeeze(1)
    return torch.maximum(mask, soft).clamp(0.0, 1.0)


def _fill_masked_image(image, mask, mode: str):
    mode = _normalized_fill_mode(mode)
    if mode == "original":
        return image
    if mode == "latent noise":
        return image
    mask4 = mask.unsqueeze(-1).to(dtype=image.dtype, device=image.device)
    if mode == "masked black":
        fill = image.new_zeros(image.shape)
    elif mode == "neutral gray":
        fill = image.new_full(image.shape, 0.5)
    elif mode == "blurred reference":
        fill = _box_blur_image(image, radius=16)
    else:
        allowed = ", ".join(sorted(set(LATENT_FILL_MODES + CONTROL_FILL_MODES)))
        raise ValueError(f"fill mode must be one of: {allowed}")
    return (image * (1.0 - mask4) + fill * mask4).clamp(0.0, 1.0)


def _normalized_fill_mode(mode: str) -> str:
    return str(mode).strip().lower()


def _box_blur_image(image, *, radius: int):
    if radius <= 0:
        return image
    _, F = _torch_modules()
    kernel = int(radius) * 2 + 1
    channels_first = image.movedim(-1, 1)
    blurred = F.avg_pool2d(
        channels_first,
        kernel,
        stride=1,
        padding=int(radius),
        count_include_pad=False,
    )
    return blurred.movedim(1, -1).clamp(0.0, 1.0)


def _resize_mask_to_latent(mask, samples):
    _, F = _torch_modules()
    ndim = int(getattr(samples, "ndim", len(samples.shape)))
    if ndim in {4, 5}:
        latent_h, latent_w = int(samples.shape[-2]), int(samples.shape[-1])
    else:
        raise ValueError("latent samples must be 4D or 5D")
    resized = F.interpolate(
        mask.unsqueeze(1).float(),
        size=(latent_h, latent_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.to(device=samples.device, dtype=samples.dtype).clamp(0.0, 1.0)


def _build_mask_preview(image, mask):
    overlay = image.new_zeros(image.shape)
    overlay[..., 0] = 1.0
    alpha = (mask.unsqueeze(-1) * 0.55).to(dtype=image.dtype, device=image.device)
    return (image * (1.0 - alpha) + overlay * alpha).clamp(0.0, 1.0)


def _repaint_profile(mode: str) -> dict:
    mode = str(mode).strip().lower()
    profiles = {
        "upscale clean": {"steps": 16, "cfg": 4.5, "denoise": 0.22},
        "edge repair": {"steps": 16, "cfg": 4.5, "denoise": 0.32},
        "structure repaint": {"steps": 22, "cfg": 4.2, "denoise": 0.52},
    }
    if mode not in profiles:
        allowed = ", ".join(REPAINT_MODES)
        raise ValueError(f"mode must be one of: {allowed}")
    return profiles[mode]


def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F
