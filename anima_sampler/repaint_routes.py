"""High-level repaint route nodes for Anima/Cosmos edit workflows."""

from __future__ import annotations

from .latent_utils import _shape_text
from .node_constants import NODE_CATEGORY
from .reference import AnimaCosmosReferenceLatent
from .repaint import (
    CONTROL_FILL_MODES,
    LATENT_FILL_MODES,
    REPAINT_MODES,
    AnimaCosmosRepaintPrepare,
    _encode_latent_image,
    _normalize_image_tensor,
)


class AnimaTReferenceRepaintRoute:
    """No-ControlNet route: source image as a time-axis reference frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "mode": (REPAINT_MODES, {"default": "structure repaint"}),
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
                    {"default": 48, "min": 0, "max": 512, "step": 1},
                ),
                "latent_fill": (LATENT_FILL_MODES, {"default": "latent noise"}),
                "noise_seed": (
                    "INT",
                    {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "latent", "mask", "mask_preview", "log")
    FUNCTION = "build"
    CATEGORY = NODE_CATEGORY

    def build(
        self,
        model,
        image,
        mask,
        vae,
        mode,
        mask_threshold,
        mask_grow,
        mask_feather,
        latent_fill,
        noise_seed,
        invert_mask,
    ):
        image = _normalize_image_tensor(image)
        latent, reference_image, processed_mask, mask_preview, prepare_log = (
            AnimaCosmosRepaintPrepare().prepare(
                image=image,
                mask=mask,
                vae=vae,
                mode=mode,
                mask_threshold=mask_threshold,
                mask_grow=mask_grow,
                mask_feather=mask_feather,
                latent_fill=latent_fill,
                noise_seed=noise_seed,
                control_fill="masked black",
                invert_mask=invert_mask,
            )
        )
        reference_latent = _reference_latent_from_image(vae, reference_image)
        patched_model, = AnimaCosmosReferenceLatent().apply(
            model=model,
            latent=reference_latent,
            enabled=True,
        )
        log = _route_log(
            route_name="AnimaTReferenceRepaintRoute",
            route="no-controlnet t-reference repaint",
            reference_latent=reference_latent,
            prepare_log=prepare_log,
            extra_lines=[
                "connect_model: route.model -> Anima Flow Corrective Sampler.model",
                "connect_latent: route.latent -> Anima Flow Corrective Sampler.latent_image",
                "controlnet: disabled",
            ],
        )
        return patched_model, latent, processed_mask, mask_preview, log


class AnimaTReferenceControlRepaintRoute:
    """ControlNet/LLLite route prep with a separate time-axis reference latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "vae": ("VAE",),
                "mode": (REPAINT_MODES, {"default": "structure repaint"}),
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
                    {"default": 48, "min": 0, "max": 512, "step": 1},
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

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = (
        "reference_latent",
        "latent",
        "control_image",
        "mask",
        "mask_preview",
        "log",
    )
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
        latent, control_image, processed_mask, mask_preview, prepare_log = (
            AnimaCosmosRepaintPrepare().prepare(
                image=_normalize_image_tensor(image),
                mask=mask,
                vae=vae,
                mode=mode,
                mask_threshold=mask_threshold,
                mask_grow=mask_grow,
                mask_feather=mask_feather,
                latent_fill=latent_fill,
                noise_seed=noise_seed,
                control_fill=control_fill,
                invert_mask=invert_mask,
            )
        )
        reference_latent = _reference_latent_from_image(vae, control_image)
        log = _route_log(
            route_name="AnimaTReferenceControlRepaintRoute",
            route="t-reference repaint with external ControlNet/LLLite",
            reference_latent=reference_latent,
            prepare_log=prepare_log,
            extra_lines=[
                "connect_control: control_image + mask -> Anima LLLite/ControlNet apply",
                (
                    "connect_reference: reference_latent -> Anima Cosmos Reference "
                    "Latent.latent after the control model/conditioning step"
                ),
                "connect_latent: route.latent -> Anima Flow Corrective Sampler.latent_image",
                "controlnet: external node required",
            ],
        )
        return reference_latent, latent, control_image, processed_mask, mask_preview, log


def _reference_latent_from_image(vae, image):
    return {"samples": _encode_latent_image(vae, image)}


def _route_log(*, route_name: str, route: str, reference_latent, prepare_log: str, extra_lines):
    samples = reference_latent.get("samples")
    return "\n\n".join(
        [
            route_name,
            "\n".join(
                [
                    f"route: {route}",
                    f"reference_latent_shape: {_shape_text(samples)}",
                    "reference_method: append masked source latent on Cosmos time axis during model apply",
                    "reference_masking: masked repaint area before encoding the time-axis reference",
                    *extra_lines,
                ]
            ),
            "prepare_log:",
            prepare_log,
        ]
    )
