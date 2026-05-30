"""Reference-latent patch nodes for Cosmos/Anima edit workflows."""

from __future__ import annotations

import types

from .node_constants import NODE_CATEGORY


class AnimaCosmosReferenceModelPatch:
    """Patch a Cosmos/Anima model so reference latents can be appended in time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY

    def patch(self, model):
        return (_patch_cosmos_reference_model(model),)


class AnimaCosmosReferenceLatent:
    """Attach one reference latent for no-ControlNet Cosmos-style image editing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = NODE_CATEGORY

    def apply(self, model, latent, enabled):
        if not enabled:
            return (model,)
        patched = _patch_cosmos_reference_model(model)
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent must contain a 'samples' tensor")

        samples = latent["samples"]
        model_obj = getattr(patched, "model", None)
        if hasattr(model_obj, "process_latent_in"):
            samples = model_obj.process_latent_in(samples)

        refs = list(getattr(patched, "model_options", {}).get("anima_ref_latents", []))
        refs.append(samples)
        patched.model_options["anima_ref_latents"] = refs
        if hasattr(patched, "add_object_patch"):
            patched.add_object_patch("anima_ref_latents", refs)
        else:
            setattr(model_obj, "anima_ref_latents", refs)
        return (patched,)


def _patch_cosmos_reference_model(model):
    patched = model.clone()
    model_options = getattr(patched, "model_options", None)
    if model_options is None:
        patched.model_options = {}
        model_options = patched.model_options
    if model_options.get("anima_cosmos_reference_patch_installed"):
        return patched

    model_obj = getattr(patched, "model", None)
    if model_obj is not None and hasattr(model_obj, "extra_conds"):
        original_extra_conds = model_obj.extra_conds

        def custom_extra_conds(self, **kwargs):
            out = original_extra_conds(**kwargs)
            ref_latents = kwargs.get("reference_latents", None)
            if ref_latents is not None:
                refs = ref_latents if isinstance(ref_latents, (list, tuple)) else [ref_latents]
                refs = [_process_reference_latent(self, ref) for ref in refs]
                out["ref_latents"] = _cond_list(refs)
            return out

        patched.add_object_patch("extra_conds", types.MethodType(custom_extra_conds, model_obj))

    previous_wrapper = model_options.get("model_function_wrapper")

    def reference_wrapper(model_apply, model_kwargs):
        refs = _collect_reference_latents(model_apply, model_kwargs)
        if not refs:
            return _call_model_wrapper(previous_wrapper, model_apply, model_kwargs)

        x = model_kwargs.get("input", None)
        if x is None or int(getattr(x, "ndim", len(x.shape))) != 5:
            return _call_model_wrapper(previous_wrapper, model_apply, model_kwargs)

        x_with_refs, original_frames = _append_reference_latents(x, refs)
        wrapped_kwargs = dict(model_kwargs)
        wrapped_kwargs["input"] = x_with_refs
        out = _call_model_wrapper(previous_wrapper, model_apply, wrapped_kwargs)
        if int(getattr(out, "ndim", len(out.shape))) == 5:
            out = out[:, :, :original_frames, :, :]
        return out

    patched.set_model_unet_function_wrapper(reference_wrapper)
    model_options["anima_cosmos_reference_patch_installed"] = True
    return patched


def _collect_reference_latents(model_apply, model_kwargs):
    cond = model_kwargs.get("c", {}) or {}
    refs = []
    from_cond = cond.get("ref_latents", [])
    if from_cond is not None:
        if isinstance(from_cond, (list, tuple)):
            refs.extend(list(from_cond))
        else:
            refs.append(from_cond)
    model_obj = getattr(model_apply, "__self__", None)
    refs.extend(list(getattr(model_obj, "anima_ref_latents", [])))
    return refs


def _append_reference_latents(x, refs):
    original_frames = int(x.shape[2])
    out = x
    for ref in refs:
        if not hasattr(ref, "shape"):
            continue
        if int(getattr(ref, "ndim", len(ref.shape))) == 4:
            ref = ref.unsqueeze(2)
        if int(getattr(ref, "ndim", len(ref.shape))) != 5:
            continue
        ref = ref.to(device=out.device, dtype=out.dtype)
        if int(ref.shape[0]) != int(out.shape[0]):
            if int(out.shape[0]) % int(ref.shape[0]) == 0:
                ref = ref.repeat(int(out.shape[0]) // int(ref.shape[0]), 1, 1, 1, 1)
            else:
                ref = ref.expand(int(out.shape[0]), -1, -1, -1, -1)
        out = _cat_time(out, ref)
    return out, original_frames


def _cat_time(x, ref):
    torch, _ = _torch_modules()
    return torch.cat([x, ref], dim=2)


def _call_model_wrapper(previous_wrapper, model_apply, model_kwargs):
    if previous_wrapper is not None:
        return previous_wrapper(model_apply, model_kwargs)
    kwargs = dict(model_kwargs)
    x = kwargs.pop("input")
    timestep = kwargs.pop("timestep")
    cond = kwargs.pop("c", {}) or {}
    return model_apply(x, timestep, **cond, **kwargs)


def _process_reference_latent(model_obj, latent):
    if hasattr(model_obj, "process_latent_in"):
        return model_obj.process_latent_in(latent)
    return latent


def _cond_list(items):
    try:
        import comfy.conds

        return comfy.conds.CONDList(items)
    except Exception:
        return items


def _torch_modules():
    import torch
    import torch.nn.functional as F

    return torch, F
