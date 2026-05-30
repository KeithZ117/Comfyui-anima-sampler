"""ComfyUI sampling bridge for Anima Flow samplers."""

from __future__ import annotations

import importlib
from typing import Any

from .cfg_schedule import cfg_at_progress
from .flow_constants import FLOW_SOLVERS
from .flow_math import _active_integration_steps
from .latent_utils import _find_x_embedder_in_features, _latent_channel_count, _shape_text
from .sampler_trace import _stats_mean, _stats_percentile, format_sampler_trace_csv
from .sampler_types import AnimaSamplerLog
from .sampling_loop import sample_anima_flow_corrective
from .sigma_schedule import _describe_model_sampling_shift, build_anima_sigmas

def run_comfy_anima_sampler(
    *,
    model: Any,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    flow_solver: str,
    flow_schedule: str,
    flow_shift: float,
    flow_rho7_tail_auto: bool,
    final_clean_pass: bool,
    flow_er_order: int,
    flow_pc3_gamma: float,
    flow_pc3_tolerance: float,
    flow_unipc_order: int,
    flow_unipc_solver_type: str,
    flow_unipc_lower_order_final: bool,
    flow_unipc_disable_corrector_first: int,
    flow_unipc_thresholding: bool,
    flow_unipc_dynamic_thresholding_ratio: float,
    flow_unipc_sample_max_value: float,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
    denoise_legacy_progress: bool,
    cfg_schedule_domain: str,
    cfg_schedule_mode: str,
    early_cfg_boost: float,
    early_cfg_until: float,
    late_cfg_scale: float,
    late_cfg_start: float,
    cfg_early_scale: float,
    cfg_early_ramp_end: float,
    cfg_peak_boost: float,
    cfg_bump_start: float,
    cfg_bump_end: float,
    cfg_beta_alpha: float,
    cfg_beta_beta: float,
    cfg_interval_start: float,
    cfg_interval_rise_end: float,
    cfg_interval_fall_start: float,
    cfg_interval_end: float,
    rf_endpoint_noise_refresh_enabled: bool,
    rf_endpoint_noise_refresh_strength: float,
    rf_endpoint_noise_refresh_until: float,
    add_noise: bool,
    disable_pbar: bool,
    collect_diagnostics: bool = False,
) -> tuple[dict[str, Any], str] | tuple[dict[str, Any], str, str]:
    """Run the sampler through ComfyUI's standard sampling bridge."""

    if "samples" not in latent:
        raise ValueError("latent input must contain a 'samples' tensor")
    if flow_solver not in FLOW_SOLVERS:
        raise ValueError(f"unsupported flow_solver: {flow_solver}")

    torch = importlib.import_module("torch")
    comfy_sample = importlib.import_module("comfy.sample")
    comfy_samplers = importlib.import_module("comfy.samplers")
    comfy_utils = importlib.import_module("comfy.utils")
    latent_preview = importlib.import_module("latent_preview")

    latent_in_shape = _shape_text(latent["samples"])
    latent_image = comfy_sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    current_channels = _latent_channel_count(latent_image)
    x_embedder_features = _find_x_embedder_in_features(torch, model)
    added_temporal_dim = len(latent["samples"].shape) == 4 and len(latent_image.shape) == 5
    channel_adapter = ""
    if _latent_channel_count(latent["samples"]) != current_channels:
        channel_adapter = (
            f"fixed latent channels {_latent_channel_count(latent['samples'])}->{current_channels} "
            "with ComfyUI model latent_format"
        )
        print(f"[comfyui-anima-sampler] {channel_adapter}")

    batch_index = latent.get("batch_index")
    if add_noise:
        noise = comfy_sample.prepare_noise(latent_image, seed, batch_index)
    else:
        noise = torch.zeros_like(latent_image)

    schedule_steps = int(steps) + int(bool(final_clean_pass))
    sigmas = build_anima_sigmas(
        model,
        schedule_steps,
        denoise=denoise,
        flow_schedule=flow_schedule,
        flow_shift=flow_shift,
        flow_rho7_tail_auto=flow_rho7_tail_auto,
        cosmos_sigma_max=cosmos_sigma_max,
        cosmos_sigma_min=cosmos_sigma_min,
        denoise_legacy_progress=denoise_legacy_progress,
    )

    if not hasattr(comfy_samplers, "KSAMPLER"):
        raise RuntimeError("This ComfyUI build does not expose comfy.samplers.KSAMPLER")
    if not hasattr(comfy_samplers, "sample"):
        raise RuntimeError("This ComfyUI build does not expose comfy.samplers.sample")

    sampler_stats: dict[str, Any] = {}
    sampler = comfy_samplers.KSAMPLER(
        sample_anima_flow_corrective,
        extra_options={
            "flow_solver": flow_solver,
            "flow_schedule": flow_schedule,
            "flow_shift": float(flow_shift),
            "flow_rho7_tail_auto": bool(flow_rho7_tail_auto),
            "final_clean_pass": bool(final_clean_pass),
            "flow_er_order": int(flow_er_order),
            "flow_pc3_gamma": float(flow_pc3_gamma),
            "flow_pc3_tolerance": float(flow_pc3_tolerance),
            "flow_unipc_order": int(flow_unipc_order),
            "flow_unipc_solver_type": str(flow_unipc_solver_type),
            "flow_unipc_lower_order_final": bool(flow_unipc_lower_order_final),
            "flow_unipc_disable_corrector_first": int(flow_unipc_disable_corrector_first),
            "flow_unipc_thresholding": bool(flow_unipc_thresholding),
            "flow_unipc_dynamic_thresholding_ratio": float(flow_unipc_dynamic_thresholding_ratio),
            "flow_unipc_sample_max_value": float(flow_unipc_sample_max_value),
            "base_cfg": float(cfg),
            "cfg_schedule_domain": str(cfg_schedule_domain),
            "cfg_schedule_mode": str(cfg_schedule_mode),
            "early_cfg_boost": float(early_cfg_boost),
            "early_cfg_until": float(early_cfg_until),
            "late_cfg_scale": float(late_cfg_scale),
            "late_cfg_start": float(late_cfg_start),
            "cfg_early_scale": float(cfg_early_scale),
            "cfg_early_ramp_end": float(cfg_early_ramp_end),
            "cfg_peak_boost": float(cfg_peak_boost),
            "cfg_bump_start": float(cfg_bump_start),
            "cfg_bump_end": float(cfg_bump_end),
            "cfg_beta_alpha": float(cfg_beta_alpha),
            "cfg_beta_beta": float(cfg_beta_beta),
            "cfg_interval_start": float(cfg_interval_start),
            "cfg_interval_rise_end": float(cfg_interval_rise_end),
            "cfg_interval_fall_start": float(cfg_interval_fall_start),
            "cfg_interval_end": float(cfg_interval_end),
            "rf_endpoint_noise_refresh_enabled": bool(rf_endpoint_noise_refresh_enabled),
            "rf_endpoint_noise_refresh_strength": float(rf_endpoint_noise_refresh_strength),
            "rf_endpoint_noise_refresh_until": float(rf_endpoint_noise_refresh_until),
            "sampler_stats": sampler_stats,
            "collect_diagnostics": bool(collect_diagnostics),
        },
    )

    active_steps = _active_integration_steps(torch, sigmas, final_clean_pass=final_clean_pass)
    callback = latent_preview.prepare_callback(model, active_steps)
    progress_disabled = disable_pbar or not comfy_utils.PROGRESS_BAR_ENABLED

    noise_mask = latent.get("noise_mask")
    samples = comfy_sample.sample_custom(
        model,
        noise,
        cfg,
        sampler,
        sigmas,
        positive,
        negative,
        latent_image,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=progress_disabled,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples

    actual_steps = active_steps
    log = AnimaSamplerLog(
        requested_steps=steps,
        actual_steps=actual_steps,
        latent_in_shape=latent_in_shape,
        latent_sample_shape=_shape_text(latent_image),
        added_temporal_dim=added_temporal_dim,
        channel_adapter=channel_adapter or "none",
        x_embedder_features=str(x_embedder_features or "unknown"),
        sampler_core=flow_solver,
        flow_schedule=flow_schedule,
        flow_shift=float(flow_shift),
        flow_rho7_tail_auto=bool(flow_rho7_tail_auto),
        final_clean_pass=bool(final_clean_pass),
        cfg_schedule_mode=str(cfg_schedule_mode),
        cfg_schedule_domain=str(cfg_schedule_domain),
        denoise_legacy_progress=bool(denoise_legacy_progress),
        model_sampling_shift=_describe_model_sampling_shift(model, flow_schedule=flow_schedule),
        denoise=denoise,
        flow_er_order=int(flow_er_order),
        flow_pc3_gamma=float(flow_pc3_gamma),
        flow_pc3_tolerance=float(flow_pc3_tolerance),
        cosmos_sigma_max=cosmos_sigma_max,
        cosmos_sigma_min=cosmos_sigma_min,
        cfg_start=cfg_at_progress(
            0.0,
            base_cfg=cfg,
            cfg_schedule_mode=cfg_schedule_mode,
            early_cfg_boost=early_cfg_boost,
            early_cfg_until=early_cfg_until,
            late_cfg_scale=late_cfg_scale,
            late_cfg_start=late_cfg_start,
            cfg_early_scale=cfg_early_scale,
            cfg_early_ramp_end=cfg_early_ramp_end,
            cfg_peak_boost=cfg_peak_boost,
            cfg_bump_start=cfg_bump_start,
            cfg_bump_end=cfg_bump_end,
            cfg_beta_alpha=cfg_beta_alpha,
            cfg_beta_beta=cfg_beta_beta,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_rise_end=cfg_interval_rise_end,
            cfg_interval_fall_start=cfg_interval_fall_start,
            cfg_interval_end=cfg_interval_end,
        ),
        cfg_mid=cfg_at_progress(
            0.5,
            base_cfg=cfg,
            cfg_schedule_mode=cfg_schedule_mode,
            early_cfg_boost=early_cfg_boost,
            early_cfg_until=early_cfg_until,
            late_cfg_scale=late_cfg_scale,
            late_cfg_start=late_cfg_start,
            cfg_early_scale=cfg_early_scale,
            cfg_early_ramp_end=cfg_early_ramp_end,
            cfg_peak_boost=cfg_peak_boost,
            cfg_bump_start=cfg_bump_start,
            cfg_bump_end=cfg_bump_end,
            cfg_beta_alpha=cfg_beta_alpha,
            cfg_beta_beta=cfg_beta_beta,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_rise_end=cfg_interval_rise_end,
            cfg_interval_fall_start=cfg_interval_fall_start,
            cfg_interval_end=cfg_interval_end,
        ),
        cfg_end=cfg_at_progress(
            1.0,
            base_cfg=cfg,
            cfg_schedule_mode=cfg_schedule_mode,
            early_cfg_boost=early_cfg_boost,
            early_cfg_until=early_cfg_until,
            late_cfg_scale=late_cfg_scale,
            late_cfg_start=late_cfg_start,
            cfg_early_scale=cfg_early_scale,
            cfg_early_ramp_end=cfg_early_ramp_end,
            cfg_peak_boost=cfg_peak_boost,
            cfg_bump_start=cfg_bump_start,
            cfg_bump_end=cfg_bump_end,
            cfg_beta_alpha=cfg_beta_alpha,
            cfg_beta_beta=cfg_beta_beta,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_rise_end=cfg_interval_rise_end,
            cfg_interval_fall_start=cfg_interval_fall_start,
            cfg_interval_end=cfg_interval_end,
        ),
        rf_endpoint_noise_refresh_enabled=bool(rf_endpoint_noise_refresh_enabled),
        rf_endpoint_noise_refresh_strength=float(rf_endpoint_noise_refresh_strength),
        rf_endpoint_noise_refresh_until=float(rf_endpoint_noise_refresh_until),
        actual_model_calls=sampler_stats.get("model_calls"),
        cache_candidates=int(sampler_stats.get("cache_candidates", 0)),
        cache_accepts=int(sampler_stats.get("cache_accepts", 0)),
        cache_rejects=int(sampler_stats.get("cache_rejects", 0)),
        forced_refresh_count=int(sampler_stats.get("forced_refresh_count", 0)),
        pc3_used_total=int(sampler_stats.get("pc3_used_total", 0)),
        pc3_used_high=int(sampler_stats.get("pc3_used_high", 0)),
        pc3_used_body=int(sampler_stats.get("pc3_used_body", 0)),
        pc3_used_tail=int(sampler_stats.get("pc3_used_tail", 0)),
        mean_cache_score=_stats_mean(sampler_stats.get("cache_scores", [])),
        p95_cache_score=_stats_percentile(sampler_stats.get("cache_scores", []), 95.0),
        mean_gamma_pc3=_stats_mean(sampler_stats.get("gamma_pc3_values", [])),
        mean_gamma3=_stats_mean(sampler_stats.get("gamma3_values", [])),
    ).as_text()
    if collect_diagnostics:
        return out, log, format_sampler_trace_csv(sampler_stats)
    return out, log
def run_comfy_native_sampler(
    *,
    model: Any,
    positive: Any,
    negative: Any,
    latent: dict[str, Any],
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    sampler_name: str,
    scheduler: str,
    add_noise: bool,
    disable_pbar: bool,
) -> tuple[dict[str, Any], str]:
    """Run a stock ComfyUI sampler/scheduler pair for experiment baselines."""

    if "samples" not in latent:
        raise ValueError("latent input must contain a 'samples' tensor")

    torch = importlib.import_module("torch")
    comfy_sample = importlib.import_module("comfy.sample")
    comfy_samplers = importlib.import_module("comfy.samplers")
    comfy_utils = importlib.import_module("comfy.utils")
    latent_preview = importlib.import_module("latent_preview")

    if hasattr(comfy_samplers, "SAMPLER_NAMES") and sampler_name not in comfy_samplers.SAMPLER_NAMES:
        raise ValueError(f"ComfyUI sampler is not available: {sampler_name}")
    if hasattr(comfy_samplers, "SCHEDULER_NAMES") and scheduler not in comfy_samplers.SCHEDULER_NAMES:
        raise ValueError(f"ComfyUI scheduler is not available: {scheduler}")

    latent_in_shape = _shape_text(latent["samples"])
    latent_image = comfy_sample.fix_empty_latent_channels(
        model,
        latent["samples"],
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    channel_adapter = ""
    if _latent_channel_count(latent["samples"]) != _latent_channel_count(latent_image):
        channel_adapter = (
            f"fixed latent channels {_latent_channel_count(latent['samples'])}->"
            f"{_latent_channel_count(latent_image)} with ComfyUI model latent_format"
        )
        print(f"[comfyui-anima-sampler] {channel_adapter}")

    batch_index = latent.get("batch_index")
    if add_noise:
        noise = comfy_sample.prepare_noise(latent_image, seed, batch_index)
    else:
        noise = torch.zeros_like(latent_image)

    callback = latent_preview.prepare_callback(model, int(steps))
    progress_disabled = disable_pbar or not comfy_utils.PROGRESS_BAR_ENABLED
    noise_mask = latent.get("noise_mask")

    samples = comfy_sample.sample(
        model,
        noise,
        int(steps),
        float(cfg),
        str(sampler_name),
        str(scheduler),
        positive,
        negative,
        latent_image,
        denoise=float(denoise),
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=progress_disabled,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples

    log = "\n".join(
        [
            "ComfyNativeSampler",
            f"sampler: {sampler_name}",
            f"scheduler: {scheduler}",
            f"steps: {int(steps)}",
            "steps_semantics: ComfyUI native denoise steps",
            "estimated_model_calls: native sampler dependent",
            f"cfg: {float(cfg):.4f}",
            f"denoise: {float(denoise):.4f}",
            f"latent_in_shape: {latent_in_shape}",
            f"latent_sample_shape: {_shape_text(latent_image)}",
            f"channel_adapter: {channel_adapter or 'none'}",
            f"model_sampling_shift: {_describe_model_sampling_shift(model, flow_schedule='native')}",
            f"add_noise: {bool(add_noise)}",
        ]
    )
    return out, log
