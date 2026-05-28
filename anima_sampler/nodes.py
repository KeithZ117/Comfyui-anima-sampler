"""ComfyUI node definitions for the Anima corrective sampler."""

from __future__ import annotations

import math

from .flow_sampler import (
    CFG_SCHEDULE_MODES,
    FLOW_SCHEDULES,
    FLOW_SOLVERS,
    run_comfy_anima_sampler,
)

ANIMA_FLOW_SETTINGS = "ANIMA_FLOW_SETTINGS"
DEFAULT_FLOW_SCHEDULE = "flow_cosmos"
PUBLIC_CFG_MODES = ["bump cfg", "const"]

ANIMA_FLOW_BASELINE = {
    "steps": 35,
    "cfg": 6.0,
    "flow_solver": "flow_pc3_damped",
    "flow_er_order": 2,
    "flow_pc3_gamma": 1.0,
    "flow_pc3_tolerance": 0.005,
    "flow_schedule": DEFAULT_FLOW_SCHEDULE,
    "flow_shift": 5.0,
    "cosmos_sigma_max": 80.0,
    "cosmos_sigma_min": 0.002,
    "denoise_legacy_progress": False,
    "cfg_legacy_progress": False,
    "cfg_schedule_mode": "beta_bump",
    "early_cfg_boost": 0.5,
    "early_cfg_until": 0.30,
    "late_cfg_scale": 1.0,
    "late_cfg_start": 0.76,
    "cfg_early_scale": 1.0,
    "cfg_early_ramp_end": 0.0,
    "cfg_peak_boost": 0.60,
    "cfg_bump_start": 0.0,
    "cfg_bump_end": 0.27,
    "cfg_beta_alpha": 2.0,
    "cfg_beta_beta": 3.0,
    "cfg_interval_start": 0.12,
    "cfg_interval_rise_end": 0.24,
    "cfg_interval_fall_start": 0.36,
    "cfg_interval_end": 0.58,
    "rf_endpoint_noise_refresh_enabled": False,
    "rf_endpoint_noise_refresh_strength": 0.15,
    "rf_endpoint_noise_refresh_until": 0.20,
}


class AnimaFlowSettings:
    """Optional advanced settings for the standalone Anima Flow sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_er_order": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_er_order"],
                        "min": 1,
                        "max": 3,
                    },
                ),
                "flow_pc3_gamma": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_gamma"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "flow_pc3_tolerance": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_tolerance"],
                        "min": 0.0001,
                        "max": 0.05,
                        "step": 0.0005,
                    },
                ),
                "cfg_early_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "cfg_early_ramp_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_ramp_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_peak_boost": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_peak_boost"],
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                    },
                ),
                "cfg_bump_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_bump_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_beta_alpha": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_alpha"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "cfg_beta_beta": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_beta"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "late_cfg_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "late_cfg_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["cfg_legacy_progress"]},
                ),
                "denoise_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["denoise_legacy_progress"]},
                ),
                "cosmos_sigma_max": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_max"],
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.5,
                    },
                ),
                "cosmos_sigma_min": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_min"],
                        "min": 0.0001,
                        "max": 1.0,
                        "step": 0.0001,
                    },
                ),
                "rf_endpoint_noise_refresh_enabled": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_enabled"]},
                ),
                "rf_endpoint_noise_refresh_strength": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_strength"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "rf_endpoint_noise_refresh_until": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_until"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            },
        }

    RETURN_TYPES = (ANIMA_FLOW_SETTINGS, "STRING")
    RETURN_NAMES = ("settings", "summary")
    FUNCTION = "build"
    CATEGORY = "Anima/Error Corrective Sampling"

    def build(
        self,
        flow_er_order,
        flow_pc3_gamma,
        flow_pc3_tolerance,
        cfg_early_scale,
        cfg_early_ramp_end,
        cfg_peak_boost,
        cfg_bump_start,
        cfg_bump_end,
        cfg_beta_alpha,
        cfg_beta_beta,
        late_cfg_scale,
        late_cfg_start,
        cfg_legacy_progress,
        denoise_legacy_progress,
        cosmos_sigma_max,
        cosmos_sigma_min,
        rf_endpoint_noise_refresh_enabled,
        rf_endpoint_noise_refresh_strength,
        rf_endpoint_noise_refresh_until,
    ):
        settings = _normalize_flow_params(
            {
                **ANIMA_FLOW_BASELINE,
                "flow_er_order": flow_er_order,
                "flow_pc3_gamma": flow_pc3_gamma,
                "flow_pc3_tolerance": flow_pc3_tolerance,
                "cfg_early_scale": cfg_early_scale,
                "cfg_early_ramp_end": cfg_early_ramp_end,
                "cfg_peak_boost": cfg_peak_boost,
                "cfg_bump_start": cfg_bump_start,
                "cfg_bump_end": cfg_bump_end,
                "cfg_beta_alpha": cfg_beta_alpha,
                "cfg_beta_beta": cfg_beta_beta,
                "late_cfg_scale": late_cfg_scale,
                "late_cfg_start": late_cfg_start,
                "cfg_legacy_progress": cfg_legacy_progress,
                "denoise_legacy_progress": denoise_legacy_progress,
                "cosmos_sigma_max": cosmos_sigma_max,
                "cosmos_sigma_min": cosmos_sigma_min,
                "rf_endpoint_noise_refresh_enabled": rf_endpoint_noise_refresh_enabled,
                "rf_endpoint_noise_refresh_strength": rf_endpoint_noise_refresh_strength,
                "rf_endpoint_noise_refresh_until": rf_endpoint_noise_refresh_until,
            }
        )
        return settings, _format_settings_summary(settings)


class AnimaFlowCorrectiveSampler:
    """Standalone RC2 Flow sampler for Anima."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["steps"],
                        "min": 1,
                        "max": 1000,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg"],
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                    },
                ),
                "cfg_mode": (PUBLIC_CFG_MODES, {"default": "bump cfg"}),
                "flow_solver": (FLOW_SOLVERS, {"default": ANIMA_FLOW_BASELINE["flow_solver"]}),
                "flow_schedule": (FLOW_SCHEDULES, {"default": ANIMA_FLOW_BASELINE["flow_schedule"]}),
                "flow_shift": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_shift"],
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "flow_settings": (ANIMA_FLOW_SETTINGS,),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "log")
    FUNCTION = "sample"
    CATEGORY = "Anima/Error Corrective Sampling"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        cfg_mode,
        flow_solver,
        flow_schedule,
        flow_shift,
        denoise,
        add_noise,
        flow_settings=None,
        vae=None,
    ):
        base_params = _normalize_settings_object(flow_settings)
        params = _normalize_flow_params(
            {
                **base_params,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "flow_solver": flow_solver,
                "flow_schedule": flow_schedule,
                "flow_shift": flow_shift,
            }
        )
        params = _apply_public_cfg_mode(params, cfg_mode)
        latent_out, log = _run_sampler_with_params(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            params=params,
            denoise=denoise,
            add_noise=add_noise,
            disable_pbar=False,
        )
        image = _decode_latent_image(vae, latent_out)
        if vae is None:
            log = f"{log}\nimage_output: unavailable (connect VAE)"
        else:
            log = f"{log}\nimage_output: decoded with connected VAE"
        return latent_out, image, log


def _run_sampler_with_params(
    *,
    model,
    positive,
    negative,
    latent_image,
    params: dict,
    denoise,
    add_noise,
    disable_pbar,
):
    return run_comfy_anima_sampler(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent_image,
        seed=int(params["seed"]),
        steps=int(params["steps"]),
        cfg=float(params["cfg"]),
        denoise=denoise,
        flow_solver=str(params["flow_solver"]),
        flow_er_order=int(params["flow_er_order"]),
        flow_pc3_gamma=float(params["flow_pc3_gamma"]),
        flow_pc3_tolerance=float(params["flow_pc3_tolerance"]),
        flow_schedule=str(params["flow_schedule"]),
        flow_shift=float(params["flow_shift"]),
        cosmos_sigma_max=float(params["cosmos_sigma_max"]),
        cosmos_sigma_min=float(params["cosmos_sigma_min"]),
        denoise_legacy_progress=bool(params["denoise_legacy_progress"]),
        cfg_schedule_domain=_cfg_domain_from_settings(params),
        cfg_schedule_mode=str(params["cfg_schedule_mode"]),
        early_cfg_boost=float(params["early_cfg_boost"]),
        early_cfg_until=float(params["early_cfg_until"]),
        late_cfg_scale=float(params["late_cfg_scale"]),
        late_cfg_start=float(params["late_cfg_start"]),
        cfg_early_scale=float(params["cfg_early_scale"]),
        cfg_early_ramp_end=float(params["cfg_early_ramp_end"]),
        cfg_peak_boost=float(params["cfg_peak_boost"]),
        cfg_bump_start=float(params["cfg_bump_start"]),
        cfg_bump_end=float(params["cfg_bump_end"]),
        cfg_beta_alpha=float(params["cfg_beta_alpha"]),
        cfg_beta_beta=float(params["cfg_beta_beta"]),
        cfg_interval_start=float(params["cfg_interval_start"]),
        cfg_interval_rise_end=float(params["cfg_interval_rise_end"]),
        cfg_interval_fall_start=float(params["cfg_interval_fall_start"]),
        cfg_interval_end=float(params["cfg_interval_end"]),
        rf_endpoint_noise_refresh_enabled=bool(params["rf_endpoint_noise_refresh_enabled"]),
        rf_endpoint_noise_refresh_strength=float(params["rf_endpoint_noise_refresh_strength"]),
        rf_endpoint_noise_refresh_until=float(params["rf_endpoint_noise_refresh_until"]),
        add_noise=add_noise,
        disable_pbar=disable_pbar,
    )


NODE_CLASS_MAPPINGS = {
    "AnimaFlowSettings": AnimaFlowSettings,
    "AnimaFlowCorrectiveSampler": AnimaFlowCorrectiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaFlowSettings": "Anima Flow Settings",
    "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
}


def _normalize_settings_object(flow_settings) -> dict:
    if flow_settings is None:
        return _normalize_flow_params(ANIMA_FLOW_BASELINE)
    if not isinstance(flow_settings, dict):
        raise ValueError("flow_settings must be an Anima Flow Settings object")

    out = dict(ANIMA_FLOW_BASELINE)
    for key, value in flow_settings.items():
        if key in ANIMA_FLOW_BASELINE:
            out[key] = value
    return _normalize_flow_params(out)


def _decode_latent_image(vae, latent: dict):
    if vae is None:
        return None
    if "samples" not in latent:
        raise ValueError("latent output must contain a 'samples' tensor")

    samples = latent["samples"]
    ndim = int(getattr(samples, "ndim", len(samples.shape)))
    if ndim == 5:
        if int(samples.shape[2]) != 1:
            raise ValueError("VAE image output requires a single-frame latent")
        samples = samples.squeeze(2)
    elif ndim != 4:
        raise ValueError("VAE image output requires a 4D image latent")
    return vae.decode(samples)


def _apply_public_cfg_mode(params: dict, cfg_mode: str) -> dict:
    out = dict(params)
    mode = str(cfg_mode).strip().lower()
    if mode == "bump cfg":
        out["cfg_schedule_mode"] = "beta_bump"
    elif mode == "const":
        out["cfg_schedule_mode"] = "constant"
        out["cfg_early_scale"] = 1.0
        out["cfg_early_ramp_end"] = 0.0
        out["cfg_peak_boost"] = 0.0
        out["late_cfg_scale"] = 1.0
    else:
        allowed = ", ".join(PUBLIC_CFG_MODES)
        raise ValueError(f"cfg_mode must be one of: {allowed}")
    return _normalize_flow_params(out)


def _normalize_flow_params(params: dict) -> dict:
    out = dict(params)
    out["steps"] = int(out["steps"])
    out["cfg"] = float(out["cfg"])
    out["flow_solver"] = str(out["flow_solver"])
    out["flow_er_order"] = int(out["flow_er_order"])
    out["flow_pc3_gamma"] = float(out["flow_pc3_gamma"])
    out["flow_pc3_tolerance"] = float(out["flow_pc3_tolerance"])
    out["flow_schedule"] = str(out["flow_schedule"])
    out["flow_shift"] = float(out["flow_shift"])
    out["cosmos_sigma_max"] = float(out["cosmos_sigma_max"])
    out["cosmos_sigma_min"] = float(out["cosmos_sigma_min"])
    out["denoise_legacy_progress"] = _as_bool(out["denoise_legacy_progress"])
    out["cfg_legacy_progress"] = _as_bool(out["cfg_legacy_progress"])
    out["cfg_schedule_mode"] = str(out["cfg_schedule_mode"])
    out["cfg_early_scale"] = float(out["cfg_early_scale"])
    out["cfg_early_ramp_end"] = float(out["cfg_early_ramp_end"])
    out["cfg_peak_boost"] = float(out["cfg_peak_boost"])
    out["cfg_bump_start"] = float(out["cfg_bump_start"])
    out["cfg_bump_end"] = float(out["cfg_bump_end"])
    out["cfg_beta_alpha"] = float(out["cfg_beta_alpha"])
    out["cfg_beta_beta"] = float(out["cfg_beta_beta"])
    out["cfg_interval_start"] = float(out["cfg_interval_start"])
    out["cfg_interval_rise_end"] = float(out["cfg_interval_rise_end"])
    out["cfg_interval_fall_start"] = float(out["cfg_interval_fall_start"])
    out["cfg_interval_end"] = float(out["cfg_interval_end"])
    out["early_cfg_boost"] = float(out["early_cfg_boost"])
    out["early_cfg_until"] = float(out["early_cfg_until"])
    out["late_cfg_scale"] = float(out["late_cfg_scale"])
    out["late_cfg_start"] = float(out["late_cfg_start"])
    out["rf_endpoint_noise_refresh_enabled"] = _as_bool(
        out["rf_endpoint_noise_refresh_enabled"]
    )
    out["rf_endpoint_noise_refresh_strength"] = float(
        out["rf_endpoint_noise_refresh_strength"]
    )
    out["rf_endpoint_noise_refresh_until"] = float(out["rf_endpoint_noise_refresh_until"])

    if out["flow_solver"] not in FLOW_SOLVERS:
        raise ValueError(f"unsupported flow_solver: {out['flow_solver']}")
    if not (1 <= out["flow_er_order"] <= 3):
        raise ValueError("flow_er_order must be in the range [1, 3]")
    if not (0.0 <= out["flow_pc3_gamma"] <= 1.0):
        raise ValueError("flow_pc3_gamma must be in the range [0, 1]")
    if not (0.0 < out["flow_pc3_tolerance"] <= 1.0):
        raise ValueError("flow_pc3_tolerance must be in the range (0, 1]")
    if out["flow_schedule"] not in FLOW_SCHEDULES:
        raise ValueError(f"unsupported flow_schedule: {out['flow_schedule']}")
    if not math.isfinite(out["flow_shift"]) or out["flow_shift"] < 1.0:
        raise ValueError("flow_shift must be finite and >= 1")
    if not (
        math.isfinite(out["cosmos_sigma_min"])
        and math.isfinite(out["cosmos_sigma_max"])
        and 0.0 < out["cosmos_sigma_min"] < out["cosmos_sigma_max"]
    ):
        raise ValueError("expected finite 0 < cosmos_sigma_min < cosmos_sigma_max")
    if out["cfg_schedule_mode"] not in CFG_SCHEDULE_MODES:
        allowed = ", ".join(CFG_SCHEDULE_MODES)
        raise ValueError(f"cfg_schedule_mode must be one of: {allowed}")
    if not (0.0 <= out["cfg_early_scale"] <= 2.0):
        raise ValueError("cfg_early_scale must be in the range [0, 2]")
    if not (0.0 <= out["cfg_early_ramp_end"] <= 1.0):
        raise ValueError("cfg_early_ramp_end must be in the range [0, 1]")
    if out["cfg_peak_boost"] < 0.0:
        raise ValueError("cfg_peak_boost must be non-negative")
    if not (0.0 <= out["cfg_bump_start"] < out["cfg_bump_end"] <= 1.0):
        raise ValueError("expected 0 <= cfg_bump_start < cfg_bump_end <= 1")
    if out["cfg_beta_alpha"] <= 1.0 or out["cfg_beta_beta"] <= 1.0:
        raise ValueError("cfg_beta_alpha and cfg_beta_beta must be > 1")
    if not (
        0.0
        <= out["cfg_interval_start"]
        <= out["cfg_interval_rise_end"]
        <= out["cfg_interval_fall_start"]
        <= out["cfg_interval_end"]
        <= 1.0
    ):
        raise ValueError(
            "expected cfg_interval_start <= cfg_interval_rise_end <= "
            "cfg_interval_fall_start <= cfg_interval_end within [0, 1]"
        )
    if not (0.0 <= out["rf_endpoint_noise_refresh_strength"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_strength must be in the range [0, 1]")
    if not (0.0 <= out["rf_endpoint_noise_refresh_until"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_until must be in the range [0, 1]")
    return out


def _cfg_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["cfg_legacy_progress"]) else "lambda"


def _denoise_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["denoise_legacy_progress"]) else "lambda"


def _estimated_model_calls(settings: dict) -> int:
    steps = max(1, int(settings["steps"]))
    if settings["flow_solver"] in {
        "flow_heun",
        "flow_pc3_damped",
        "flow_pc3_fsal_gated",
    }:
        return max(1, steps * 2 - 1)
    if settings["flow_solver"] == "flow_3m_sparse_pc3_fsal":
        sparse_budget = min(10, max(5, round(0.23 * steps)))
        return max(1, steps + sparse_budget)
    return steps


def _format_settings_summary(settings: dict) -> str:
    return "\n".join(
        [
            "AnimaFlowSettings",
            (
                "note: optional advanced settings; sampler steps/cfg/cfg_mode/"
                "solver/scheduler/shift override this object"
            ),
            f"steps_default: {settings['steps']}",
            f"cfg_default: {settings['cfg']:.2f}",
            f"estimated_model_calls_at_defaults: {_estimated_model_calls(settings)}",
            f"sampler_default_flow_solver: {settings['flow_solver']}",
            f"sampler_default_flow_schedule: {settings['flow_schedule']}",
            f"sampler_default_flow_shift: {settings['flow_shift']:.4f}",
            f"flow_er_order: {settings['flow_er_order']}",
            f"flow_pc3_gamma: {settings['flow_pc3_gamma']:.4f}",
            f"flow_pc3_tolerance: {settings['flow_pc3_tolerance']:.6f}",
            f"cosmos_sigma_max: {settings['cosmos_sigma_max']:.4f}",
            f"cosmos_sigma_min: {settings['cosmos_sigma_min']:.6f}",
            f"cfg_schedule_mode: {settings['cfg_schedule_mode']}",
            f"cfg_schedule_domain: {_cfg_domain_from_settings(settings)}",
            f"cfg_early_scale: {settings['cfg_early_scale']:.4f}",
            f"cfg_early_ramp_end: {settings['cfg_early_ramp_end']:.4f}",
            f"cfg_peak_boost: {settings['cfg_peak_boost']:.4f}",
            f"cfg_bump_start: {settings['cfg_bump_start']:.4f}",
            f"cfg_bump_end: {settings['cfg_bump_end']:.4f}",
            f"cfg_beta_alpha: {settings['cfg_beta_alpha']:.4f}",
            f"cfg_beta_beta: {settings['cfg_beta_beta']:.4f}",
            f"late_cfg_scale: {settings['late_cfg_scale']:.4f}",
            f"late_cfg_start: {settings['late_cfg_start']:.4f}",
            f"cfg_legacy_progress: {settings['cfg_legacy_progress']}",
            f"denoise_legacy_progress: {settings['denoise_legacy_progress']}",
            f"denoise_schedule_domain: {_denoise_domain_from_settings(settings)}",
            (
                "rf_endpoint_noise_refresh_enabled: "
                f"{settings['rf_endpoint_noise_refresh_enabled']}"
            ),
            (
                "rf_endpoint_noise_refresh_strength: "
                f"{settings['rf_endpoint_noise_refresh_strength']:.4f}"
            ),
            (
                "rf_endpoint_noise_refresh_until: "
                f"{settings['rf_endpoint_noise_refresh_until']:.4f}"
            ),
        ]
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got: {value!r}")
