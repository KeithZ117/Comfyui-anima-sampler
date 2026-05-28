"""Anima-specific sampling loop for ComfyUI.

This module keeps the risky ComfyUI integration narrow:

- ComfyUI still prepares the model, conditioning, masks, and latent IO.
- This project supplies the actual per-step loop through ``KSAMPLER``.
- Dynamic CFG is applied by updating ComfyUI's ``CFGGuider`` before each model
  call, instead of patching Anima's transformer internals.

The loop is intentionally a minimal Flow Matching Euler solver. Cosmos/Predict
technical reports describe training with ``x_t = (1 - t) x + t eps`` and a
velocity target. ComfyUI's FLOW wrapper returns the corresponding denoised
``x_0`` estimate, so the sampler can recover velocity as ``(x_t - x_0) / t``
and integrate one step toward the next Flow timestep.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any

from .scheduler import (
    build_flow_cosmos_lambda_biased_sigmas,
    build_flow_cosmos_rho_rf_tail_sigmas,
    build_flow_cosmos_shift_rf_tail_sigmas,
    build_flow_cosmos_sigmas,
    build_simple_sigmas,
)


FLOW_SOLVERS = [
    "flow_euler",
    "flow_heun",
    "flow_pc3_damped",
    "flow_pc3_fsal_gated",
    "flow_3m_damped",
    "flow_3m_sparse_pc3_fsal",
    "flow_er",
]
FLOW_SCHEDULES = [
    "flow_cosmos",
    "flow_cosmos_lambda_biased_strong",
    "flow_cosmos_rho7_rf_tail_auto",
    "simple",
]

CFG_SCHEDULE_DOMAINS = [
    "lambda",
    "rf_t",
    "progress",
]
CFG_SCHEDULE_MODES = [
    "beta_bump",
    "limited_interval",
    "legacy_boost",
    "constant",
]

FLOW_SHIFT_EPSILON = 1e-6


def _flow_shift_log_line(flow_schedule: str, flow_shift: float) -> str:
    if str(flow_schedule) == "flow_cosmos":
        return f"flow_shift: {float(flow_shift):.4f}"
    return f"flow_shift: {float(flow_shift):.4f} (ignored by {flow_schedule})"


def _flow_shift_is_active(flow_schedule: str, flow_shift: float) -> bool:
    try:
        value = float(flow_shift)
    except (TypeError, ValueError):
        return False
    return (
        str(flow_schedule) == "flow_cosmos"
        and math.isfinite(value)
        and value > 1.0 + FLOW_SHIFT_EPSILON
    )


@dataclass(frozen=True)
class AnimaSamplerLog:
    """Small text log returned from the ComfyUI node."""

    requested_steps: int
    actual_steps: int
    latent_in_shape: str
    latent_sample_shape: str
    added_temporal_dim: bool
    channel_adapter: str
    x_embedder_features: str
    sampler_core: str
    flow_schedule: str
    flow_shift: float
    cfg_schedule_mode: str
    cfg_schedule_domain: str
    denoise_legacy_progress: bool
    model_sampling_shift: str
    denoise: float
    flow_er_order: int
    flow_pc3_gamma: float
    flow_pc3_tolerance: float
    cosmos_sigma_max: float
    cosmos_sigma_min: float
    cfg_start: float
    cfg_mid: float
    cfg_end: float
    rf_endpoint_noise_refresh_enabled: bool
    rf_endpoint_noise_refresh_strength: float
    rf_endpoint_noise_refresh_until: float
    actual_model_calls: int | None = None
    cache_candidates: int = 0
    cache_accepts: int = 0
    cache_rejects: int = 0
    forced_refresh_count: int = 0
    pc3_used_total: int = 0
    pc3_used_high: int = 0
    pc3_used_body: int = 0
    pc3_used_tail: int = 0
    mean_cache_score: float | None = None
    p95_cache_score: float | None = None
    mean_gamma_pc3: float | None = None
    mean_gamma3: float | None = None

    def as_text(self) -> str:
        lines = [
            "AnimaFlowCorrectiveSampler",
            f"requested_steps: {self.requested_steps}",
            f"actual_steps: {self.actual_steps}",
            "steps_semantics: RF integration intervals, not diffusion denoise steps",
            f"estimated_model_calls: {self.estimated_model_calls()}",
        ]
        if self.actual_model_calls is not None:
            lines.append(f"actual_model_calls: {self.actual_model_calls}")
        lines.extend(
            [
                f"latent_in_shape: {self.latent_in_shape}",
                f"latent_sample_shape: {self.latent_sample_shape}",
                f"added_temporal_dim: {self.added_temporal_dim}",
                f"channel_adapter: {self.channel_adapter}",
                f"x_embedder_features: {self.x_embedder_features}",
                f"sampler_core: {self.sampler_core}",
                f"flow_schedule: {self.flow_schedule}",
                _flow_shift_log_line(self.flow_schedule, self.flow_shift),
                f"cfg_schedule_mode: {self.cfg_schedule_mode}",
                f"cfg_schedule_domain: {self.cfg_schedule_domain}",
                f"denoise_legacy_progress: {self.denoise_legacy_progress}",
                f"model_sampling_shift: {self.model_sampling_shift}",
                f"denoise: {self.denoise:.4f}",
                f"flow_er_order: {self.flow_er_order}",
                f"flow_pc3_gamma: {self.flow_pc3_gamma:.4f}",
                f"flow_pc3_tolerance: {self.flow_pc3_tolerance:.6f}",
                f"cosmos_sigma_max: {self.cosmos_sigma_max:.4f}",
                f"cosmos_sigma_min: {self.cosmos_sigma_min:.6f}",
                f"cfg_start: {self.cfg_start:.4f}",
                f"cfg_mid: {self.cfg_mid:.4f}",
                f"cfg_end: {self.cfg_end:.4f}",
            ]
        )
        if self.cache_candidates or self.cache_accepts:
            accept_rate = self.cache_accepts / max(self.cache_candidates, 1)
            lines.extend(
                [
                    f"cache_candidates: {self.cache_candidates}",
                    f"cache_accepts: {self.cache_accepts}",
                    f"cache_rejects: {self.cache_rejects}",
                    f"cache_accept_rate: {accept_rate:.4f}",
                    f"forced_refresh_count: {self.forced_refresh_count}",
                ]
            )
            if self.mean_cache_score is not None:
                lines.append(f"mean_cache_score: {self.mean_cache_score:.4f}")
            if self.p95_cache_score is not None:
                lines.append(f"p95_cache_score: {self.p95_cache_score:.4f}")
        if self.pc3_used_total:
            lines.extend(
                [
                    f"pc3_used_total: {self.pc3_used_total}",
                    f"pc3_used_high: {self.pc3_used_high}",
                    f"pc3_used_body: {self.pc3_used_body}",
                    f"pc3_used_tail: {self.pc3_used_tail}",
                ]
            )
        if self.mean_gamma_pc3 is not None:
            lines.append(f"mean_gamma_pc3: {self.mean_gamma_pc3:.4f}")
        if self.mean_gamma3 is not None:
            lines.append(f"mean_gamma3: {self.mean_gamma3:.4f}")
        lines.extend(
            [
                f"rf_endpoint_noise_refresh_enabled: {self.rf_endpoint_noise_refresh_enabled}",
                f"rf_endpoint_noise_refresh_strength: {self.rf_endpoint_noise_refresh_strength:.4f}",
                f"rf_endpoint_noise_refresh_until: {self.rf_endpoint_noise_refresh_until:.4f}",
            ]
        )
        return "\n".join(lines)

    def estimated_model_calls(self) -> int:
        if self.sampler_core in {
            "flow_heun",
            "flow_pc3_damped",
            "flow_pc3_fsal_gated",
        }:
            return max(1, self.actual_steps * 2 - 1)
        if self.sampler_core == "flow_3m_sparse_pc3_fsal":
            sparse_budget = min(10, max(5, round(0.23 * self.actual_steps)))
            return max(1, self.actual_steps + sparse_budget)
        return max(1, self.actual_steps)


@dataclass
class FlowERState:
    """History needed by the RF x0 LMS multistep corrector."""

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_lambda: Any | None = None


@dataclass
class FlowPC3State:
    """History needed by the RF x0 exponential PC3 corrector.

    Store only x0 predictions evaluated on accepted actual sampler states.
    Endpoint predictions from a predictor state are intentionally not history.
    """

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None


@dataclass
class FlowPC3StepResult:
    """Accepted PC3 step plus diagnostics used by FSAL cache gates."""

    x: Any
    state: FlowPC3State
    x_heun: Any
    x_am3: Any
    gamma: Any
    error: Any


@dataclass
class Flow3MStepResult:
    """Accepted one-eval 3M step plus diagnostics for sparse PC3 gating."""

    x: Any
    state: FlowERState
    x_2m: Any
    x_3m: Any
    gamma3: Any
    e32: Any
    order: int
    coeff_l1: float


@dataclass
class FlowFSALCacheState:
    """Endpoint denoised cache for PC3 FSAL-style current-call reuse."""

    pending_denoised: Any | None = None
    pending_t: Any | None = None
    pending_score: float | None = None
    pending_e_x: float | None = None
    consecutive_reuse: int = 0
    steps_since_fresh: int = 0
    force_next_order_le_2: bool = False

    def clear_pending(self):
        self.pending_denoised = None
        self.pending_t = None
        self.pending_score = None
        self.pending_e_x = None


def cfg_at_progress(
    progress: float,
    *,
    base_cfg: float,
    cfg_schedule_mode: str = "legacy_boost",
    early_cfg_boost: float = 0.0,
    early_cfg_until: float = 0.0,
    late_cfg_scale: float = 1.0,
    late_cfg_start: float = 1.0,
    cfg_early_scale: float = 1.0,
    cfg_early_ramp_end: float = 0.0,
    cfg_peak_boost: float = 0.0,
    cfg_bump_start: float = 0.08,
    cfg_bump_end: float = 0.68,
    cfg_beta_alpha: float = 4.0,
    cfg_beta_beta: float = 7.0,
    cfg_interval_start: float = 0.12,
    cfg_interval_rise_end: float = 0.24,
    cfg_interval_fall_start: float = 0.36,
    cfg_interval_end: float = 0.58,
) -> float:
    """Return the CFG value for a normalized sampling progress position.

    ``progress`` is 0 at the first denoising transition and 1 near the last.
    ``legacy_boost`` preserves the original early-high behavior. The newer
    modes keep very early guidance neutral or mild, then add a bounded bump
    around the early/mid-high structure-forming region.
    """

    progress = _clamp01(progress)
    base_cfg = max(0.0, float(base_cfg))
    mode = str(cfg_schedule_mode)

    if mode == "constant":
        return base_cfg

    if mode == "legacy_boost":
        early_cfg_until = _clamp01(early_cfg_until)
        late_cfg_start = _clamp01(late_cfg_start)
        if early_cfg_until > 0.0 and progress < early_cfg_until:
            alpha = 1.0 - progress / early_cfg_until
            return max(0.0, base_cfg + float(early_cfg_boost) * alpha)

        if late_cfg_start < 1.0 and progress > late_cfg_start:
            alpha = (progress - late_cfg_start) / (1.0 - late_cfg_start)
            late_cfg = base_cfg * float(late_cfg_scale)
            return max(0.0, base_cfg + (late_cfg - base_cfg) * alpha)

        return base_cfg

    early_scale = max(0.0, float(cfg_early_scale))
    early_ramp = _clamp01(cfg_early_ramp_end)
    late_cfg_start = _clamp01(late_cfg_start)
    late_scale = max(0.0, float(late_cfg_scale))

    early_mul = early_scale + (1.0 - early_scale) * _smoothstep(0.0, early_ramp, progress)
    late_mul = 1.0 + (late_scale - 1.0) * _smoothstep(late_cfg_start, 1.0, progress)
    cfg = base_cfg * early_mul * late_mul

    if mode == "limited_interval":
        bump = _interval_window(
            progress,
            start=cfg_interval_start,
            rise_end=cfg_interval_rise_end,
            fall_start=cfg_interval_fall_start,
            end=cfg_interval_end,
        )
        return max(0.0, cfg + float(cfg_peak_boost) * bump)

    if mode == "beta_bump":
        denom = max(float(cfg_bump_end) - float(cfg_bump_start), 1e-8)
        z = (progress - float(cfg_bump_start)) / denom
        bump = _beta_bump_unit(z, alpha=cfg_beta_alpha, beta=cfg_beta_beta)
        return max(0.0, cfg + float(cfg_peak_boost) * bump)

    allowed = ", ".join(CFG_SCHEDULE_MODES)
    raise ValueError(f"cfg_schedule_mode must be one of: {allowed}")


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 == edge0:
        return 1.0 if value >= edge1 else 0.0

    z = _clamp01((float(value) - float(edge0)) / (float(edge1) - float(edge0)))
    return z * z * (3.0 - 2.0 * z)


def _beta_bump_unit(z: float, *, alpha: float, beta: float, eps: float = 1e-12) -> float:
    z = _clamp01(z)
    if z <= 0.0 or z >= 1.0:
        return 0.0

    alpha = max(float(alpha), 1.0001)
    beta = max(float(beta), 1.0001)
    z_peak = (alpha - 1.0) / (alpha + beta - 2.0)
    z_peak = min(1.0 - eps, max(eps, z_peak))

    value = (z ** (alpha - 1.0)) * ((1.0 - z) ** (beta - 1.0))
    peak = (z_peak ** (alpha - 1.0)) * ((1.0 - z_peak) ** (beta - 1.0))
    return float(value / (peak + eps))


def _interval_window(
    progress: float,
    *,
    start: float,
    rise_end: float,
    fall_start: float,
    end: float,
) -> float:
    up = _smoothstep(start, rise_end, progress)
    down = 1.0 - _smoothstep(fall_start, end, progress)
    return _clamp01(up * down)


def cfg_schedule_position(
    torch,
    t,
    sigmas,
    step_index: int,
    *,
    domain: str,
    total_steps: int | None = None,
    eps: float = 1e-6,
) -> float:
    """Return normalized denoise progress in the selected RF schedule domain."""

    if domain == "progress":
        if total_steps is None:
            total_steps = int(sigmas.shape[0]) - 1
        return _clamp01(step_index / max(total_steps - 1, 1))

    if domain == "rf_t":
        t_start = _scalar_float(torch, sigmas[0])
        t_end = _finite_schedule_terminal(torch, sigmas)
        current = _scalar_float(torch, t)
        denom = max(abs(t_start - t_end), eps)
        return _clamp01((t_start - current) / denom)

    if domain == "lambda":
        lambda_start = _scalar_float(torch, _rf_lambda(torch, sigmas[0], eps=eps))
        lambda_end = _scalar_float(torch, _rf_lambda(torch, _finite_schedule_terminal(torch, sigmas), eps=eps))
        current_lambda = _scalar_float(torch, _rf_lambda(torch, t, eps=eps))
        denom = max(abs(lambda_end - lambda_start), eps)
        return _clamp01((current_lambda - lambda_start) / denom)

    allowed = ", ".join(CFG_SCHEDULE_DOMAINS)
    raise ValueError(f"cfg_schedule_domain must be one of: {allowed}")


def build_anima_sigmas(
    model: Any,
    steps: int,
    *,
    denoise: float,
    flow_schedule: str,
    flow_shift: float = 1.0,
    cosmos_sigma_max: float = 80.0,
    cosmos_sigma_min: float = 0.002,
    denoise_legacy_progress: bool = False,
):
    """Build the ComfyUI ``sigmas`` tensor for the selected Flow schedule.

    ComfyUI's sampler API names this tensor ``sigmas``. For the custom
    ``flow_cosmos*`` schedules, the values stored in it are normalized
    rectified-flow times ``t`` expected by the model wrapper.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not (0.0 < denoise <= 1.0):
        raise ValueError("denoise must be in the range (0, 1]")
    flow_schedule = str(flow_schedule)
    flow_shift = float(flow_shift)
    cosmos_sigma_max = float(cosmos_sigma_max)
    cosmos_sigma_min = float(cosmos_sigma_min)
    if not math.isfinite(flow_shift) or flow_shift < 1.0:
        raise ValueError("flow_shift must be finite and >= 1")
    if not (
        math.isfinite(cosmos_sigma_min)
        and math.isfinite(cosmos_sigma_max)
        and 0.0 < cosmos_sigma_min < cosmos_sigma_max
    ):
        raise ValueError("expected finite 0 < cosmos_sigma_min < cosmos_sigma_max")

    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception as exc:
        raise RuntimeError("model must expose get_model_object('model_sampling')") from exc

    if not hasattr(model_sampling, "sigmas"):
        raise RuntimeError("model_sampling must expose a sigmas attribute")

    source_sigmas = model_sampling.sigmas
    schedule_steps = steps
    use_rf_denoise = (
        denoise < 0.9999
        and not denoise_legacy_progress
        and flow_schedule.startswith("flow_cosmos")
    )
    if denoise < 0.9999 and not use_rf_denoise:
        schedule_steps = max(steps, int(steps / denoise))

    if use_rf_denoise:
        values = _build_rf_denoise_sigmas(
            steps,
            denoise=denoise,
            flow_schedule=flow_schedule,
            flow_shift=flow_shift,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule == "flow_cosmos":
        values = _build_flow_cosmos_sigmas(
            schedule_steps,
            flow_shift=flow_shift,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule == "flow_cosmos_lambda_biased_strong":
        values = build_flow_cosmos_lambda_biased_sigmas(
            schedule_steps,
            strength="strong",
            sigma_max=cosmos_sigma_max,
            sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule.startswith("flow_cosmos_rho7_rf_tail_"):
        params = _rho_rf_tail_params(flow_schedule)
        values = build_flow_cosmos_rho_rf_tail_sigmas(
            schedule_steps,
            order=7.0,
            sigma_max=cosmos_sigma_max,
            sigma_min=cosmos_sigma_min,
            **params,
        )
    elif flow_schedule == "simple":
        values = build_simple_sigmas(source_sigmas, schedule_steps)
    else:
        raise ValueError(f"unsupported flow_schedule: {flow_schedule}")

    if denoise < 0.9999 and not use_rf_denoise:
        values = values[-(steps + 1) :]

    torch = importlib.import_module("torch")
    dtype = getattr(source_sigmas, "dtype", torch.float32)
    device = getattr(source_sigmas, "device", None)
    tensor_kwargs = {"dtype": dtype}
    if device is not None:
        tensor_kwargs["device"] = device
    return torch.tensor(values, **tensor_kwargs)


def _build_rf_denoise_sigmas(
    steps: int,
    *,
    denoise: float,
    flow_schedule: str,
    flow_shift: float,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
) -> list[float]:
    if flow_schedule == "flow_cosmos":
        start_sigma_max = (
            cosmos_sigma_max * flow_shift
            if _flow_shift_is_active(flow_schedule, flow_shift)
            else cosmos_sigma_max
        )
        sigma_start = _rf_denoise_start_external_sigma(
            denoise,
            sigma_max=start_sigma_max,
            sigma_min=cosmos_sigma_min,
        )
        return _build_flow_cosmos_sigmas(
            steps,
            flow_shift=flow_shift,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
            sigma_start=sigma_start,
        )

    sigma_start = _rf_denoise_start_external_sigma(
        denoise,
        sigma_max=cosmos_sigma_max,
        sigma_min=cosmos_sigma_min,
    )

    if flow_schedule == "flow_cosmos_lambda_biased_strong":
        return build_flow_cosmos_lambda_biased_sigmas(
            steps,
            strength="strong",
            sigma_max=sigma_start,
            sigma_min=cosmos_sigma_min,
        )
    if flow_schedule.startswith("flow_cosmos_rho7_rf_tail_"):
        params = _rho_rf_tail_params(flow_schedule)
        return build_flow_cosmos_rho_rf_tail_sigmas(
            steps,
            order=7.0,
            sigma_max=sigma_start,
            sigma_min=cosmos_sigma_min,
            **params,
        )
    raise ValueError(f"unsupported RF denoise flow_schedule: {flow_schedule}")


def _build_flow_cosmos_sigmas(
    steps: int,
    *,
    flow_shift: float,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
    sigma_start: float | None = None,
) -> list[float]:
    if _flow_shift_is_active("flow_cosmos", flow_shift):
        return build_flow_cosmos_shift_rf_tail_sigmas(
            steps,
            beta=float(flow_shift),
            sigma_max=cosmos_sigma_max,
            sigma_min=cosmos_sigma_min,
            sigma_start=sigma_start,
        )
    if sigma_start is not None:
        return _exact_logspace_rflow_sigmas(
            steps,
            sigma_max=float(sigma_start),
            sigma_min=cosmos_sigma_min,
        )
    return build_flow_cosmos_sigmas(
        steps,
        sigma_max=cosmos_sigma_max,
        sigma_min=cosmos_sigma_min,
    )


def _rho_rf_tail_params(flow_schedule: str) -> dict[str, float | None]:
    if flow_schedule == "flow_cosmos_rho7_rf_tail_auto":
        return {"tail_lambda_start": None, "tail_delta_ell_max": 0.5}
    raise ValueError(f"unsupported rho7 RF-tail schedule: {flow_schedule}")


def _rf_denoise_start_external_sigma(
    denoise: float,
    *,
    sigma_max: float,
    sigma_min: float,
) -> float:
    log_sigma = math.log(sigma_min) + float(denoise) * (
        math.log(sigma_max) - math.log(sigma_min)
    )
    return math.exp(log_sigma)


def _exact_logspace_rflow_sigmas(
    steps: int,
    *,
    sigma_max: float,
    sigma_min: float,
) -> list[float]:
    if steps < 1:
        raise ValueError("steps must be at least 1")
    if steps == 1:
        return [_rflow_time_from_external_sigma(sigma_max), 0.0]

    log_start = math.log(sigma_max)
    log_end = math.log(sigma_min)
    values = []
    for step in range(steps):
        alpha = step / (steps - 1)
        sigma = math.exp(log_start + alpha * (log_end - log_start))
        values.append(_rflow_time_from_external_sigma(sigma))
    values.append(0.0)
    return values


def _rflow_time_from_external_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    return float(sigma) / (float(sigma) + 1.0)


def _describe_model_sampling_shift(model: Any, *, flow_schedule: str) -> str:
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        return "unknown"

    class_name = type(model_sampling).__name__
    shift = getattr(model_sampling, "shift", None)
    if shift is not None:
        try:
            shift_text = f"{float(shift):.4g}"
        except (TypeError, ValueError):
            shift_text = str(shift)
        if class_name in {"ModelSamplingDiscreteFlow", "ModelSamplingFlux"}:
            if flow_schedule.startswith("flow_cosmos"):
                return f"{class_name}: native shift={shift_text} present, bypassed by {flow_schedule}"
            return f"{class_name}: native shift={shift_text} baked into sigmas"
        return f"{class_name}: shift={shift_text}"

    if class_name == "ModelSamplingCosmosRFlow":
        return f"{class_name}: sigma-ratio flow table"
    return f"{class_name}: no shift attribute"


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
    flow_er_order: int,
    flow_pc3_gamma: float,
    flow_pc3_tolerance: float,
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
) -> tuple[dict[str, Any], str]:
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

    sigmas = build_anima_sigmas(
        model,
        steps,
        denoise=denoise,
        flow_schedule=flow_schedule,
        flow_shift=flow_shift,
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
            "flow_er_order": int(flow_er_order),
            "flow_pc3_gamma": float(flow_pc3_gamma),
            "flow_pc3_tolerance": float(flow_pc3_tolerance),
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
        },
    )

    callback = latent_preview.prepare_callback(model, int(sigmas.shape[0]) - 1)
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

    actual_steps = int(sigmas.shape[0]) - 1
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


def _init_sampler_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "model_calls": 0,
            "cache_candidates": 0,
            "cache_accepts": 0,
            "cache_rejects": 0,
            "forced_refresh_count": 0,
            "pc3_used_total": 0,
            "pc3_used_high": 0,
            "pc3_used_body": 0,
            "pc3_used_tail": 0,
            "cache_scores": [],
            "gamma_pc3_values": [],
            "gamma3_values": [],
        }
    )
    return stats


def _record_model_call(stats: dict[str, Any] | None):
    if stats is not None:
        stats["model_calls"] = int(stats.get("model_calls", 0)) + 1


def _stats_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _stats_percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0))
    return float(ordered[index])


def _reset_fsal_cache(state: FlowFSALCacheState):
    state.clear_pending()
    state.consecutive_reuse = 0
    state.steps_since_fresh = 0
    state.force_next_order_le_2 = True


def _model_current_denoised(
    model,
    x,
    t,
    s_in,
    *,
    denoise_mask,
    model_options,
    seed: int,
    stats: dict[str, Any] | None,
):
    denoised = model(
        x,
        t * s_in,
        denoise_mask=denoise_mask,
        model_options=model_options,
        seed=seed,
    )
    _record_model_call(stats)
    return _restore_sampler_channels(denoised, x)


def _try_use_fsal_cache(
    torch,
    cache_state: FlowFSALCacheState,
    t,
    *,
    step_index: int,
    total_steps: int,
    force_fresh: bool,
    stats: dict[str, Any] | None,
    eps: float = 1e-6,
):
    if force_fresh:
        if cache_state.pending_denoised is not None and stats is not None:
            stats["forced_refresh_count"] = int(stats.get("forced_refresh_count", 0)) + 1
        return None, False, None
    if cache_state.pending_denoised is None or cache_state.pending_t is None:
        return None, False, None
    if step_index >= total_steps - 2:
        return None, False, None
    if cache_state.consecutive_reuse >= 3 or cache_state.steps_since_fresh >= 4:
        return None, False, None

    t_current = _scalar_float(torch, t)
    t_cached = _scalar_float(torch, cache_state.pending_t)
    if abs(t_current - t_cached) > eps:
        return None, False, None

    cached = cache_state.pending_denoised
    score = cache_state.pending_score
    cache_state.clear_pending()
    cache_state.consecutive_reuse += 1
    cache_state.steps_since_fresh += 1
    if stats is not None:
        stats["cache_accepts"] = int(stats.get("cache_accepts", 0)) + 1
    return cached, True, score


def _mark_fresh_current(cache_state: FlowFSALCacheState):
    cache_state.clear_pending()
    cache_state.consecutive_reuse = 0
    cache_state.steps_since_fresh = 0


def _pc3_fsal_cache_score(
    torch,
    *,
    x_pred,
    x_next,
    x_heun,
    x_am3,
    t_next,
    tolerance: float,
    eps: float = 1e-6,
) -> tuple[float, float, float]:
    den_x = torch.maximum(_rms(torch, x_next), _rms(torch, x_heun)) + eps
    e_x = _rms(torch, x_next - x_pred) / den_x
    e_pc3 = _rms(torch, x_am3 - x_heun) / (_rms(torch, x_heun) + eps)

    sqrt_t = math.sqrt(max(0.0, min(1.0, _scalar_float(torch, t_next))))
    tau_x = 0.0012 + 0.0038 * sqrt_t
    tau_pc3 = max(0.0, float(tolerance)) * (0.75 + 2.0 * sqrt_t)
    if tau_pc3 <= 0.0:
        return math.inf, _scalar_float(torch, e_x), _scalar_float(torch, e_pc3)

    score = max(
        _scalar_float(torch, e_x) / tau_x,
        _scalar_float(torch, e_pc3) / tau_pc3,
    )
    return float(score), _scalar_float(torch, e_x), _scalar_float(torch, e_pc3)


def _store_fsal_candidate(
    torch,
    cache_state: FlowFSALCacheState,
    *,
    denoised_next,
    t_next,
    score: float,
    e_x: float,
    step_index: int,
    total_steps: int,
    next_is_boundary: bool,
    stats: dict[str, Any] | None,
):
    if stats is not None:
        stats["cache_candidates"] = int(stats.get("cache_candidates", 0)) + 1
        stats.setdefault("cache_scores", []).append(float(score))

    if score < 1.0 and step_index < total_steps - 3 and not next_is_boundary:
        cache_state.pending_denoised = denoised_next
        cache_state.pending_t = t_next
        cache_state.pending_score = float(score)
        cache_state.pending_e_x = float(e_x)
        return

    cache_state.clear_pending()
    if stats is not None:
        stats["cache_rejects"] = int(stats.get("cache_rejects", 0)) + 1


def sample_anima_flow_corrective(
    model: Any,
    x,
    sigmas,
    extra_args: dict[str, Any],
    callback=None,
    disable=None,
    *,
    flow_solver: str,
    flow_schedule: str,
    flow_shift: float = 1.0,
    flow_er_order: int,
    flow_pc3_gamma: float,
    flow_pc3_tolerance: float,
    base_cfg: float,
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
    sampler_stats: dict[str, Any] | None = None,
):
    """ComfyUI ``KSAMPLER`` function implementing Flow Matching solvers.

    The model wrapper returns a denoised ``x_0`` estimate. For the flow
    parameterization ``x_t = (1 - t) x_0 + t eps``, velocity is
    ``v = (x_t - x_0) / t``. Euler integration from ``t`` to ``t_next`` is:

    ``x_next = x_t + (t_next - t) * v``.
    """

    torch = importlib.import_module("torch")
    k_sampling = importlib.import_module("comfy.k_diffusion.sampling")

    total_steps = int(sigmas.shape[0]) - 1
    if total_steps <= 0:
        return x

    cfg_guider = getattr(model, "inner_model", None)
    original_cfg = getattr(cfg_guider, "cfg", None)
    can_set_cfg = cfg_guider is not None and (
        hasattr(cfg_guider, "set_cfg") or hasattr(cfg_guider, "cfg")
    )
    warned_cfg = False

    seed = int(extra_args.get("seed", 0) or 0)
    generator = _make_generator(torch, x.device, seed + 1337)
    model_options = extra_args.get("model_options", {})
    denoise_mask = extra_args.get("denoise_mask", None)
    s_in = x.new_ones([x.shape[0]])

    er_state = FlowERState()
    pc3_state = FlowPC3State()
    fsal_cache_state = FlowFSALCacheState()
    stats = _init_sampler_stats(sampler_stats)
    hybrid_tail_start_step = _hybrid_tail_start_step(
        torch,
        sigmas,
        flow_schedule,
        flow_shift=flow_shift,
    )
    sparse_pc3_budget = _sparse_pc3_budget(total_steps)
    sparse_pc3_used = {"high": 0, "body": 0, "tail": 0}
    sparse_last_pc3_step = -999

    try:
        for step_index in k_sampling.trange(total_steps, disable=disable):
            comfy_sigma = sigmas[step_index]
            comfy_sigma_next = sigmas[step_index + 1]
            # ComfyUI calls these values sigmas. Inside this RF sampler they
            # are normalized flow times t for all solver math below.
            t = comfy_sigma
            t_next = comfy_sigma_next
            if hybrid_tail_start_step is not None and step_index == hybrid_tail_start_step:
                er_state = FlowERState()
                pc3_state = FlowPC3State()
                _reset_fsal_cache(fsal_cache_state)
                sparse_last_pc3_step = -999
            cfg_position = cfg_schedule_position(
                torch,
                t,
                sigmas,
                step_index,
                domain=cfg_schedule_domain,
                total_steps=total_steps,
            )

            cfg_step = cfg_at_progress(
                cfg_position,
                base_cfg=base_cfg,
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
            )
            if can_set_cfg:
                _set_cfg(cfg_guider, cfg_step)
            elif not warned_cfg:
                print("[comfyui-anima-sampler] Dynamic CFG unavailable; using static CFG.")
                warned_cfg = True

            t_for_model = t
            fsal_enabled = flow_solver in {
                "flow_pc3_fsal_gated",
                "flow_3m_sparse_pc3_fsal",
            }
            force_fresh_current = (
                not fsal_enabled
                or step_index >= total_steps - 2
                or (hybrid_tail_start_step is not None and step_index == hybrid_tail_start_step)
            )
            denoised, used_cached_denoised, cache_score = _try_use_fsal_cache(
                torch,
                fsal_cache_state,
                t_for_model,
                step_index=step_index,
                total_steps=total_steps,
                force_fresh=force_fresh_current,
                stats=stats,
            )
            if denoised is None:
                denoised = _model_current_denoised(
                    model,
                    x,
                    t_for_model,
                    s_in,
                    denoise_mask=denoise_mask,
                    model_options=model_options,
                    seed=seed,
                    stats=stats,
                )
                _mark_fresh_current(fsal_cache_state)

            if callback is not None:
                callback(
                    {
                        "i": step_index,
                        "sigma": comfy_sigma,
                        "sigma_hat": t_for_model,
                        "rf_t": t,
                        "rf_t_hat": t_for_model,
                        "denoised": denoised,
                        "x": x,
                        "fsal_cached": used_cached_denoised,
                    }
                )

            x_step_start = x
            if flow_solver == "flow_er":
                x, er_state = flow_er_step(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=er_state,
                    max_order=flow_er_order,
                )
                x, refresh_applied = rf_endpoint_noise_refresh(
                    torch,
                    x,
                    x_step_start,
                    denoised,
                    t_for_model,
                    t_next,
                    generator,
                    enabled=rf_endpoint_noise_refresh_enabled,
                    refresh_strength=rf_endpoint_noise_refresh_strength,
                    refresh_until=rf_endpoint_noise_refresh_until,
                )
                if refresh_applied:
                    er_state = FlowERState()
                continue

            if flow_solver == "flow_3m_damped":
                result_3m = flow_3m_damped_step(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=er_state,
                    max_gamma=flow_pc3_gamma,
                    tolerance=flow_pc3_tolerance,
                    force_order_le_2=fsal_cache_state.force_next_order_le_2,
                )
                fsal_cache_state.force_next_order_le_2 = False
                er_state = result_3m.state
                if stats is not None:
                    stats.setdefault("gamma3_values", []).append(_scalar_float(torch, result_3m.gamma3))
                x, refresh_applied = rf_endpoint_noise_refresh(
                    torch,
                    result_3m.x,
                    x_step_start,
                    denoised,
                    t_for_model,
                    t_next,
                    generator,
                    enabled=rf_endpoint_noise_refresh_enabled,
                    refresh_strength=rf_endpoint_noise_refresh_strength,
                    refresh_until=rf_endpoint_noise_refresh_until,
                )
                if refresh_applied:
                    er_state = FlowERState()
                continue

            if flow_solver == "flow_euler" or float(t_next) <= 0.0:
                x_det = flow_euler_step(x, denoised, t_for_model, t_next)
                x, _refresh_applied = rf_endpoint_noise_refresh(
                    torch,
                    x_det,
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    generator,
                    enabled=rf_endpoint_noise_refresh_enabled,
                    refresh_strength=rf_endpoint_noise_refresh_strength,
                    refresh_until=rf_endpoint_noise_refresh_until,
                )
                continue

            if flow_solver not in {
                "flow_heun",
                "flow_pc3_damped",
                "flow_pc3_fsal_gated",
                "flow_3m_sparse_pc3_fsal",
            }:
                raise ValueError(f"unsupported flow_solver: {flow_solver}")

            sparse_3m_result = None
            sparse_phase = _sparse_pc3_phase(step_index, total_steps, hybrid_tail_start_step)
            sparse_risk = 0.0
            if flow_solver == "flow_3m_sparse_pc3_fsal":
                sparse_3m_result = flow_3m_damped_step(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=er_state,
                    max_gamma=flow_pc3_gamma,
                    tolerance=flow_pc3_tolerance,
                    force_order_le_2=fsal_cache_state.force_next_order_le_2 or bool(used_cached_denoised),
                )
                fsal_cache_state.force_next_order_le_2 = False
                x_pred = sparse_3m_result.x
                sparse_risk = _sparse_pc3_risk(
                    torch,
                    denoised=denoised,
                    t=t_for_model,
                    t_next=t_next,
                    state=er_state,
                    result=sparse_3m_result,
                    tolerance=flow_pc3_tolerance,
                )
            elif flow_solver in {"flow_pc3_damped", "flow_pc3_fsal_gated"}:
                x_pred = flow_pc3_predictor_step(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=pc3_state,
                )
            else:
                x_pred = flow_euler_step(x, denoised, t_for_model, t_next)
            cfg_next_position = cfg_schedule_position(
                torch,
                t_next,
                sigmas,
                step_index + 1,
                domain=cfg_schedule_domain,
                total_steps=total_steps,
            )
            cfg_next = cfg_at_progress(
                cfg_next_position,
                base_cfg=base_cfg,
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
            )
            if can_set_cfg:
                _set_cfg(cfg_guider, cfg_next)

            if flow_solver == "flow_3m_sparse_pc3_fsal":
                do_endpoint_pc3 = _should_do_sparse_pc3(
                    step_index,
                    total_steps,
                    phase=sparse_phase,
                    risk=sparse_risk,
                    used=sparse_pc3_used,
                    budget=sparse_pc3_budget,
                    last_pc3_step=sparse_last_pc3_step,
                    tail_start_step=hybrid_tail_start_step,
                )
                if not do_endpoint_pc3:
                    if used_cached_denoised:
                        fsal_cache_state.force_next_order_le_2 = True
                    else:
                        er_state = sparse_3m_result.state
                        pc3_state = FlowPC3State(
                            previous_denoised=denoised,
                            previous_lambda=_rf_lambda(torch, t_for_model),
                        )
                    if stats is not None:
                        stats.setdefault("gamma3_values", []).append(_scalar_float(torch, sparse_3m_result.gamma3))
                    fsal_cache_state.clear_pending()
                    x, refresh_applied = rf_endpoint_noise_refresh(
                        torch,
                        x_pred,
                        x_step_start,
                        denoised,
                        t_for_model,
                        t_next,
                        generator,
                        enabled=rf_endpoint_noise_refresh_enabled,
                        refresh_strength=rf_endpoint_noise_refresh_strength,
                        refresh_until=rf_endpoint_noise_refresh_until,
                    )
                    if refresh_applied:
                        er_state = FlowERState()
                        pc3_state = FlowPC3State()
                        _reset_fsal_cache(fsal_cache_state)
                        sparse_last_pc3_step = -999
                    continue

            denoised_next = model(
                x_pred,
                t_next * s_in,
                denoise_mask=denoise_mask,
                model_options=model_options,
                seed=seed,
            )
            _record_model_call(stats)
            denoised_next = _restore_sampler_channels(denoised_next, x_pred)
            if flow_solver == "flow_heun":
                x_det = flow_heun_step(x, denoised, denoised_next, t_for_model, t_next)
            elif flow_solver in {
                "flow_pc3_damped",
                "flow_pc3_fsal_gated",
                "flow_3m_sparse_pc3_fsal",
            }:
                pc3_state_before_step = pc3_state
                pc3_result = flow_pc3_damped_step_result(
                    x,
                    denoised,
                    denoised_next,
                    t_for_model,
                    t_next,
                    state=pc3_state,
                    max_gamma=flow_pc3_gamma,
                    tolerance=flow_pc3_tolerance,
                )
                x_det = pc3_result.x
                if used_cached_denoised:
                    pc3_state = pc3_state_before_step
                    fsal_cache_state.force_next_order_le_2 = True
                else:
                    pc3_state = pc3_result.state
                if stats is not None:
                    stats["pc3_used_total"] = int(stats.get("pc3_used_total", 0)) + 1
                    stats[f"pc3_used_{sparse_phase}"] = int(stats.get(f"pc3_used_{sparse_phase}", 0)) + 1
                    stats.setdefault("gamma_pc3_values", []).append(_scalar_float(torch, pc3_result.gamma))
                if flow_solver == "flow_3m_sparse_pc3_fsal":
                    if not used_cached_denoised:
                        er_state = sparse_3m_result.state
                    sparse_pc3_used[sparse_phase] += 1
                    sparse_last_pc3_step = step_index
                    if stats is not None:
                        stats.setdefault("gamma3_values", []).append(_scalar_float(torch, sparse_3m_result.gamma3))
            else:
                raise ValueError(f"unsupported flow_solver: {flow_solver}")
            x, refresh_applied = rf_endpoint_noise_refresh(
                torch,
                x_det,
                x,
                denoised,
                t_for_model,
                t_next,
                generator,
                enabled=rf_endpoint_noise_refresh_enabled,
                refresh_strength=rf_endpoint_noise_refresh_strength,
                refresh_until=rf_endpoint_noise_refresh_until,
            )
            if flow_solver in {"flow_pc3_damped", "flow_pc3_fsal_gated", "flow_3m_sparse_pc3_fsal"} and refresh_applied:
                pc3_state = FlowPC3State()
                if flow_solver == "flow_3m_sparse_pc3_fsal":
                    er_state = FlowERState()
                    sparse_last_pc3_step = -999
                _reset_fsal_cache(fsal_cache_state)
            elif flow_solver in {"flow_pc3_fsal_gated", "flow_3m_sparse_pc3_fsal"}:
                next_is_boundary = (
                    hybrid_tail_start_step is not None
                    and step_index + 1 == hybrid_tail_start_step
                )
                score, e_x, _e_pc3 = _pc3_fsal_cache_score(
                    torch,
                    x_pred=x_pred,
                    x_next=x,
                    x_heun=pc3_result.x_heun,
                    x_am3=pc3_result.x_am3,
                    t_next=t_next,
                    tolerance=flow_pc3_tolerance,
                )
                _store_fsal_candidate(
                    torch,
                    fsal_cache_state,
                    denoised_next=denoised_next,
                    t_next=t_next,
                    score=score,
                    e_x=e_x,
                    step_index=step_index,
                    total_steps=total_steps,
                    next_is_boundary=next_is_boundary,
                    stats=stats,
                )
    finally:
        if can_set_cfg and original_cfg is not None:
            _set_cfg(cfg_guider, original_cfg)

    return x


def flow_euler_step(x, denoised, t, t_next):
    """Advance one Flow Matching Euler step using ComfyUI's denoised output."""

    if float(t_next) <= 0.0:
        return denoised
    if float(t) <= 0.0:
        return denoised

    velocity = flow_velocity(x, denoised, t)
    return x + (t_next - t) * velocity


def rf_endpoint_noise_refresh(
    torch,
    deterministic_next,
    x,
    denoised,
    t,
    t_next,
    generator,
    *,
    enabled: bool = True,
    refresh_strength: float = 0.15,
    s_noise: float = 1.0,
    refresh_until: float | None = 0.20,
    refresh_from: float | None = 0.999,
    eps: float = 1e-6,
):
    """Refresh RF endpoint noise directly on top of a deterministic solver step."""

    if not enabled:
        return deterministic_next, False

    t_f = _scalar_float(torch, t)
    t_next_f = _scalar_float(torch, t_next)
    if t_next_f <= eps:
        return deterministic_next, False

    t_current = _broadcast_time(torch, t, x)
    t_next_broadcast = _broadcast_time(torch, t_next, x)
    t_safe = torch.clamp(t_current, min=eps)

    if refresh_strength <= 0.0 or s_noise <= 0.0:
        return deterministic_next, False
    if refresh_until is not None and t_next_f < float(refresh_until):
        return deterministic_next, False
    if refresh_from is not None and t_f > float(refresh_from):
        return deterministic_next, False

    refresh_eff = max(0.0, min(1.0, float(refresh_strength)))
    endpoint_noise = (x - (1.0 - t_current) * denoised) / t_safe
    keep = math.sqrt(max(0.0, 1.0 - refresh_eff * refresh_eff))
    noise = _randn_like(torch, x, generator)
    refreshed_noise = keep * endpoint_noise + refresh_eff * float(s_noise) * noise
    return deterministic_next + t_next_broadcast * (refreshed_noise - endpoint_noise), True

def flow_heun_step(x, denoised, denoised_pred, t, t_next, *, eps: float = 1e-6):
    """Advance one RF x0 exponential Heun step using two denoised estimates."""

    if float(t_next) <= 0.0:
        return denoised
    if float(t) <= 0.0:
        return denoised

    torch = importlib.import_module("torch")
    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    ratio = t_next_tensor / t_current

    lambda_current = _rf_lambda(torch, t, eps=eps)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - lambda_current
    if _scalar_float(torch, h) <= eps:
        return ratio * x + (1.0 - ratio) * denoised

    c = t_next_tensor * torch.exp(lambda_current)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    endpoint_weight = k1 / h
    current_weight = k0 - endpoint_weight
    return ratio * x + c * (current_weight * denoised + endpoint_weight * denoised_pred)


def flow_pc3_predictor_step(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowPC3State | None = None,
    eps: float = 1e-6,
):
    """Predict an RF PC3 endpoint using AB2 history when it is valid."""

    if float(t_next) <= 0.0:
        return denoised
    if float(t) <= 0.0:
        return denoised

    torch = importlib.import_module("torch")
    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    current_lambda = _rf_lambda(torch, t, eps=eps)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    ratio = t_next_tensor / t_current

    if _scalar_float(torch, h) <= eps:
        return ratio * x + (1.0 - ratio) * denoised

    if state is None or state.previous_denoised is None or state.previous_lambda is None:
        return ratio * x + (1.0 - ratio) * denoised

    h_previous = current_lambda - state.previous_lambda
    if _scalar_float(torch, h_previous) <= eps:
        return ratio * x + (1.0 - ratio) * denoised

    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    current_weight = k0 + k1 / h_previous
    previous_weight = -k1 / h_previous
    return ratio * x + c * (
        current_weight * denoised + previous_weight * state.previous_denoised
    )


def flow_pc3_damped_step(
    x,
    denoised,
    denoised_pred,
    t,
    t_next,
    *,
    state: FlowPC3State | None = None,
    max_gamma: float = 1.0,
    tolerance: float = 0.005,
    eps: float = 1e-6,
):
    """Advance one RF x0 exponential PC3 step with Heun fallback damping."""

    result = flow_pc3_damped_step_result(
        x,
        denoised,
        denoised_pred,
        t,
        t_next,
        state=state,
        max_gamma=max_gamma,
        tolerance=tolerance,
        eps=eps,
    )
    return result.x, result.state


def flow_pc3_damped_step_result(
    x,
    denoised,
    denoised_pred,
    t,
    t_next,
    *,
    state: FlowPC3State | None = None,
    max_gamma: float = 1.0,
    tolerance: float = 0.005,
    eps: float = 1e-6,
) -> FlowPC3StepResult:
    """Advance one RF x0 PC3 step and return diagnostics for cache gates."""

    if state is None:
        state = FlowPC3State()

    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    next_state = FlowPC3State(
        previous_denoised=denoised,
        previous_lambda=current_lambda,
    )

    x_heun = flow_heun_step(x, denoised, denoised_pred, t, t_next, eps=eps)
    if float(t_next) <= 0.0 or float(t) <= 0.0:
        zero = x.new_tensor(0.0)
        return FlowPC3StepResult(x_heun, next_state, x_heun, x_heun, zero, zero)
    if state.previous_denoised is None or state.previous_lambda is None:
        zero = x.new_tensor(0.0)
        return FlowPC3StepResult(x_heun, next_state, x_heun, x_heun, zero, zero)

    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_previous = current_lambda - state.previous_lambda
    if _scalar_float(torch, h) <= eps or _scalar_float(torch, h_previous) <= eps:
        zero = x.new_tensor(0.0)
        return FlowPC3StepResult(x_heun, next_state, x_heun, x_heun, zero, zero)

    gamma_max = max(0.0, min(1.0, float(max_gamma)))
    tol = max(0.0, float(tolerance))
    if gamma_max <= 0.0 or tol <= 0.0:
        zero = x.new_tensor(0.0)
        return FlowPC3StepResult(x_heun, next_state, x_heun, x_heun, zero, zero)

    ratio = t_next_tensor / t_current
    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    k2 = torch.exp(h) * h * h - 2.0 * k1

    previous_node = state.previous_lambda - current_lambda
    previous_weight = (k2 - h * k1) / (previous_node * (previous_node - h))
    current_weight = (
        k2 - (previous_node + h) * k1 + previous_node * h * k0
    ) / (previous_node * h)
    endpoint_weight = (k2 - previous_node * k1) / ((h - previous_node) * h)

    x_am3 = ratio * x + c * (
        previous_weight * state.previous_denoised
        + current_weight * denoised
        + endpoint_weight * denoised_pred
    )

    error = _rms(torch, x_am3 - x_heun) / (_rms(torch, x_heun) + eps)
    tolerance_tensor = x.new_tensor(tol)
    gamma_error = torch.sqrt(torch.clamp(tolerance_tensor / (error + eps), min=0.0))
    gamma_error = torch.clamp(gamma_error, min=0.0, max=1.0)
    gamma_lambda = _flow_pc3_lambda_gate(torch, current_lambda, lambda_next)
    gamma = gamma_max * gamma_lambda * gamma_error
    return FlowPC3StepResult(
        x_heun + gamma * (x_am3 - x_heun),
        next_state,
        x_heun,
        x_am3,
        gamma,
        error,
    )


def flow_3m_damped_step(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowERState | None = None,
    max_gamma: float = 1.0,
    tolerance: float = 0.005,
    force_order_le_2: bool = False,
    eps: float = 1e-6,
) -> Flow3MStepResult:
    """One-eval lambda-native 2M/3M step with damped 3M extrapolation."""

    if state is None:
        state = FlowERState()

    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    next_state = FlowERState(
        previous_denoised=denoised,
        previous_lambda=current_lambda,
        previous_previous_denoised=state.previous_denoised,
        previous_previous_lambda=state.previous_lambda,
    )

    if float(t_next) <= 0.0 or float(t) <= 0.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(denoised, next_state, denoised, denoised, zero, zero, 1, 0.0)

    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_f = _scalar_float(torch, h)
    ratio = t_next_tensor / t_current
    if h_f <= eps:
        x_1m = ratio * x + (1.0 - ratio) * denoised
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 0.0)

    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    x_1m = ratio * x + c * k0 * denoised

    if state.previous_denoised is None or state.previous_lambda is None:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 1.0)

    h_prev = current_lambda - state.previous_lambda
    h_prev_f = _scalar_float(torch, h_prev)
    if h_prev_f <= eps:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_1m, next_state, x_1m, x_1m, zero, zero, 1, 1.0)

    w0_2 = k0 + k1 / h_prev
    w1_2 = -k1 / h_prev
    x_2m = ratio * x + c * (w0_2 * denoised + w1_2 * state.previous_denoised)

    if (
        force_order_le_2
        or state.previous_previous_denoised is None
        or state.previous_previous_lambda is None
    ):
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, 1.0)

    h_prevprev = state.previous_lambda - state.previous_previous_lambda
    h_prevprev_f = _scalar_float(torch, h_prevprev)
    if h_prevprev_f <= eps:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, 1.0)

    r1 = h_f / h_prev_f
    r2 = h_prev_f / h_prevprev_f
    k2 = torch.exp(h) * h * h - 2.0 * k1
    a = h_prev
    b = h_prev + h_prevprev
    w0_3 = k0 + ((a + b) / (a * b)) * k1 + k2 / (a * b)
    w1_3 = -(k2 + b * k1) / (a * (b - a))
    w2_3 = (k2 + a * k1) / (b * (b - a))
    coeff_l1 = _scalar_float(torch, (torch.abs(w0_3) + torch.abs(w1_3) + torch.abs(w2_3)) / (torch.abs(k0) + eps))

    if not (0.50 <= r1 <= 2.00 and 0.50 <= r2 <= 2.00) or h_f > 0.55 or coeff_l1 > 5.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_2m, zero, zero, 2, coeff_l1)

    x_3m = ratio * x + c * (
        w0_3 * denoised
        + w1_3 * state.previous_denoised
        + w2_3 * state.previous_previous_denoised
    )
    e32 = _rms(torch, x_3m - x_2m) / (_rms(torch, x_2m) + eps)

    lambda_next_f = _scalar_float(torch, lambda_next)
    if lambda_next_f < -2.5:
        gamma_cap = 0.65
    elif lambda_next_f > 4.5:
        gamma_cap = 0.70
    else:
        gamma_cap = 0.85
    gamma_cap *= max(0.0, min(1.15, float(max_gamma)))
    gamma_cap = min(gamma_cap, 0.95)
    if not (0.65 <= r1 <= 1.60 and 0.65 <= r2 <= 1.60):
        gamma_cap *= 0.75

    sqrt_t = math.sqrt(max(0.0, min(1.0, _scalar_float(torch, t_next))))
    tau32 = max(0.0, float(tolerance)) * (0.50 + 0.80 * sqrt_t)
    if gamma_cap <= 0.0 or tau32 <= 0.0:
        zero = x.new_tensor(0.0)
        return Flow3MStepResult(x_2m, next_state, x_2m, x_3m, zero, e32, 2, coeff_l1)

    gamma3 = gamma_cap * min(1.0, math.sqrt(tau32 / (_scalar_float(torch, e32) + eps)))
    x_next = x_2m + gamma3 * (x_3m - x_2m)
    return Flow3MStepResult(x_next, next_state, x_2m, x_3m, x.new_tensor(gamma3), e32, 3, coeff_l1)


def _sparse_pc3_phase(step_index: int, total_steps: int, tail_start_step: int | None) -> str:
    if tail_start_step is not None and step_index >= tail_start_step:
        return "tail"
    progress = step_index / max(total_steps - 1, 1)
    if progress < 0.25:
        return "high"
    if progress >= 0.68:
        return "tail"
    return "body"


def _sparse_pc3_budget(total_steps: int) -> dict[str, int]:
    if total_steps <= 30:
        return {"high": 1, "body": 2, "tail": 3}
    if total_steps <= 38:
        return {"high": 1, "body": 3, "tail": 4}
    return {"high": 1, "body": 4, "tail": 5}


def _sparse_pc3_risk(
    torch,
    *,
    denoised,
    t,
    t_next,
    state: FlowERState,
    result: Flow3MStepResult,
    tolerance: float,
    eps: float = 1e-6,
) -> float:
    current_lambda = _rf_lambda(torch, t, eps=eps)
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = _scalar_float(torch, lambda_next - current_lambda)
    sqrt_t = math.sqrt(max(0.0, min(1.0, _scalar_float(torch, t))))
    tau32 = max(eps, float(tolerance) * (0.50 + 0.80 * sqrt_t))
    risk = _scalar_float(torch, result.e32) / tau32

    if state.previous_denoised is not None and state.previous_lambda is not None:
        h_prev = _scalar_float(torch, current_lambda - state.previous_lambda)
        if h_prev > eps:
            c1 = (h / h_prev) * _scalar_float(
                torch,
                _rms(torch, denoised - state.previous_denoised) / (_rms(torch, denoised) + eps),
            )
            tau1 = 0.08 + 0.08 * sqrt_t
            risk = max(risk, 0.5 * c1 / tau1)

    if (
        state.previous_denoised is not None
        and state.previous_lambda is not None
        and state.previous_previous_denoised is not None
        and state.previous_previous_lambda is not None
    ):
        h_prev = _scalar_float(torch, current_lambda - state.previous_lambda)
        h_prevprev = _scalar_float(torch, state.previous_lambda - state.previous_previous_lambda)
        if h_prev > eps and h_prevprev > eps:
            s_i = (denoised - state.previous_denoised) / h_prev
            s_prev = (state.previous_denoised - state.previous_previous_denoised) / h_prevprev
            c2 = _scalar_float(
                torch,
                h
                * h
                * _rms(torch, s_i - s_prev)
                / ((h_prev + h_prevprev) * (_rms(torch, denoised) + eps)),
            )
            tau2 = 0.006 + 0.010 * sqrt_t
            risk = max(risk, c2 / tau2)

    return float(risk)


def _should_do_sparse_pc3(
    step_index: int,
    total_steps: int,
    *,
    phase: str,
    risk: float,
    used: dict[str, int],
    budget: dict[str, int],
    last_pc3_step: int,
    tail_start_step: int | None,
) -> bool:
    if step_index >= total_steps - 1:
        return False
    if used.get(phase, 0) >= budget.get(phase, 0):
        return False

    if phase == "high":
        threshold = 1.6
        min_gap = 3
    elif phase == "body":
        threshold = 1.0
        min_gap = 2
    else:
        threshold = 0.75
        min_gap = 1

    if step_index - last_pc3_step < min_gap:
        return False

    trigger = risk > threshold or risk > 1.8
    if phase == "tail":
        tail_start = tail_start_step if tail_start_step is not None else int(0.68 * total_steps)
        tail_offset = step_index - tail_start
        if tail_offset >= 2 and tail_offset % 3 == 0:
            trigger = True
    return trigger


def flow_er_step(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowERState | None = None,
    max_order: int = 2,
    eps: float = 1e-6,
):
    """Advance one deterministic RF x0 exponential LMS step."""

    if state is None:
        state = FlowERState()

    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    next_state = FlowERState(
        previous_denoised=denoised,
        previous_lambda=current_lambda,
        previous_previous_denoised=state.previous_denoised,
        previous_previous_lambda=state.previous_lambda,
    )

    if float(t_next) <= 0.0:
        return denoised, next_state
    if float(t) <= 0.0:
        return denoised, next_state

    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=0.0)
    order = max(1, min(3, int(max_order)))
    denoised_history = [denoised]
    lambda_history = [current_lambda]
    if state.previous_denoised is not None and state.previous_lambda is not None:
        denoised_history.append(state.previous_denoised)
        lambda_history.append(state.previous_lambda)
    if (
        state.previous_previous_denoised is not None
        and state.previous_previous_lambda is not None
    ):
        denoised_history.append(state.previous_previous_denoised)
        lambda_history.append(state.previous_previous_lambda)
    order = min(order, len(denoised_history))

    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    if _scalar_float(torch, h) <= eps:
        ratio = t_next_tensor / t_current
        return ratio * x + (1.0 - ratio) * denoised, next_state

    if order >= 2 and _scalar_float(torch, current_lambda - lambda_history[1]) <= eps:
        order = 1
    if order >= 3 and _scalar_float(torch, lambda_history[1] - lambda_history[2]) <= eps:
        order = 2
    if order == 1:
        ratio = t_next_tensor / t_current
        return ratio * x + (1.0 - ratio) * denoised, next_state

    c = t_next_tensor * torch.exp(current_lambda)
    k0 = torch.expm1(h)
    k1 = torch.exp(h) * h - k0
    weights = [k0]
    if order == 2:
        a = lambda_history[1] - current_lambda
        weights = [k0 - k1 / a, k1 / a]
    elif order >= 3:
        k2 = torch.exp(h) * h * h - 2.0 * k1
        a = lambda_history[1] - current_lambda
        b = lambda_history[2] - current_lambda
        weights = [
            (k2 - (a + b) * k1 + a * b * k0) / (a * b),
            (k2 - b * k1) / (a * (a - b)),
            (k2 - a * k1) / (b * (b - a)),
        ]

    x_next = (t_next_tensor / t_current) * x
    for weight, denoised_item in zip(weights, denoised_history):
        x_next = x_next + c * weight * denoised_item

    return x_next, next_state


def flow_velocity(x, denoised, t):
    """Convert a denoised x0 prediction into Flow Matching velocity."""

    return (x - denoised) / t


def _broadcast_time(torch, value, like):
    if hasattr(value, "to"):
        out = value.to(device=like.device, dtype=like.dtype)
    else:
        out = torch.tensor(value, device=like.device, dtype=like.dtype)

    while out.ndim < like.ndim:
        out = out[..., None]
    return out


def _scalar_float(torch, value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(torch.as_tensor(value).detach().cpu())


def _rms(torch, value):
    return torch.sqrt(torch.mean(value.float() * value.float()))


def _flow_pc3_lambda_gate(torch, lambda_current, lambda_next):
    high_noise_gate = torch.sigmoid((lambda_current + 2.5) / 0.5)
    low_noise_gate = torch.sigmoid((4.5 - lambda_next) / 0.8)
    return high_noise_gate * low_noise_gate


def _finite_schedule_terminal(torch, sigmas):
    if int(sigmas.shape[0]) >= 2:
        return sigmas[-2]
    return sigmas[-1]


def _hybrid_tail_start_step(
    torch,
    sigmas,
    flow_schedule: str,
    *,
    flow_shift: float = 1.0,
) -> int | None:
    is_rho_tail = str(flow_schedule).startswith("flow_cosmos_rho7_rf_tail_")
    is_shift_tail = _flow_shift_is_active(flow_schedule, flow_shift)
    if not (is_rho_tail or is_shift_tail):
        return None

    finite_count = int(sigmas.shape[0]) - 1
    if finite_count < 3:
        return None

    times = [_scalar_float(torch, sigmas[index]) for index in range(finite_count)]
    ells = [-math.log(max(value, 1e-12)) for value in times]
    gaps = [right - left for left, right in zip(ells, ells[1:])]
    if len(gaps) < 2:
        return None

    for index in range(len(gaps) - 1):
        t_value = min(max(times[index], 1e-12), 1.0 - 1e-12)
        sigma = t_value / (1.0 - t_value)
        lambda_value = -math.log(max(sigma, 1e-12))
        if lambda_value < -1e-6:
            continue

        tail_gaps = gaps[index:]
        average_gap = sum(tail_gaps) / len(tail_gaps)
        tolerance = max(1e-5, abs(average_gap) * 1e-4)
        if max(abs(gap - average_gap) for gap in tail_gaps) <= tolerance:
            return index
    return None


def _as_tensor_like(torch, value, like):
    if hasattr(value, "to"):
        return value.to(device=like.device, dtype=like.dtype)
    return torch.tensor(value, device=like.device, dtype=like.dtype)


def _rf_lambda(torch, t, *, eps: float = 1e-6):
    if hasattr(t, "to"):
        value = t
    else:
        value = torch.as_tensor(t)
    value = torch.clamp(value, min=eps, max=1.0 - eps)
    return torch.log1p(-value) - torch.log(value)


def _rf_external_sigma(torch, t, *, like, eps: float = 1e-6):
    value = _as_tensor_like(torch, t, like)
    value = torch.clamp(value, min=eps, max=1.0 - eps)
    return value / (1.0 - value)


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


def _randn_like(torch, x, generator):
    try:
        return torch.randn(
            x.shape,
            dtype=x.dtype,
            layout=x.layout,
            device=x.device,
            generator=generator,
        )
    except (TypeError, RuntimeError):
        return torch.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)


def _make_generator(torch, device, seed: int):
    try:
        generator = torch.Generator(device=device)
    except Exception:
        try:
            generator = torch.Generator(device="cpu")
        except Exception:
            return None

    try:
        generator.manual_seed(seed)
    except Exception:
        return None
    return generator


def _set_cfg(cfg_guider: Any, value: float) -> None:
    if hasattr(cfg_guider, "set_cfg"):
        cfg_guider.set_cfg(value)
    else:
        cfg_guider.cfg = value


def _clamp01(value: float) -> float:
    if math.isnan(value):
        raise ValueError("progress value must not be NaN")
    return min(1.0, max(0.0, float(value)))
