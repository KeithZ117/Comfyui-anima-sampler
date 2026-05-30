"""CFG schedule helpers for Anima Flow sampling."""

from __future__ import annotations

from .flow_constants import CFG_SCHEDULE_DOMAINS, CFG_SCHEDULE_MODES
from .flow_math import _clamp01, _finite_schedule_terminal, _rf_lambda, _scalar_float

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
    modes keep very early guidance neutral or mild, then either ramp guidance
    upward or add a bounded bump around the structure-forming region.
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

    if mode == "low_to_high":
        start_cfg = base_cfg * early_scale
        ramp = _smoothstep(cfg_interval_start, cfg_interval_rise_end, progress)
        return max(0.0, start_cfg + (base_cfg - start_cfg) * ramp)

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
