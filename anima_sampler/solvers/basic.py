"""Basic one-step Flow solver primitives."""

from __future__ import annotations

import importlib
import math

from ..flow_math import _as_tensor_like, _broadcast_time, _randn_like, _rf_lambda, _scalar_float

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
def flow_velocity(x, denoised, t):
    """Convert a denoised x0 prediction into Flow Matching velocity."""

    return (x - denoised) / t
