"""Math and tensor helpers shared by Flow schedules and solvers."""

from __future__ import annotations

import math
from typing import Any

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
def _active_integration_steps(torch, sigmas, *, final_clean_pass: bool) -> int:
    total = int(sigmas.shape[0]) - 1
    if (
        bool(final_clean_pass)
        and total >= 1
        and _scalar_float(torch, sigmas[-1]) <= 0.0
    ):
        return max(1, total - 1)
    return total
def _accelerating_tail_start_step(torch, sigmas) -> int | None:
    finite_count = int(sigmas.shape[0]) - 1
    if finite_count < 8:
        return None

    times = [_scalar_float(torch, sigmas[index]) for index in range(finite_count)]
    ells = [-math.log(max(value, 1e-12)) for value in times]
    gaps = [right - left for left, right in zip(ells, ells[1:])]
    if len(gaps) < 4:
        return None

    for index in range(3, len(gaps)):
        t_value = min(max(times[index], 1e-12), 1.0 - 1e-12)
        lambda_value = math.log((1.0 - t_value) / t_value)
        if lambda_value < 0.0:
            continue

        history = gaps[max(0, index - 6) : index]
        if len(history) < 3:
            continue
        base_gap = sorted(history)[len(history) // 2]
        if base_gap <= 0.0:
            continue
        local_future = gaps[index : min(len(gaps), index + 3)]
        if gaps[index] >= max(1.65 * base_gap, base_gap + 0.04) and all(
            gap >= 1.25 * base_gap for gap in local_future
        ):
            return index
    return None
def _hybrid_tail_start_step(
    torch,
    sigmas,
    flow_schedule: str,
    *,
    flow_shift: float = 1.0,
    flow_rho7_tail_auto: bool = False,
) -> int | None:
    is_rho_tail = str(flow_schedule) == "flow_cosmos_rho7" and bool(flow_rho7_tail_auto)
    is_shift_tail = str(flow_schedule) == "flow_cosmos_rf_tail"
    is_s_tail = str(flow_schedule) == "flow_rf_linear_s_tail_shift5"
    if is_s_tail:
        return _accelerating_tail_start_step(torch, sigmas)
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
