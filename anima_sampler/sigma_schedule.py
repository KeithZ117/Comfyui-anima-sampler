"""Sigma/time schedule builders for Anima Flow sampling."""

from __future__ import annotations

import importlib
import math
from typing import Any

from .flow_constants import FLOW_SCHEDULES
from .scheduler import (
    build_flow_cosmos_lambda_biased_sigmas,
    build_flow_cosmos_rho_rf_tail_sigmas,
    build_flow_cosmos_rho_sigmas,
    build_flow_cosmos_shift_rf_tail_sigmas,
    build_flow_cosmos_sigmas,
    build_flow_rf_linear_s_tail_shift5_sigmas,
    build_flow_rf_linear_shift_sigmas,
    build_simple_sigmas,
)

def build_anima_sigmas(
    model: Any,
    steps: int,
    *,
    denoise: float,
    flow_schedule: str,
    flow_shift: float = 1.0,
    flow_rho7_tail_auto: bool = False,
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
    if flow_schedule == "flow_cosmos_rho7_rf_tail_auto":
        flow_schedule = "flow_cosmos_rho7"
        flow_rho7_tail_auto = True
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
            flow_rho7_tail_auto=flow_rho7_tail_auto,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule == "flow_cosmos":
        values = _build_flow_cosmos_sigmas(
            schedule_steps,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule == "flow_cosmos_rf_tail":
        values = _build_flow_cosmos_rf_tail_sigmas(
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
    elif flow_schedule == "flow_cosmos_rho7":
        values = _build_flow_cosmos_rho7_sigmas(
            schedule_steps,
            flow_rho7_tail_auto=flow_rho7_tail_auto,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    elif flow_schedule == "flow_rf_linear_shift":
        values = build_flow_rf_linear_shift_sigmas(
            schedule_steps,
            shift=flow_shift,
        )
    elif flow_schedule == "flow_rf_linear_s_tail_shift5":
        values = build_flow_rf_linear_s_tail_shift5_sigmas(schedule_steps)
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
    flow_rho7_tail_auto: bool,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
) -> list[float]:
    if flow_schedule == "flow_cosmos":
        sigma_start = _rf_denoise_start_external_sigma(
            denoise,
            sigma_max=cosmos_sigma_max,
            sigma_min=cosmos_sigma_min,
        )
        return _build_flow_cosmos_sigmas(
            steps,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
            sigma_start=sigma_start,
        )

    if flow_schedule == "flow_cosmos_rf_tail":
        start_sigma_max = cosmos_sigma_max * flow_shift
        sigma_start = _rf_denoise_start_external_sigma(
            denoise,
            sigma_max=start_sigma_max,
            sigma_min=cosmos_sigma_min,
        )
        return _build_flow_cosmos_rf_tail_sigmas(
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
    if flow_schedule == "flow_cosmos_rho7":
        return _build_flow_cosmos_rho7_sigmas(
            steps,
            flow_rho7_tail_auto=flow_rho7_tail_auto,
            cosmos_sigma_max=sigma_start,
            cosmos_sigma_min=cosmos_sigma_min,
        )
    raise ValueError(f"unsupported RF denoise flow_schedule: {flow_schedule}")
def _build_flow_cosmos_sigmas(
    steps: int,
    *,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
    sigma_start: float | None = None,
) -> list[float]:
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
def _build_flow_cosmos_rf_tail_sigmas(
    steps: int,
    *,
    flow_shift: float,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
    sigma_start: float | None = None,
) -> list[float]:
    return build_flow_cosmos_shift_rf_tail_sigmas(
        steps,
        beta=float(flow_shift),
        sigma_max=cosmos_sigma_max,
        sigma_min=cosmos_sigma_min,
        sigma_start=sigma_start,
    )
def _build_flow_cosmos_rho7_sigmas(
    steps: int,
    *,
    flow_rho7_tail_auto: bool,
    cosmos_sigma_max: float,
    cosmos_sigma_min: float,
) -> list[float]:
    if flow_rho7_tail_auto:
        return build_flow_cosmos_rho_rf_tail_sigmas(
            steps,
            order=7.0,
            sigma_max=cosmos_sigma_max,
            sigma_min=cosmos_sigma_min,
            tail_lambda_start=None,
            tail_delta_ell_max=0.5,
        )
    return build_flow_cosmos_rho_sigmas(
        steps,
        order=7.0,
        sigma_max=cosmos_sigma_max,
        sigma_min=cosmos_sigma_min,
    )
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
            rf_linear_schedules = {
                "flow_rf_linear_shift",
                "flow_rf_linear_s_tail_shift5",
            }
            bypasses_native_shift = (
                flow_schedule.startswith("flow_cosmos")
                or flow_schedule in rf_linear_schedules
            )
            if bypasses_native_shift:
                return f"{class_name}: native shift={shift_text} present, bypassed by {flow_schedule}"
            return f"{class_name}: native shift={shift_text} baked into sigmas"
        return f"{class_name}: shift={shift_text}"

    if class_name == "ModelSamplingCosmosRFlow":
        return f"{class_name}: sigma-ratio flow table"
    return f"{class_name}: no shift attribute"
