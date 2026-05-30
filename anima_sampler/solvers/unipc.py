"""UniPC-style RF x0 Flow solver."""

from __future__ import annotations

import importlib
from typing import Any

from ..flow_math import _as_tensor_like, _rf_lambda, _scalar_float
from ..sampler_types import FlowUniPC2State, FlowUniPC2StepResult
from .basic import flow_euler_step

def flow_unipc2_x0_step(
    x,
    denoised,
    t,
    t_next,
    *,
    state: FlowUniPC2State | None = None,
    step_index: int | None = None,
    total_steps: int | None = None,
    solver_order: int = 2,
    solver_type: str = "bh2",
    lower_order_final: bool = True,
    disable_corrector: tuple[int, ...] = (),
    thresholding: bool = False,
    dynamic_thresholding_ratio: float = 0.995,
    sample_max_value: float = 1.0,
    eps: float = 1e-6,
) -> FlowUniPC2StepResult:
    """Advance one RF x0 UniPC step with Cosmos 2.5's BH update structure.

    Cosmos 2.5's FlowUniPC scheduler predicts flow/velocity and converts it to
    x0 internally. ComfyUI's Anima FLOW wrapper already returns denoised x0, so
    this implements the same predict-x0 UniP/UniC BH update directly in x0
    space while preserving the official order warmup, lower-order final,
    disable-corrector, solver-type, and thresholding switches.
    """

    if state is None:
        state = FlowUniPC2State()
    solver_order = int(solver_order)
    if solver_order <= 0:
        raise ValueError("solver_order must be positive")
    solver_type = str(solver_type)
    if solver_type not in {"bh1", "bh2"}:
        raise ValueError("solver_type must be 'bh1' or 'bh2'")
    if step_index is None:
        step_index = int(state.lower_order_nums)
    else:
        step_index = int(step_index)
    if total_steps is None:
        total_steps = max(step_index + 1, step_index + solver_order + 1)
    else:
        total_steps = int(total_steps)

    torch = importlib.import_module("torch")
    current_lambda = _rf_lambda(torch, t, eps=eps)
    current_t = torch.clamp(_as_tensor_like(torch, t, x), min=eps, max=1.0 - eps)
    current_model_output = denoised
    if thresholding:
        current_model_output = _flow_unipc_threshold_sample(
            torch,
            current_model_output,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
            sample_max_value=sample_max_value,
        )

    model_outputs, sigma_history, lambda_history = _flow_unipc_state_history(
        state,
        solver_order=solver_order,
    )

    x_corrected = x
    corrector_order = 0
    use_corrector = (
        step_index > 0
        and (step_index - 1) not in set(int(value) for value in disable_corrector)
        and state.last_sample is not None
        and model_outputs[-1] is not None
        and sigma_history[-1] is not None
        and lambda_history[-1] is not None
    )
    if use_corrector:
        corrector_order = min(
            max(1, int(state.this_order)),
            _flow_unipc_available_order(model_outputs, sigma_history, lambda_history),
        )
        x_corrected = _flow_unipc2_x0_correct(
            torch,
            last_sample=state.last_sample,
            this_sample=x,
            current_model_output=current_model_output,
            current_t=current_t,
            current_lambda=current_lambda,
            model_outputs=model_outputs,
            sigma_history=sigma_history,
            lambda_history=lambda_history,
            order=corrector_order,
            solver_type=solver_type,
            eps=eps,
        )

    model_outputs = [*model_outputs[1:], current_model_output]
    sigma_history = [*sigma_history[1:], current_t]
    lambda_history = [*lambda_history[1:], current_lambda]

    x_predictor = x
    if float(t_next) <= 0.0 or float(t) <= 0.0:
        x_next = current_model_output
        x_predictor = x_next
        predictor_order = 1
    else:
        if lower_order_final:
            this_order = min(solver_order, max(1, total_steps - step_index))
        else:
            this_order = solver_order
        predictor_order = min(
            max(1, this_order),
            int(state.lower_order_nums) + 1,
            _flow_unipc_available_order(model_outputs, sigma_history, lambda_history),
        )
        x_predictor = _flow_unipc2_x0_predict(
            torch,
            x=x,
            model_outputs=model_outputs,
            sigma_history=sigma_history,
            lambda_history=lambda_history,
            t=current_t,
            t_next=t_next,
            order=predictor_order,
            solver_type=solver_type,
            eps=eps,
        )
        x_next = _flow_unipc2_x0_predict(
            torch,
            x=x_corrected,
            model_outputs=model_outputs,
            sigma_history=sigma_history,
            lambda_history=lambda_history,
            t=current_t,
            t_next=t_next,
            order=predictor_order,
            solver_type=solver_type,
            eps=eps,
        )

    next_state = FlowUniPC2State(
        previous_denoised=current_model_output,
        previous_t=current_t,
        previous_lambda=current_lambda,
        previous_previous_denoised=state.previous_denoised,
        previous_previous_t=state.previous_t,
        previous_previous_lambda=state.previous_lambda,
        last_sample=x_corrected,
        lower_order_nums=min(solver_order, int(state.lower_order_nums) + 1),
        this_order=predictor_order,
        model_outputs=tuple(model_outputs),
        sigma_history=tuple(sigma_history),
        lambda_history=tuple(lambda_history),
    )
    return FlowUniPC2StepResult(
        x_next,
        next_state,
        x_corrected,
        x_predictor,
        predictor_order,
        corrector_order,
    )
def _flow_unipc_state_history(
    state: FlowUniPC2State,
    *,
    solver_order: int,
) -> tuple[list[Any | None], list[Any | None], list[Any | None]]:
    if state.model_outputs is not None:
        model_outputs = list(state.model_outputs)
        sigma_history = list(state.sigma_history or [])
        lambda_history = list(state.lambda_history or [])
    else:
        model_outputs = [state.previous_previous_denoised, state.previous_denoised]
        sigma_history = [state.previous_previous_t, state.previous_t]
        lambda_history = [state.previous_previous_lambda, state.previous_lambda]

    def fit_length(values: list[Any | None]) -> list[Any | None]:
        if len(values) < solver_order:
            return [None] * (solver_order - len(values)) + values
        if len(values) > solver_order:
            return values[-solver_order:]
        return values

    return fit_length(model_outputs), fit_length(sigma_history), fit_length(lambda_history)
def _flow_unipc_available_order(
    model_outputs: list[Any | None],
    sigma_history: list[Any | None],
    lambda_history: list[Any | None],
) -> int:
    count = 0
    for model_output, sigma, lambda_value in zip(
        reversed(model_outputs),
        reversed(sigma_history),
        reversed(lambda_history),
    ):
        if model_output is None or sigma is None or lambda_value is None:
            break
        count += 1
    return max(1, count)
def _flow_unipc2_x0_predict(
    torch,
    *,
    x,
    model_outputs,
    sigma_history,
    lambda_history,
    t,
    t_next,
    order: int,
    solver_type: str,
    eps: float,
):
    t_current = torch.clamp(_as_tensor_like(torch, t, x), min=eps, max=1.0 - eps)
    t_next_tensor = torch.clamp(_as_tensor_like(torch, t_next, x), min=eps, max=1.0 - eps)
    current_lambda = lambda_history[-1]
    lambda_next = _rf_lambda(torch, t_next, eps=eps)
    h = lambda_next - current_lambda
    h_f = _scalar_float(torch, h)
    denoised = model_outputs[-1]
    if h_f <= eps:
        return flow_euler_step(x, denoised, t_current, t_next_tensor)

    alpha_next = 1.0 - t_next_tensor
    hh = -h
    rks = []
    d1s = []
    if order >= 2:
        for index in range(1, order):
            previous_model_output = model_outputs[-(index + 1)]
            previous_lambda = lambda_history[-(index + 1)]
            rk = (previous_lambda - current_lambda) / h
            if abs(_scalar_float(torch, rk)) <= eps:
                order = 1
                rks = []
                d1s = []
                break
            rks.append(rk)
            d1s.append((previous_model_output - denoised) / rk)
    rks.append(x.new_tensor(1.0))
    h_phi_1, b_h, rhos_p = _flow_unipc_bh_rhos(
        torch,
        hh,
        rks=rks,
        order=order,
        solver_type=solver_type,
        like=x,
        predictor=True,
        eps=eps,
    )
    base = (t_next_tensor / t_current) * x - alpha_next * h_phi_1 * denoised
    if order < 2:
        return base

    pred_res = _flow_unipc_weighted_sum(rhos_p, d1s)
    return base - alpha_next * b_h * pred_res
def _flow_unipc2_x0_correct(
    torch,
    *,
    last_sample,
    this_sample,
    current_model_output,
    current_t,
    current_lambda,
    model_outputs,
    sigma_history,
    lambda_history,
    order: int,
    solver_type: str,
    eps: float,
):
    previous_model_output = model_outputs[-1]
    previous_t = sigma_history[-1]
    previous_lambda = lambda_history[-1]
    previous_t_tensor = torch.clamp(_as_tensor_like(torch, previous_t, last_sample), min=eps, max=1.0 - eps)
    current_t_tensor = torch.clamp(_as_tensor_like(torch, current_t, last_sample), min=eps, max=1.0 - eps)
    h = current_lambda - previous_lambda
    h_f = _scalar_float(torch, h)
    if h_f <= eps:
        return this_sample

    alpha_current = 1.0 - current_t_tensor
    hh = -h
    rks = []
    d1s = []
    if order >= 2:
        for index in range(1, order):
            earlier_model_output = model_outputs[-(index + 1)]
            earlier_lambda = lambda_history[-(index + 1)]
            rk = (earlier_lambda - previous_lambda) / h
            if abs(_scalar_float(torch, rk)) <= eps:
                order = 1
                rks = []
                d1s = []
                break
            rks.append(rk)
            d1s.append((earlier_model_output - previous_model_output) / rk)
    rks.append(last_sample.new_tensor(1.0))
    h_phi_1, b_h, rhos_c = _flow_unipc_bh_rhos(
        torch,
        hh,
        rks=rks,
        order=order,
        solver_type=solver_type,
        like=last_sample,
        predictor=False,
        eps=eps,
    )
    base = (current_t_tensor / previous_t_tensor) * last_sample - alpha_current * h_phi_1 * previous_model_output
    d1_current = current_model_output - previous_model_output

    corr_res = _flow_unipc_weighted_sum(rhos_c[:-1], d1s)
    return base - alpha_current * b_h * (corr_res + rhos_c[-1] * d1_current)
def _flow_unipc_bh_rhos(
    torch,
    hh,
    *,
    rks,
    order: int,
    solver_type: str,
    like,
    predictor: bool,
    eps: float,
):
    hh = hh.to(device=like.device, dtype=like.dtype) if hasattr(hh, "to") else like.new_tensor(float(hh))
    if abs(_scalar_float(torch, hh)) <= eps:
        h_phi_1 = hh
    else:
        h_phi_1 = torch.expm1(hh)

    if solver_type == "bh1":
        b_h = hh
    elif solver_type == "bh2":
        b_h = torch.expm1(hh)
    else:
        raise ValueError("solver_type must be 'bh1' or 'bh2'")

    if order <= 1:
        return h_phi_1, b_h, [like.new_tensor(0.5)]
    if abs(_scalar_float(torch, b_h)) <= eps:
        return h_phi_1, b_h, [like.new_tensor(0.5)] * (order - (1 if predictor else 0))

    if predictor and order == 2:
        return h_phi_1, b_h, [like.new_tensor(0.5)]
    if (not predictor) and order == 1:
        return h_phi_1, b_h, [like.new_tensor(0.5)]

    rks = torch.stack(
        [
            value.to(device=like.device, dtype=like.dtype)
            if hasattr(value, "to")
            else like.new_tensor(float(value))
            for value in rks
        ]
    )

    rows = []
    rhs = []
    h_phi_k = h_phi_1 / hh - 1.0
    factorial_i = 1
    for index in range(1, order + 1):
        rows.append(torch.pow(rks, index - 1))
        rhs.append(h_phi_k * factorial_i / b_h)
        factorial_i *= index + 1
        h_phi_k = h_phi_k / hh - 1.0 / factorial_i
    matrix = torch.stack(rows)
    vector = torch.stack(rhs).to(device=like.device, dtype=like.dtype)
    if predictor:
        rhos = torch.linalg.solve(matrix[:-1, :-1], vector[:-1])
    else:
        rhos = torch.linalg.solve(matrix, vector)
    return h_phi_1, b_h, [value.to(device=like.device, dtype=like.dtype) for value in rhos]
def _flow_unipc_weighted_sum(weights, values):
    if not values:
        return 0.0
    out = values[0] * weights[0]
    for weight, value in zip(weights[1:], values[1:]):
        out = out + weight * value
    return out
def _flow_unipc_threshold_sample(
    torch,
    sample,
    *,
    dynamic_thresholding_ratio: float,
    sample_max_value: float,
):
    dtype = sample.dtype
    value = sample.float() if dtype not in (torch.float32, torch.float64) else sample
    batch_size = int(value.shape[0])
    flat = value.reshape(batch_size, -1)
    thresholds = torch.quantile(flat.abs(), float(dynamic_thresholding_ratio), dim=1)
    thresholds = torch.clamp(thresholds, min=1.0, max=float(sample_max_value)).reshape(batch_size, 1)
    flat = torch.clamp(flat, -thresholds, thresholds) / thresholds
    return flat.reshape(value.shape).to(dtype)
