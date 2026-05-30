"""ComfyUI KSAMPLER loop for Anima Flow solvers."""

from __future__ import annotations

import importlib
from typing import Any

from .cfg_schedule import cfg_at_progress, cfg_schedule_position
from .flow_math import (
    _active_integration_steps,
    _finite_schedule_terminal,
    _hybrid_tail_start_step,
    _make_generator,
    _scalar_float,
    _set_cfg,
)
from .latent_utils import _restore_sampler_channels
from .sampler_trace import (
    _append_sampler_trace,
    _init_sampler_stats,
    _record_model_call,
    _sampler_trace_phase,
)
from .sampler_types import FlowERState, FlowPC3State, FlowUniPC2State
from .solvers.basic import flow_euler_step, flow_heun_step, rf_endpoint_noise_refresh
from .solvers.er import flow_3m_damped_step, flow_ab2_step, flow_er_step
from .solvers.pc3 import (
    _flow_pc3_endpoint_skip_note,
    _flow_pc3_next_state,
    _flow_pc3_predictor_max_order,
    _flow_pc3_should_endpoint_correct,
    flow_pc3_damped_step_result,
    flow_pc3_predictor_step_result,
)
from .solvers.unipc import flow_unipc2_x0_step

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
    flow_rho7_tail_auto: bool = False,
    final_clean_pass: bool = True,
    flow_er_order: int,
    flow_pc3_gamma: float,
    flow_pc3_tolerance: float,
    flow_unipc_order: int = 2,
    flow_unipc_solver_type: str = "bh2",
    flow_unipc_lower_order_final: bool = True,
    flow_unipc_disable_corrector_first: int = 0,
    flow_unipc_thresholding: bool = False,
    flow_unipc_dynamic_thresholding_ratio: float = 0.995,
    flow_unipc_sample_max_value: float = 1.0,
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
    collect_diagnostics: bool = False,
):
    """ComfyUI ``KSAMPLER`` function implementing Flow Matching solvers.

    The model wrapper returns a denoised ``x_0`` estimate. For the flow
    parameterization ``x_t = (1 - t) x_0 + t eps``, velocity is
    ``v = (x_t - x_0) / t``. Euler integration from ``t`` to ``t_next`` is:

    ``x_next = x_t + (t_next - t) * v``.
    """

    torch = importlib.import_module("torch")
    k_sampling = importlib.import_module("comfy.k_diffusion.sampling")

    total_steps = _active_integration_steps(torch, sigmas, final_clean_pass=final_clean_pass)
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
    unipc_state = FlowUniPC2State()
    stats = _init_sampler_stats(sampler_stats, collect_trace=collect_diagnostics)
    hybrid_tail_start_step = _hybrid_tail_start_step(
        torch,
        sigmas,
        flow_schedule,
        flow_shift=flow_shift,
        flow_rho7_tail_auto=flow_rho7_tail_auto,
    )
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
                unipc_state = FlowUniPC2State()
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

            model_calls_before_step = int(stats.get("model_calls", 0)) if stats is not None else None
            t_for_model = t
            used_cached_denoised = False
            cache_score = None
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
                    }
                )

            x_step_start = x
            trace_phase = _sampler_trace_phase(step_index, total_steps, hybrid_tail_start_step)
            if flow_solver == "flow_er" or flow_solver == "flow_ab2":
                if flow_solver == "flow_ab2":
                    x, er_state = flow_ab2_step(
                        x,
                        denoised,
                        t_for_model,
                        t_next,
                        state=er_state,
                    )
                else:
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
                _append_sampler_trace(
                    torch,
                    stats,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver=flow_solver,
                    phase=trace_phase,
                    t=t_for_model,
                    t_next=t_next,
                    cfg=cfg_step,
                    cfg_next=None,
                    x_before=x_step_start,
                    x_after=x,
                    denoised=denoised,
                    cache_used=used_cached_denoised,
                    cache_score=cache_score,
                    endpoint_call=False,
                    predictor_order=2 if flow_solver == "flow_ab2" and er_state.previous_previous_denoised is not None else 1,
                    refresh_applied=refresh_applied,
                    model_calls_before=model_calls_before_step,
                )
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
                    force_order_le_2=False,
                )
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
                _append_sampler_trace(
                    torch,
                    stats,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver=flow_solver,
                    phase=trace_phase,
                    t=t_for_model,
                    t_next=t_next,
                    cfg=cfg_step,
                    cfg_next=None,
                    x_before=x_step_start,
                    x_after=x,
                    denoised=denoised,
                    x_pred=result_3m.x,
                    cache_used=used_cached_denoised,
                    cache_score=cache_score,
                    endpoint_call=False,
                    predictor_order=int(result_3m.order),
                    gamma3=result_3m.gamma3,
                    refresh_applied=refresh_applied,
                    model_calls_before=model_calls_before_step,
                )
                continue

            if flow_solver == "flow_unipc2_x0":
                unipc_result = flow_unipc2_x0_step(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=unipc_state,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver_order=flow_unipc_order,
                    solver_type=flow_unipc_solver_type,
                    lower_order_final=flow_unipc_lower_order_final,
                    disable_corrector=tuple(range(max(0, int(flow_unipc_disable_corrector_first)))),
                    thresholding=flow_unipc_thresholding,
                    dynamic_thresholding_ratio=flow_unipc_dynamic_thresholding_ratio,
                    sample_max_value=flow_unipc_sample_max_value,
                )
                unipc_state = unipc_result.state
                x, refresh_applied = rf_endpoint_noise_refresh(
                    torch,
                    unipc_result.x,
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
                    unipc_state = FlowUniPC2State()
                _append_sampler_trace(
                    torch,
                    stats,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver=flow_solver,
                    phase=trace_phase,
                    t=t_for_model,
                    t_next=t_next,
                    cfg=cfg_step,
                    cfg_next=None,
                    x_before=x_step_start,
                    x_after=x,
                    denoised=denoised,
                    x_pred=unipc_result.x_predictor,
                    x_corrected=unipc_result.x,
                    cache_used=used_cached_denoised,
                    cache_score=cache_score,
                    endpoint_call=False,
                    predictor_order=int(unipc_result.predictor_order),
                    corrector_order=int(unipc_result.corrector_order),
                    refresh_applied=refresh_applied,
                    model_calls_before=model_calls_before_step,
                )
                continue

            if flow_solver == "flow_euler" or float(t_next) <= 0.0:
                x_det = flow_euler_step(x, denoised, t_for_model, t_next)
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
                _append_sampler_trace(
                    torch,
                    stats,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver=flow_solver,
                    phase=trace_phase,
                    t=t_for_model,
                    t_next=t_next,
                    cfg=cfg_step,
                    cfg_next=None,
                    x_before=x_step_start,
                    x_after=x,
                    denoised=denoised,
                    x_pred=x_det,
                    cache_used=used_cached_denoised,
                    cache_score=cache_score,
                    endpoint_call=False,
                    predictor_order=1,
                    refresh_applied=refresh_applied,
                    model_calls_before=model_calls_before_step,
                    note=(
                        "pc3_terminal"
                        if flow_solver == "flow_pc3_damped" and float(t_next) <= 0.0
                        else ""
                    ),
                )
                continue

            if flow_solver not in {"flow_heun", "flow_pc3_damped"}:
                raise ValueError(f"unsupported flow_solver: {flow_solver}")

            x_pred_order = 1
            if flow_solver == "flow_pc3_damped":
                predictor_result = flow_pc3_predictor_step_result(
                    x,
                    denoised,
                    t_for_model,
                    t_next,
                    state=pc3_state,
                    max_order=_flow_pc3_predictor_max_order(step_index, total_steps),
                )
                x_pred = predictor_result.x
                x_pred_order = int(predictor_result.order)
            else:
                x_pred = flow_euler_step(x, denoised, t_for_model, t_next)

            if flow_solver == "flow_pc3_damped" and not _flow_pc3_should_endpoint_correct(
                torch,
                pc3_state,
                x_pred_order,
                step_index,
                total_steps,
                t_next,
            ):
                skip_note = _flow_pc3_endpoint_skip_note(
                    torch,
                    pc3_state,
                    x_pred_order,
                    step_index,
                    total_steps,
                    t_next,
                )
                pc3_state = _flow_pc3_next_state(torch, pc3_state, denoised, t_for_model)
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
                    pc3_state = FlowPC3State()
                _append_sampler_trace(
                    torch,
                    stats,
                    step_index=step_index,
                    total_steps=total_steps,
                    solver=flow_solver,
                    phase=trace_phase,
                    t=t_for_model,
                    t_next=t_next,
                    cfg=cfg_step,
                    cfg_next=None,
                    x_before=x_step_start,
                    x_after=x,
                    denoised=denoised,
                    x_pred=x_pred,
                    cache_used=used_cached_denoised,
                    cache_score=cache_score,
                    endpoint_call=False,
                    predictor_order=x_pred_order,
                    refresh_applied=refresh_applied,
                    model_calls_before=model_calls_before_step,
                    note=skip_note,
                )
                continue

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

            denoised_next = model(
                x_pred,
                t_next * s_in,
                denoise_mask=denoise_mask,
                model_options=model_options,
                seed=seed,
            )
            _record_model_call(stats)
            denoised_next = _restore_sampler_channels(denoised_next, x_pred)
            trace_x_corrected = None
            trace_gamma_pc3 = None
            trace_corrector_order = 0
            if flow_solver == "flow_heun":
                x_det = flow_heun_step(x, denoised, denoised_next, t_for_model, t_next)
                trace_x_corrected = x_det
                trace_corrector_order = 1
            elif flow_solver == "flow_pc3_damped":
                pc3_result = flow_pc3_damped_step_result(
                    x,
                    denoised,
                    denoised_next,
                    t_for_model,
                    t_next,
                    state=pc3_state,
                    max_gamma=flow_pc3_gamma,
                    tolerance=flow_pc3_tolerance,
                    x_pred=x_pred,
                    predictor_order=x_pred_order,
                )
                x_det = pc3_result.x
                trace_x_corrected = pc3_result.x_corrected
                trace_gamma_pc3 = pc3_result.gamma
                trace_corrector_order = int(pc3_result.corrector_order)
                pc3_state = pc3_result.state
                if stats is not None:
                    stats["pc3_used_total"] = int(stats.get("pc3_used_total", 0)) + 1
                    stats[f"pc3_used_{trace_phase}"] = int(stats.get(f"pc3_used_{trace_phase}", 0)) + 1
                    stats.setdefault("gamma_pc3_values", []).append(_scalar_float(torch, pc3_result.gamma))
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
            if flow_solver == "flow_pc3_damped" and refresh_applied:
                pc3_state = FlowPC3State()
            _append_sampler_trace(
                torch,
                stats,
                step_index=step_index,
                total_steps=total_steps,
                solver=flow_solver,
                phase=trace_phase,
                t=t_for_model,
                t_next=t_next,
                cfg=cfg_step,
                cfg_next=cfg_next,
                x_before=x_step_start,
                x_after=x,
                denoised=denoised,
                x_pred=x_pred,
                x_corrected=trace_x_corrected,
                cache_used=used_cached_denoised,
                cache_score=cache_score,
                endpoint_call=True,
                predictor_order=x_pred_order,
                corrector_order=trace_corrector_order,
                gamma=trace_gamma_pc3,
                gamma3=None,
                refresh_applied=refresh_applied,
                model_calls_before=model_calls_before_step,
            )

        if final_clean_pass:
            cfg_clean = cfg_at_progress(
                1.0,
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
                _set_cfg(cfg_guider, cfg_clean)
            clean_sigma = sigmas[total_steps] if total_steps < int(sigmas.shape[0]) else _finite_schedule_terminal(torch, sigmas)
            clean_x_before = x
            clean_calls_before = int(stats.get("model_calls", 0)) if stats is not None else None
            x = _model_current_denoised(
                model,
                x,
                clean_sigma,
                s_in,
                denoise_mask=denoise_mask,
                model_options=model_options,
                seed=seed,
                stats=stats,
            )
            _append_sampler_trace(
                torch,
                stats,
                step_index=total_steps,
                total_steps=total_steps,
                solver=flow_solver,
                phase="clean",
                t=clean_sigma,
                t_next=clean_sigma,
                cfg=cfg_clean,
                cfg_next=None,
                x_before=clean_x_before,
                x_after=x,
                denoised=x,
                x_pred=x,
                cache_used=False,
                endpoint_call=True,
                predictor_order=0,
                corrector_order=0,
                model_calls_before=clean_calls_before,
                note="final_clean_pass",
            )
    finally:
        if can_set_cfg and original_cfg is not None:
            _set_cfg(cfg_guider, original_cfg)

    return x
