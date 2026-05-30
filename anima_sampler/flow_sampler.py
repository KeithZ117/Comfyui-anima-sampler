"""Compatibility facade for the refactored Anima Flow sampler modules.

New code should import from the focused modules next to this file. This module
keeps the historical ``anima_sampler.flow_sampler`` import surface stable.
"""

from __future__ import annotations

from .cfg_schedule import _beta_bump_unit, _interval_window, _smoothstep, cfg_at_progress, cfg_schedule_position
from .comfy_runner import run_comfy_anima_sampler, run_comfy_native_sampler
from .flow_constants import CFG_SCHEDULE_DOMAINS, CFG_SCHEDULE_MODES, FLOW_SCHEDULES, FLOW_SOLVERS
from .flow_math import (
    _accelerating_tail_start_step,
    _active_integration_steps,
    _as_tensor_like,
    _broadcast_time,
    _clamp01,
    _finite_schedule_terminal,
    _flow_pc3_lambda_gate,
    _hybrid_tail_start_step,
    _make_generator,
    _randn_like,
    _rf_external_sigma,
    _rf_lambda,
    _rms,
    _scalar_float,
    _set_cfg,
)
from .latent_utils import (
    _find_x_embedder_in_features,
    _first_linear_in_features,
    _infer_cosmos_latent_channels,
    _latent_channel_count,
    _normalize_cosmos_latent,
    _restore_image_latent,
    _restore_sampler_channels,
    _shape_text,
    _torch_cat_like,
)
from .sampler_trace import (
    _append_sampler_trace,
    _init_sampler_stats,
    _record_model_call,
    _relative_rms,
    _sampler_trace_enabled,
    _sampler_trace_phase,
    _stats_mean,
    _stats_percentile,
    _trace_float,
    format_sampler_trace_csv,
    format_sampler_trace_json,
)
from .sampler_types import (
    AnimaSamplerLog,
    Flow3MStepResult,
    FlowERState,
    FlowPC3PredictorResult,
    FlowPC3State,
    FlowPC3StepResult,
    FlowUniPC2State,
    FlowUniPC2StepResult,
    _flow_shift_log_line,
)
from .sampling_loop import _model_current_denoised, sample_anima_flow_corrective
from .sigma_schedule import (
    _build_flow_cosmos_rf_tail_sigmas,
    _build_flow_cosmos_rho7_sigmas,
    _build_flow_cosmos_sigmas,
    _build_rf_denoise_sigmas,
    _describe_model_sampling_shift,
    _exact_logspace_rflow_sigmas,
    _rf_denoise_start_external_sigma,
    _rflow_time_from_external_sigma,
    build_anima_sigmas,
)
from .solvers.basic import flow_euler_step, flow_heun_step, flow_velocity, rf_endpoint_noise_refresh
from .solvers.er import flow_3m_damped_step, flow_ab2_step, flow_er_step
from .solvers.pc3 import (
    _flow_pc3_can_correct,
    _flow_pc3_clamped_gamma,
    _flow_pc3_endpoint_skip_note,
    _flow_pc3_history_depth,
    _flow_pc3_lower_order_final,
    _flow_pc3_next_state,
    _flow_pc3_predictor_max_order,
    _flow_pc3_should_endpoint_correct,
    flow_pc3_damped_step,
    flow_pc3_damped_step_result,
    flow_pc3_predictor_step,
    flow_pc3_predictor_step_result,
)
from .solvers.unipc import (
    _flow_unipc2_x0_correct,
    _flow_unipc2_x0_predict,
    _flow_unipc_available_order,
    _flow_unipc_bh_rhos,
    _flow_unipc_state_history,
    _flow_unipc_threshold_sample,
    _flow_unipc_weighted_sum,
    flow_unipc2_x0_step,
)

__all__ = ['FLOW_SOLVERS', 'FLOW_SCHEDULES', 'CFG_SCHEDULE_DOMAINS', 'CFG_SCHEDULE_MODES', '_flow_shift_log_line', 'AnimaSamplerLog', 'FlowERState', 'FlowPC3State', 'FlowPC3StepResult', 'FlowPC3PredictorResult', 'Flow3MStepResult', 'FlowUniPC2State', 'FlowUniPC2StepResult', 'cfg_at_progress', '_smoothstep', '_beta_bump_unit', '_interval_window', 'cfg_schedule_position', 'build_anima_sigmas', '_build_rf_denoise_sigmas', '_build_flow_cosmos_sigmas', '_build_flow_cosmos_rf_tail_sigmas', '_build_flow_cosmos_rho7_sigmas', '_rf_denoise_start_external_sigma', '_exact_logspace_rflow_sigmas', '_rflow_time_from_external_sigma', '_describe_model_sampling_shift', 'run_comfy_anima_sampler', 'run_comfy_native_sampler', '_normalize_cosmos_latent', '_infer_cosmos_latent_channels', '_latent_channel_count', '_torch_cat_like', '_restore_image_latent', '_shape_text', '_init_sampler_stats', '_record_model_call', '_sampler_trace_enabled', '_trace_float', '_relative_rms', '_append_sampler_trace', 'format_sampler_trace_csv', 'format_sampler_trace_json', '_stats_mean', '_stats_percentile', '_model_current_denoised', 'sample_anima_flow_corrective', 'flow_euler_step', 'flow_ab2_step', 'rf_endpoint_noise_refresh', 'flow_heun_step', 'flow_pc3_predictor_step', 'flow_pc3_predictor_step_result', '_flow_pc3_next_state', '_flow_pc3_can_correct', '_flow_pc3_history_depth', '_flow_pc3_lower_order_final', '_flow_pc3_predictor_max_order', '_flow_pc3_should_endpoint_correct', '_flow_pc3_endpoint_skip_note', '_flow_pc3_clamped_gamma', 'flow_pc3_damped_step', 'flow_pc3_damped_step_result', 'flow_3m_damped_step', 'flow_unipc2_x0_step', '_flow_unipc_state_history', '_flow_unipc_available_order', '_flow_unipc2_x0_predict', '_flow_unipc2_x0_correct', '_flow_unipc_bh_rhos', '_flow_unipc_weighted_sum', '_flow_unipc_threshold_sample', '_sampler_trace_phase', 'flow_er_step', 'flow_velocity', '_broadcast_time', '_scalar_float', '_rms', '_flow_pc3_lambda_gate', '_finite_schedule_terminal', '_active_integration_steps', '_accelerating_tail_start_step', '_hybrid_tail_start_step', '_as_tensor_like', '_rf_lambda', '_rf_external_sigma', '_restore_sampler_channels', '_find_x_embedder_in_features', '_first_linear_in_features', '_randn_like', '_make_generator', '_set_cfg', '_clamp01']
