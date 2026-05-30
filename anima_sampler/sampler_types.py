"""Sampler log and solver state dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

def _flow_shift_log_line(flow_schedule: str, flow_shift: float) -> str:
    if str(flow_schedule) in {"flow_cosmos_rf_tail", "flow_rf_linear_shift"}:
        return f"flow_shift: {float(flow_shift):.4f}"
    if str(flow_schedule) == "flow_rf_linear_s_tail_shift5":
        return (
            f"flow_shift: {float(flow_shift):.4f} "
            "(ignored; schedule uses fixed shift5)"
        )
    return f"flow_shift: {float(flow_shift):.4f} (ignored by {flow_schedule})"


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
    flow_rho7_tail_auto: bool
    final_clean_pass: bool
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
                f"flow_rho7_tail_auto: {self.flow_rho7_tail_auto}",
                f"final_clean_pass: {self.final_clean_pass}",
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
        }:
            calls = max(1, self.actual_steps * 2 - 1)
            return calls + int(self.final_clean_pass)
        return max(1, self.actual_steps) + int(self.final_clean_pass)
@dataclass
class FlowERState:
    """History needed by the RF x0 LMS multistep corrector."""

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_lambda: Any | None = None
@dataclass
class FlowPC3State:
    """Accepted-state history for the RF x0 exponential PC3 solver.

    Store only x0 predictions evaluated on accepted actual sampler states.
    Endpoint predictions from a predictor state are intentionally not history.
    """

    previous_denoised: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_lambda: Any | None = None
@dataclass
class FlowPC3StepResult:
    """Accepted PC3 step plus diagnostics for logs and tests."""

    x: Any
    state: FlowPC3State
    x_predictor: Any
    x_corrected: Any
    gamma: Any
    error: Any
    predictor_order: int
    corrector_order: int
@dataclass
class FlowPC3PredictorResult:
    """PC3 predictor proposal and the internal order actually used."""

    x: Any
    order: int
@dataclass
class Flow3MStepResult:
    """Accepted one-eval 3M step plus diagnostics."""

    x: Any
    state: FlowERState
    x_2m: Any
    x_3m: Any
    gamma3: Any
    e32: Any
    order: int
    coeff_l1: float
@dataclass
class FlowUniPC2State:
    """History for the RF x0 UniPC predictor/corrector."""

    previous_denoised: Any | None = None
    previous_t: Any | None = None
    previous_lambda: Any | None = None
    previous_previous_denoised: Any | None = None
    previous_previous_t: Any | None = None
    previous_previous_lambda: Any | None = None
    last_sample: Any | None = None
    lower_order_nums: int = 0
    this_order: int = 1
    model_outputs: tuple[Any | None, ...] | None = None
    sigma_history: tuple[Any | None, ...] | None = None
    lambda_history: tuple[Any | None, ...] | None = None
@dataclass
class FlowUniPC2StepResult:
    """Accepted UniPC step plus diagnostics for tests and logs."""

    x: Any
    state: FlowUniPC2State
    x_corrected: Any
    x_predictor: Any
    predictor_order: int
    corrector_order: int
