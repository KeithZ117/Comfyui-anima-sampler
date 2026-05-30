"""Diagnostics and trace formatting for Anima Flow sampling."""

from __future__ import annotations

import json
from typing import Any

from .flow_math import _rf_lambda, _rms, _scalar_float

def _init_sampler_stats(stats: dict[str, Any] | None, *, collect_trace: bool = False) -> dict[str, Any]:
    if stats is None:
        stats = {}
    stats.clear()
    stats.update(
        {
            "collect_trace": bool(collect_trace),
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
    if collect_trace:
        stats["step_trace"] = []
    return stats
def _record_model_call(stats: dict[str, Any] | None):
    if stats is not None:
        stats["model_calls"] = int(stats.get("model_calls", 0)) + 1
def _sampler_trace_enabled(stats: dict[str, Any] | None) -> bool:
    return bool(stats is not None and stats.get("collect_trace"))
def _trace_float(torch, value) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            if int(value.numel()) == 1:
                return float(value.detach().float().cpu().item())
            return float(_rms(torch, value).detach().float().cpu().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None
def _relative_rms(torch, left, right, *, eps: float = 1e-6) -> float | None:
    if left is None or right is None:
        return None
    try:
        return _scalar_float(torch, _rms(torch, left - right) / (_rms(torch, right) + eps))
    except (TypeError, RuntimeError):
        return None
def _append_sampler_trace(
    torch,
    stats: dict[str, Any] | None,
    *,
    step_index: int,
    total_steps: int,
    solver: str,
    phase: str,
    t,
    t_next,
    cfg: float,
    cfg_next: float | None,
    x_before,
    x_after,
    denoised=None,
    x_pred=None,
    x_corrected=None,
    cache_used: bool = False,
    cache_score: float | None = None,
    endpoint_call: bool = False,
    predictor_order: int = 0,
    corrector_order: int = 0,
    gamma=None,
    gamma3=None,
    refresh_applied: bool = False,
    model_calls_before: int | None = None,
    note: str = "",
) -> None:
    if not _sampler_trace_enabled(stats):
        return

    lambda_current = _trace_float(torch, _rf_lambda(torch, t))
    lambda_next = _trace_float(torch, _rf_lambda(torch, t_next))
    trace = {
        "step": int(step_index),
        "total_steps": int(total_steps),
        "solver": str(solver),
        "phase": str(phase),
        "t": _trace_float(torch, t),
        "t_next": _trace_float(torch, t_next),
        "lambda": lambda_current,
        "lambda_next": lambda_next,
        "lambda_gap": (
            None
            if lambda_current is None or lambda_next is None
            else float(lambda_next - lambda_current)
        ),
        "cfg": float(cfg),
        "cfg_next": None if cfg_next is None else float(cfg_next),
        "model_calls_before": model_calls_before,
        "model_calls_after": int(stats.get("model_calls", 0)),
        "cache_used": bool(cache_used),
        "cache_score": cache_score,
        "endpoint_call": bool(endpoint_call),
        "predictor_order": int(predictor_order),
        "corrector_order": int(corrector_order),
        "gamma_pc3": _trace_float(torch, gamma),
        "gamma3": _trace_float(torch, gamma3),
        "update_rel_rms": _relative_rms(torch, x_after, x_before),
        "predictor_rel_rms": _relative_rms(torch, x_pred, x_before),
        "accepted_vs_predictor_rel_rms": _relative_rms(torch, x_after, x_pred),
        "correction_rel_rms": _relative_rms(torch, x_corrected, x_pred),
        "denoised_rel_rms": _relative_rms(torch, denoised, x_before),
        "refresh_applied": bool(refresh_applied),
        "note": str(note),
    }
    stats.setdefault("step_trace", []).append(trace)
def format_sampler_trace_csv(stats: dict[str, Any] | None) -> str:
    if not stats:
        return "step,total_steps,solver,phase,t,t_next,lambda,lambda_next,lambda_gap,cfg,cfg_next,model_calls_before,model_calls_after,cache_used,cache_score,endpoint_call,predictor_order,corrector_order,gamma_pc3,gamma3,update_rel_rms,predictor_rel_rms,accepted_vs_predictor_rel_rms,correction_rel_rms,denoised_rel_rms,refresh_applied,note"
    rows = list(stats.get("step_trace", []))
    columns = [
        "step",
        "total_steps",
        "solver",
        "phase",
        "t",
        "t_next",
        "lambda",
        "lambda_next",
        "lambda_gap",
        "cfg",
        "cfg_next",
        "model_calls_before",
        "model_calls_after",
        "cache_used",
        "cache_score",
        "endpoint_call",
        "predictor_order",
        "corrector_order",
        "gamma_pc3",
        "gamma3",
        "update_rel_rms",
        "predictor_rel_rms",
        "accepted_vs_predictor_rel_rms",
        "correction_rel_rms",
        "denoised_rel_rms",
        "refresh_applied",
        "note",
    ]
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for key in columns:
            value = row.get(key)
            if value is None:
                values.append("")
            elif isinstance(value, bool):
                values.append("1" if value else "0")
            elif isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                text = str(value).replace('"', '""')
                values.append(f'"{text}"' if "," in text or "\n" in text else text)
        lines.append(",".join(values))
    return "\n".join(lines)
def format_sampler_trace_json(stats: dict[str, Any] | None) -> str:
    rows = [] if not stats else list(stats.get("step_trace", []))
    return json.dumps(rows, ensure_ascii=True, indent=2)
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
def _sampler_trace_phase(step_index: int, total_steps: int, tail_start_step: int | None) -> str:
    if tail_start_step is not None and step_index >= tail_start_step:
        return "tail"
    progress = step_index / max(total_steps - 1, 1)
    if progress < 0.25:
        return "high"
    if progress >= 0.68:
        return "tail"
    return "body"
