"""Sigma schedule builders for Anima corrective sampling.

The scheduler experiments are intentionally narrow:

- derive sigmas from the model's native ``model_sampling.sigmas`` table;
- expose Cosmos/FlowMatch schedule variants for inference;
- optionally spend more transitions in the high-noise region where composition
  forms.

This module does not import ComfyUI. The ComfyUI-facing node can call
``early_dense_simple_scheduler(model_sampling, steps, ...)`` later.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import exp, floor, log
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PhaseSteps:
    """Number of transitions assigned to each denoising phase."""

    early: int
    mid: int
    late: int

    @property
    def total(self) -> int:
        return self.early + self.mid + self.late


def build_simple_sigmas(model_sigmas: Sequence[float], steps: int) -> list[float]:
    """Reproduce ComfyUI's simple scheduler against a sigma table.

    ComfyUI's ``simple_scheduler`` samples the model's native sigma table from
    high noise toward low noise by index, then appends zero as the final sigma.
    If the requested transition count is denser than the native table, this
    interpolates between table entries to avoid repeated no-op sigma steps.
    """

    sigmas = _as_float_list(model_sigmas)
    _validate_steps(steps)
    if not sigmas:
        raise ValueError("model_sigmas must not be empty")

    if steps > len(sigmas) - 1:
        out = [
            _sigma_at_position(sigmas, 1.0 - step / steps, "linear")
            for step in range(steps)
        ]
        out.append(0.0)
        return out

    stride = len(sigmas) / steps
    out = []
    for step in range(steps):
        index = -(1 + int(step * stride))
        out.append(float(sigmas[index]))
    out.append(0.0)
    return out


def build_flow_cosmos_sigmas(
    steps: int,
    *,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a Cosmos RFlow-shaped schedule in normalized Flow time.

    ComfyUI's ``FLOW_COSMOS`` models use external sigmas and convert them to
    rectified-flow time with ``t = sigma / (sigma + 1)``. Anima's model wrapper
    expects the normalized flow time directly, so this helper applies the
    Cosmos shape but returns values in the ``[0, 1]`` range.

    The current implementation uses exact endpoints instead of table
    subsampling.
    """

    _validate_steps(steps)
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")

    external_sigmas = _logspace(sigma_max, sigma_min, steps)
    return [
        _cosmos_rflow_time(sigma, sigma_max=sigma_max)
        for sigma in external_sigmas
    ] + [0.0]


def build_flow_cosmos_beta_sigmas(
    steps: int,
    *,
    beta: float = 0.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a report-aligned beta-shifted Cosmos Flow schedule.

    Cosmos-Predict2.5 reports the training-time shift
    ``t_shift = beta * t / (1 + (beta - 1) * t)``. In sigma-ratio form this is
    equivalent to multiplying external ``sigma = t / (1 - t)`` by ``beta``.
    The returned values are still normalized RF time values.

    ``beta=0`` is a project-level sentinel for "no extra inference shift"; it
    returns the same schedule as ``build_flow_cosmos_sigmas``. The mathematical
    shift formula itself should only be used with positive beta values.

    The current implementation uses exact endpoints instead of table
    subsampling.
    """

    _validate_steps(steps)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if beta == 0.0:
        return build_flow_cosmos_sigmas(
            steps,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
        )

    shifted_sigma_max = sigma_max * beta
    external_sigmas = _logspace(shifted_sigma_max, sigma_min * beta, steps)
    return [
        _cosmos_rflow_time(sigma, sigma_max=shifted_sigma_max)
        for sigma in external_sigmas
    ] + [0.0]


def build_flow_cosmos_lambda_biased_sigmas(
    steps: int,
    *,
    strength: str = "default",
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
    grid_size: int = 20000,
) -> list[float]:
    """Build a lambda-density-shaped Cosmos Flow schedule.

    The schedule keeps the RF endpoints fixed and samples normalized flow time
    through a mild density profile in ``lambda = log((1 - t) / t)``. Higher
    density means smaller local lambda steps around the selected noise regions.
    """

    _validate_steps(steps)
    if strength not in _LAMBDA_BIASED_PROFILES:
        allowed = ", ".join(sorted(_LAMBDA_BIASED_PROFILES))
        raise ValueError(f"strength must be one of: {allowed}")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    if steps == 1:
        return [_cosmos_rflow_time(sigma_max, sigma_max=sigma_max), 0.0]

    lambda_start = -log(sigma_max)
    lambda_end = -log(sigma_min)
    lambda_grid = _linspace(lambda_start, lambda_end, grid_size - 1)
    profile = _LAMBDA_BIASED_PROFILES[strength]
    density = [_lambda_biased_density(value, profile) for value in lambda_grid]

    cdf = [0.0]
    total = 0.0
    for index in range(1, len(lambda_grid)):
        delta = lambda_grid[index] - lambda_grid[index - 1]
        total += 0.5 * (density[index] + density[index - 1]) * delta
        cdf.append(total)
    if total <= 0.0:
        raise ValueError("lambda density integral must be positive")
    cdf = [value / total for value in cdf]

    out = []
    for step in range(steps):
        target = step / (steps - 1)
        lambda_value = _inverse_monotonic_grid(lambda_grid, cdf, target)
        sigma = exp(-lambda_value)
        out.append(_cosmos_rflow_time(sigma, sigma_max=sigma_max))
    out.append(0.0)
    return out


def build_flow_cosmos_rho_sigmas(
    steps: int,
    *,
    order: float = 7.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a Predict2-style rho/order schedule in normalized RF time.

    The archived Predict2 scheduler uses ``sigma_min=0.002``,
    ``sigma_max=80.0``, and ``order=7.0`` for its rectified-flow scheduler.
    This helper exposes that inference grid while keeping this project's final
    terminal entry at ``0.0`` for direct denoise.
    """

    _validate_steps(steps)
    if order <= 0.0:
        raise ValueError("order must be positive")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")

    external_sigmas = _rho_space_descending(
        sigma_max,
        sigma_min,
        steps,
        order=order,
    )
    return [_cosmos_rflow_time(sigma, sigma_max=sigma_max) for sigma in external_sigmas] + [0.0]


def build_flow_cosmos_rho_rf_tail_sigmas(
    steps: int,
    *,
    tail_lambda_start: float | None = 0.5,
    tail_delta_ell_max: float | None = None,
    order: float = 7.0,
    sigma_max: float = 80.0,
    sigma_min: float = 0.002,
) -> list[float]:
    """Build a rho7 high/mid prefix with an RF-native uniform-ell tail.

    ``tail_lambda_start`` chooses the fixed switch point in
    ``lambda = -log(sigma_ext)``. Pass ``tail_delta_ell_max`` instead to use
    an automatic switch: keep the latest possible rho prefix while ensuring the
    resulting low-noise tail is uniform in ``ell`` with a step size no larger
    than the requested RF log-time gap when feasible. ``ell = -log(t)`` is the
    natural exponential-solver time for RF x0 prediction.
    """

    _validate_steps(steps)
    if order <= 0.0:
        raise ValueError("order must be positive")
    if not (0.0 < sigma_min < sigma_max):
        raise ValueError("expected 0 < sigma_min < sigma_max")
    if tail_delta_ell_max is not None and tail_delta_ell_max <= 0.0:
        raise ValueError("tail_delta_ell_max must be positive")
    if tail_lambda_start is None and tail_delta_ell_max is None:
        raise ValueError("expected tail_lambda_start or tail_delta_ell_max")

    reference_rho = _rho_space_descending(
        sigma_max,
        sigma_min,
        steps,
        order=order,
    )

    if steps < 3:
        external_sigmas = reference_rho
    elif tail_delta_ell_max is not None:
        external_sigmas = _rho_with_auto_ell_tail(
            reference_rho,
            sigma_min=sigma_min,
            max_delta_ell=float(tail_delta_ell_max),
        )
    else:
        sigma_switch = exp(-float(tail_lambda_start))
        external_sigmas = _rho_with_fixed_lambda_tail(
            reference_rho,
            sigma_switch=sigma_switch,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    return [_cosmos_rflow_time(sigma, sigma_max=sigma_max) for sigma in external_sigmas] + [0.0]


def _rho_with_fixed_lambda_tail(
    reference_rho: Sequence[float],
    *,
    sigma_switch: float,
    sigma_min: float,
    sigma_max: float,
) -> list[float]:
    steps = len(reference_rho)
    if sigma_switch >= sigma_max:
        external_sigmas = _ell_space_descending(sigma_max, sigma_min, steps)
    elif sigma_switch <= sigma_min or steps < 3:
        external_sigmas = list(reference_rho)
    else:
        prefix = [sigma for sigma in reference_rho if sigma > sigma_switch]
        prefix.append(sigma_switch)
        if len(prefix) > steps - 1:
            prefix = list(reference_rho[: steps - 1])

        tail_count = steps - len(prefix)
        if tail_count < 1:
            return list(reference_rho)
        tail = _ell_space_descending(sigma_switch, sigma_min, tail_count + 1)[1:]
        external_sigmas = prefix + tail
    return external_sigmas


def _rho_with_auto_ell_tail(
    reference_rho: Sequence[float],
    *,
    sigma_min: float,
    max_delta_ell: float,
) -> list[float]:
    latest_valid_index: int | None = None
    ell_end = _ell_from_external_sigma(sigma_min)

    for index in range(len(reference_rho) - 1):
        sigma = reference_rho[index]
        current_lambda = -log(sigma)
        if current_lambda <= 0.0:
            continue

        tail_count = len(reference_rho) - index - 1
        if tail_count < 1:
            continue

        tail_delta_ell = (ell_end - _ell_from_external_sigma(sigma)) / tail_count
        if tail_delta_ell <= max_delta_ell:
            latest_valid_index = index

    if latest_valid_index is not None:
        prefix = list(reference_rho[: latest_valid_index + 1])
        tail_count = len(reference_rho) - len(prefix)
        tail = _ell_space_descending(prefix[-1], sigma_min, tail_count + 1)[1:]
        return prefix + tail

    for index in range(len(reference_rho) - 1):
        sigma = reference_rho[index]
        if -log(sigma) > 0.0:
            prefix = list(reference_rho[: index + 1])
            tail_count = len(reference_rho) - len(prefix)
            tail = _ell_space_descending(prefix[-1], sigma_min, tail_count + 1)[1:]
            return prefix + tail

    return list(reference_rho)


def build_early_dense_sigmas(
    model_sigmas: Sequence[float],
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
    early_end: float = 0.70,
    mid_end: float = 0.22,
    interpolation: str = "linear",
) -> list[float]:
    """Build a simple-derived schedule with extra high-noise transitions.

    ``early_end`` and ``mid_end`` are normalized positions in the native sigma
    table, where ``1.0`` means the highest-noise table entry and ``0.0`` means
    zero. With the defaults, half of the sampling transitions are spent moving
    from the highest sigma down to roughly the top 70 percent of the native
    sigma table.

    The returned list has ``steps + 1`` entries and ends with ``0.0``.
    """

    sigmas = _as_float_list(model_sigmas)
    _validate_steps(steps)
    _validate_boundaries(early_end, mid_end)
    if len(sigmas) < 2:
        raise ValueError("model_sigmas must contain at least two entries")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    phase_steps = allocate_phase_steps(
        steps,
        early_step_ratio=early_step_ratio,
        mid_step_ratio=mid_step_ratio,
    )
    positions = build_phase_positions(
        phase_steps,
        early_end=early_end,
        mid_end=mid_end,
    )
    return [_sigma_at_position(sigmas, pos, interpolation) for pos in positions]


def build_anchored_sigmas(
    model_sigmas: Sequence[float],
    anchor_positions: Sequence[float],
    interval_steps: Sequence[int],
    *,
    interpolation: str = "linear",
) -> list[float]:
    """Build sigmas from explicit descending normalized anchors.

    This is useful for analyzing timestep ideas from other models without
    baking those anchors into Anima defaults. For example, a 20-step Wan-style
    reference schedule can be represented as:

    ``anchor_positions=[1.0, 0.9375, 0.8333333, 0.625, 0.0]`` and
    ``interval_steps=[5, 5, 5, 5]``.
    """

    sigmas = _as_float_list(model_sigmas)
    anchors = _as_float_list(anchor_positions)
    steps = list(interval_steps)

    if len(sigmas) < 2:
        raise ValueError("model_sigmas must contain at least two entries")
    if len(anchors) < 2:
        raise ValueError("anchor_positions must contain at least two entries")
    if len(steps) != len(anchors) - 1:
        raise ValueError("interval_steps must have one entry per anchor interval")
    if any(step < 1 for step in steps):
        raise ValueError("interval_steps entries must be at least 1")
    if anchors[0] != 1.0 or anchors[-1] != 0.0:
        raise ValueError("anchor_positions must start at 1.0 and end at 0.0")
    if any(left <= right for left, right in zip(anchors, anchors[1:])):
        raise ValueError("anchor_positions must be strictly descending")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'")

    positions = build_anchored_positions(anchors, steps)
    return [_sigma_at_position(sigmas, pos, interpolation) for pos in positions]


def build_anchored_positions(
    anchor_positions: Sequence[float],
    interval_steps: Sequence[int],
) -> list[float]:
    """Build descending positions while preserving all supplied anchors."""

    anchors = _as_float_list(anchor_positions)
    steps = list(interval_steps)

    positions: list[float] = []
    for index, count in enumerate(steps):
        segment = _linspace(anchors[index], anchors[index + 1], count)
        if index > 0:
            segment = segment[1:]
        positions.extend(segment)

    positions[-1] = 0.0
    return positions


def early_dense_simple_scheduler(
    model_sampling: object,
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
    early_end: float = 0.70,
    mid_end: float = 0.22,
    interpolation: str = "linear",
):
    """ComfyUI-style scheduler wrapper returning a torch FloatTensor.

    This mirrors ComfyUI scheduler handlers, but keeps the custom path local to
    this project. It imports torch only inside the wrapper so the pure schedule
    functions remain easy to test without a ComfyUI environment.
    """

    if not hasattr(model_sampling, "sigmas"):
        raise ValueError("model_sampling must expose a sigmas attribute")

    values = build_early_dense_sigmas(
        model_sampling.sigmas,
        steps,
        early_step_ratio=early_step_ratio,
        mid_step_ratio=mid_step_ratio,
        early_end=early_end,
        mid_end=mid_end,
        interpolation=interpolation,
    )

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for ComfyUI scheduler output") from exc

    source = model_sampling.sigmas
    device = getattr(source, "device", None)
    dtype = getattr(source, "dtype", None)
    if dtype is None:
        dtype = torch.float32
    return torch.tensor(values, dtype=dtype, device=device)


def allocate_phase_steps(
    steps: int,
    *,
    early_step_ratio: float = 0.50,
    mid_step_ratio: float = 0.32,
) -> PhaseSteps:
    """Allocate integer transition counts to early/mid/late phases."""

    _validate_steps(steps)
    if steps < 3:
        raise ValueError("early-dense scheduling requires at least 3 steps")

    late_step_ratio = 1.0 - early_step_ratio - mid_step_ratio
    ratios = [early_step_ratio, mid_step_ratio, late_step_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("phase step ratios must be positive and sum below 1")

    total_ratio = sum(ratios)
    raw = [steps * ratio / total_ratio for ratio in ratios]
    base = [max(1, floor(value)) for value in raw]

    while sum(base) > steps:
        index = max(range(3), key=lambda i: base[i])
        base[index] -= 1

    remainders = [value - floor(value) for value in raw]
    while sum(base) < steps:
        index = max(range(3), key=lambda i: remainders[i])
        base[index] += 1
        remainders[index] = 0.0

    return PhaseSteps(early=base[0], mid=base[1], late=base[2])


def build_phase_positions(
    phase_steps: PhaseSteps,
    *,
    early_end: float,
    mid_end: float,
) -> list[float]:
    """Build descending normalized sigma-table positions for all phases."""

    _validate_boundaries(early_end, mid_end)
    if phase_steps.total < 1:
        raise ValueError("phase_steps must contain at least one transition")

    positions = _linspace(1.0, early_end, phase_steps.early)
    positions += _linspace(early_end, mid_end, phase_steps.mid)[1:]
    positions += _linspace(mid_end, 0.0, phase_steps.late)[1:]
    positions[-1] = 0.0
    return positions


def _sigma_at_position(
    sigmas: Sequence[float],
    position: float,
    interpolation: str,
) -> float:
    if position <= 0.0:
        return 0.0
    if position >= 1.0:
        return float(sigmas[-1])

    table_index = position * (len(sigmas) - 1)
    if interpolation == "nearest":
        return float(sigmas[round(table_index)])

    lower_index = floor(table_index)
    upper_index = min(lower_index + 1, len(sigmas) - 1)
    alpha = table_index - lower_index
    lower = float(sigmas[lower_index])
    upper = float(sigmas[upper_index])
    return lower + (upper - lower) * alpha


def _linspace(start: float, end: float, transitions: int) -> list[float]:
    if transitions < 1:
        raise ValueError("transitions must be at least 1")
    step = (end - start) / transitions
    return [start + step * index for index in range(transitions + 1)]


def _logspace(start: float, end: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    log_start = log(start)
    log_end = log(end)
    step = (log_end - log_start) / (count - 1)
    return [exp(log_start + step * index) for index in range(count)]


def _rho_space_descending(
    start: float,
    end: float,
    count: int,
    *,
    order: float,
) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    start_root = start ** (1.0 / order)
    end_root = end ** (1.0 / order)
    step = (end_root - start_root) / (count - 1)
    return [(start_root + step * index) ** order for index in range(count)]


def _ell_space_descending(start: float, end: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("count must be at least 1")
    if count == 1:
        return [float(start)]

    ell_start = _ell_from_external_sigma(start)
    ell_end = _ell_from_external_sigma(end)
    step = (ell_end - ell_start) / (count - 1)
    return [_external_sigma_from_ell(ell_start + step * index) for index in range(count)]


def _ell_from_external_sigma(sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    return log(1.0 + 1.0 / float(sigma))


def _external_sigma_from_ell(ell: float) -> float:
    return 1.0 / (exp(float(ell)) - 1.0)


_LAMBDA_BIASED_PROFILES = {
    "light": (
        (0.00, -2.3, 0.7),
        (0.20, 0.8, 1.4),
        (0.08, 3.2, 0.9),
    ),
    "default": (
        (0.10, -2.3, 0.7),
        (0.25, 0.8, 1.4),
        (0.12, 3.2, 0.9),
    ),
    "strong": (
        (0.15, -2.3, 0.7),
        (0.35, 0.8, 1.4),
        (0.18, 3.2, 0.9),
    ),
}


def _lambda_biased_density(value: float, profile: Sequence[tuple[float, float, float]]) -> float:
    density = 1.0
    for amp, center, width in profile:
        if amp <= 0.0:
            continue
        density += amp * exp(-0.5 * ((value - center) / width) ** 2)
    return density


def _inverse_monotonic_grid(values: Sequence[float], cdf: Sequence[float], target: float) -> float:
    if target <= 0.0:
        return float(values[0])
    if target >= 1.0:
        return float(values[-1])

    index = bisect_left(cdf, target)
    if index <= 0:
        return float(values[0])
    if index >= len(cdf):
        return float(values[-1])

    cdf0 = cdf[index - 1]
    cdf1 = cdf[index]
    if cdf1 <= cdf0:
        return float(values[index])

    alpha = (target - cdf0) / (cdf1 - cdf0)
    return float(values[index - 1] + alpha * (values[index] - values[index - 1]))


def _cosmos_rflow_time(sigma: float, *, sigma_max: float) -> float:
    if sigma <= 0.0:
        return 0.0
    sigma = min(float(sigma), float(sigma_max))
    return sigma / (sigma + 1.0)


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _validate_steps(steps: int) -> None:
    if not isinstance(steps, int):
        raise TypeError("steps must be an int")
    if steps < 1:
        raise ValueError("steps must be at least 1")


def _validate_boundaries(early_end: float, mid_end: float) -> None:
    if not (1.0 > early_end > mid_end > 0.0):
        raise ValueError("expected 1.0 > early_end > mid_end > 0.0")
