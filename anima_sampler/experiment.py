"""Experiment helpers for Anima sampler parameter sweeps."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from .flow_sampler import (
    CFG_SCHEDULE_MODES,
    FLOW_SCHEDULES,
    FLOW_SOLVERS,
)


PARAMETER_SWEEP_KEYS = [
    "seed",
    "steps",
    "flow_solver",
    "flow_er_order",
    "flow_pc3_gamma",
    "flow_pc3_tolerance",
    "flow_schedule",
    "flow_shift",
    "cfg_legacy_progress",
    "denoise_legacy_progress",
    "cosmos_sigma_max",
    "cosmos_sigma_min",
    "cfg",
    "cfg_schedule_mode",
    "cfg_early_scale",
    "cfg_early_ramp_end",
    "cfg_peak_boost",
    "cfg_bump_start",
    "cfg_bump_end",
    "cfg_beta_alpha",
    "cfg_beta_beta",
    "cfg_interval_start",
    "cfg_interval_rise_end",
    "cfg_interval_fall_start",
    "cfg_interval_end",
    "early_cfg_boost",
    "early_cfg_until",
    "rf_endpoint_noise_refresh_enabled",
    "rf_endpoint_noise_refresh_strength",
    "rf_endpoint_noise_refresh_until",
    "late_cfg_scale",
    "late_cfg_start",
]

NO_SECONDARY_SWEEP = "<none>"
PARAMETER_MATRIX_KEYS = [NO_SECONDARY_SWEEP, *PARAMETER_SWEEP_KEYS]

_INTEGER_PARAMETERS = {"seed", "steps", "flow_er_order"}
_BOOLEAN_PARAMETERS = {
    "cfg_legacy_progress",
    "denoise_legacy_progress",
    "rf_endpoint_noise_refresh_enabled",
}
_ENUM_PARAMETERS = {
    "cfg_schedule_mode": CFG_SCHEDULE_MODES,
    "flow_solver": FLOW_SOLVERS,
    "flow_schedule": FLOW_SCHEDULES,
}


def parse_sweep_values(text: str, parameter: str, *, max_runs: int = 8) -> list[int | float | str | bool]:
    """Parse comma/newline/space separated parameter values for a sweep."""

    if parameter not in PARAMETER_SWEEP_KEYS:
        raise ValueError(f"unsupported sweep_parameter: {parameter}")
    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")

    parts = [part for part in re.split(r"[\s,;]+", text.strip()) if part]
    values: list[int | float | str | bool] = []
    for part in parts[:max_runs]:
        if parameter in _ENUM_PARAMETERS:
            if part not in _ENUM_PARAMETERS[parameter]:
                allowed = ", ".join(_ENUM_PARAMETERS[parameter])
                raise ValueError(f"{parameter} must be one of: {allowed}")
            values.append(part)
        elif parameter in _INTEGER_PARAMETERS:
            values.append(_parse_integer_value(part))
        elif parameter in _BOOLEAN_PARAMETERS:
            values.append(_parse_boolean_value(part))
        else:
            value = float(part)
            if math.isnan(value) or math.isinf(value):
                raise ValueError("sweep values must be finite numbers")
            values.append(value)
    return values


def build_parameter_combinations(
    primary_parameter: str,
    primary_values_text: str,
    secondary_parameter: str = NO_SECONDARY_SWEEP,
    secondary_values_text: str = "",
    *,
    max_runs: int = 18,
) -> list[dict[str, int | float | str | bool]]:
    """Build a deterministic sweep list, optionally as a two-parameter matrix."""

    if max_runs < 1:
        raise ValueError("max_runs must be at least 1")
    if primary_parameter not in PARAMETER_SWEEP_KEYS:
        raise ValueError(f"unsupported primary_sweep_parameter: {primary_parameter}")
    if secondary_parameter not in PARAMETER_MATRIX_KEYS:
        raise ValueError(f"unsupported secondary_sweep_parameter: {secondary_parameter}")
    if secondary_parameter == primary_parameter:
        raise ValueError("primary_sweep_parameter and secondary_sweep_parameter must differ")

    primary_values = parse_sweep_values(primary_values_text, primary_parameter, max_runs=max_runs)
    if not primary_values:
        raise ValueError("primary_sweep_values must contain at least one value")

    if secondary_parameter == NO_SECONDARY_SWEEP:
        return [{primary_parameter: value} for value in primary_values[:max_runs]]

    secondary_values = parse_sweep_values(
        secondary_values_text,
        secondary_parameter,
        max_runs=max_runs,
    )
    if not secondary_values:
        raise ValueError("secondary_sweep_values must contain at least one value")

    combinations: list[dict[str, int | float | str | bool]] = []
    for secondary_value in secondary_values:
        for primary_value in primary_values:
            combination = {
                primary_parameter: primary_value,
                secondary_parameter: secondary_value,
            }
            if not _is_allowed_schedule_solver_combination(combination):
                continue
            combinations.append(combination)
            if len(combinations) >= max_runs:
                return combinations
    return combinations


def _is_allowed_schedule_solver_combination(overrides: dict[str, Any]) -> bool:
    """Return whether a schedule/solver pair should be included in a matrix."""

    if "flow_schedule" not in overrides or "flow_solver" not in overrides:
        return True

    return True


def _parse_integer_value(text: str) -> int:
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)

    value = float(text)
    if math.isnan(value) or math.isinf(value):
        raise ValueError("sweep values must be finite numbers")
    if not value.is_integer():
        raise ValueError("integer sweep values must be whole numbers")
    return int(value)


def _parse_boolean_value(text: str) -> bool:
    value = text.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("boolean sweep values must be true/false, 1/0, yes/no, or on/off")


def build_labeled_comparison_grid(
    images: Iterable[Any],
    labels: list[str],
    *,
    columns: int,
    label_height: int,
    gap: int,
):
    """Return a ComfyUI IMAGE tensor containing labeled comparison tiles."""

    image_list = list(images)
    if not image_list:
        raise ValueError("images must contain at least one image tensor")
    if len(image_list) != len(labels):
        raise ValueError("images and labels must have the same length")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    torch = _import_torch()
    pil_images = [_tensor_to_pil(image) for image in image_list]
    tile_w = max(image.width for image in pil_images)
    tile_h = max(image.height for image in pil_images)
    columns = min(columns, len(pil_images))
    rows = math.ceil(len(pil_images) / columns)

    labeled_tiles = [
        _make_labeled_tile(image, label, tile_w=tile_w, tile_h=tile_h, label_height=label_height)
        for image, label in zip(pil_images, labels)
    ]

    grid_w = columns * tile_w + max(0, columns - 1) * gap
    grid_h = rows * (tile_h + label_height) + max(0, rows - 1) * gap

    from PIL import Image

    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    for index, tile in enumerate(labeled_tiles):
        row = index // columns
        col = index % columns
        x = col * (tile_w + gap)
        y = row * (tile_h + label_height + gap)
        grid.paste(tile, (x, y))

    tensor = _pil_to_tensor(grid, torch)
    return tensor


def _make_labeled_tile(image, label: str, *, tile_w: int, tile_h: int, label_height: int):
    from PIL import Image, ImageDraw, ImageFont

    tile = Image.new("RGB", (tile_w, tile_h + label_height), (18, 18, 18))
    x = (tile_w - image.width) // 2
    y = (tile_h - image.height) // 2
    tile.paste(image, (x, y))

    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, tile_h, tile_w, tile_h + label_height), fill=(28, 28, 28))
    font = ImageFont.load_default()
    lines = _wrap_label(label, max_chars=max(20, tile_w // 8))
    text_y = tile_h + 8
    for line in lines:
        if text_y > tile_h + label_height - 12:
            break
        draw.text((10, text_y), line, fill=(238, 238, 238), font=font)
        text_y += 14
    return tile


def _wrap_label(text: str, *, max_chars: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        current = ""
        for word in raw_line.split():
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_chars:
                current += " " + word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _tensor_to_pil(tensor):
    import numpy as np
    from PIL import Image

    if len(tensor.shape) == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().float().clamp(0, 1).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _pil_to_tensor(image, torch):
    import numpy as np

    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _import_torch():
    import torch

    return torch
