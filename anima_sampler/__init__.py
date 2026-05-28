"""ComfyUI-loadable Anima sampler package.

This folder can be copied directly into ``ComfyUI/custom_nodes`` as
``custom_nodes/anima_sampler``. The project root ``__init__.py`` also re-exports
the same mappings for users who copy the whole repository folder instead.
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as exc:
    print(f"[anima_sampler] Failed to load ComfyUI nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

from .scheduler import (
    PhaseSteps,
    build_anchored_sigmas,
    build_early_dense_sigmas,
    build_flow_cosmos_beta_sigmas,
    build_flow_cosmos_lambda_biased_sigmas,
    build_flow_cosmos_rho_rf_tail_sigmas,
    build_flow_cosmos_rho_sigmas,
    build_flow_cosmos_shift_rf_tail_sigmas,
    build_flow_cosmos_sigmas,
    build_simple_sigmas,
    early_dense_simple_scheduler,
)
from .flow_sampler import CFG_SCHEDULE_DOMAINS, CFG_SCHEDULE_MODES

__all__ = [
    "PhaseSteps",
    "CFG_SCHEDULE_DOMAINS",
    "CFG_SCHEDULE_MODES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "build_anchored_sigmas",
    "build_early_dense_sigmas",
    "build_flow_cosmos_beta_sigmas",
    "build_flow_cosmos_lambda_biased_sigmas",
    "build_flow_cosmos_rho_rf_tail_sigmas",
    "build_flow_cosmos_rho_sigmas",
    "build_flow_cosmos_shift_rf_tail_sigmas",
    "build_flow_cosmos_sigmas",
    "build_simple_sigmas",
    "early_dense_simple_scheduler",
]
