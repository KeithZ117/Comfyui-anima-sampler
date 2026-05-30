"""Shared constants for Anima Flow sampling."""

FLOW_SOLVERS = [
    "flow_euler",
    "flow_ab2",
    "flow_heun",
    "flow_pc3_damped",
    "flow_3m_damped",
    "flow_unipc2_x0",
    "flow_er",
]
FLOW_SCHEDULES = [
    "flow_cosmos",
    "flow_cosmos_rf_tail",
    "flow_cosmos_lambda_biased_strong",
    "flow_cosmos_rho7",
    "flow_rf_linear_shift",
    "flow_rf_linear_s_tail_shift5",
    "simple",
]

CFG_SCHEDULE_DOMAINS = [
    "lambda",
    "rf_t",
    "progress",
]
CFG_SCHEDULE_MODES = [
    "beta_bump",
    "low_to_high",
    "limited_interval",
    "legacy_boost",
    "constant",
]
