import unittest

from anima_sampler.nodes import (
    AnimaFlowSettings,
    AnimaFlowMatrixSweep,
    AnimaFlowParameterSweep,
    AnimaFlowCorrectiveSampler,
    AnimaFlowTestPrompt,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _estimated_model_calls,
    _normalize_settings_object,
)
from anima_sampler.flow_sampler import CFG_SCHEDULE_MODES


class NodeRegistrationTests(unittest.TestCase):
    def test_anima_sampler_node_is_registered(self):
        self.assertIn("AnimaFlowCorrectiveSampler", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["AnimaFlowCorrectiveSampler"],
            "Anima Flow Corrective Sampler",
        )

    def test_only_flow_sampler_node_is_registered(self):
        self.assertEqual(
            set(NODE_CLASS_MAPPINGS),
            {
                "AnimaFlowSettings",
                "AnimaFlowCorrectiveSampler",
                "AnimaFlowParameterSweep",
                "AnimaFlowMatrixSweep",
                "AnimaFlowTestPrompt",
            },
        )

    def test_flow_settings_node_is_registered(self):
        self.assertIn("AnimaFlowSettings", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["AnimaFlowSettings"],
            "Anima Flow Settings",
        )

    def test_flow_settings_defaults_match_experimental_baseline(self):
        inputs = AnimaFlowSettings.INPUT_TYPES()["required"]
        self.assertEqual(inputs["steps"][1]["default"], 35)
        self.assertEqual(inputs["cfg"][1]["default"], 6.0)
        self.assertEqual(
            inputs["flow_solver"][0],
            [
                "flow_euler",
                "flow_heun",
                "flow_pc3_damped",
                "flow_pc3_fsal_gated",
                "flow_3m_damped",
                "flow_3m_sparse_pc3_fsal",
                "flow_er",
            ],
        )
        self.assertEqual(inputs["flow_solver"][1]["default"], "flow_pc3_damped")
        self.assertEqual(inputs["flow_er_order"][1]["default"], 2)
        self.assertEqual(inputs["flow_pc3_gamma"][1]["default"], 1.0)
        self.assertEqual(inputs["flow_pc3_tolerance"][1]["default"], 0.005)
        self.assertEqual(
            inputs["flow_schedule"][0],
            [
                "flow_cosmos",
                "flow_cosmos_lambda_biased_strong",
                "flow_cosmos_beta5",
                "flow_cosmos_rho7_rf_tail_auto",
                "simple",
            ],
        )
        self.assertEqual(inputs["flow_schedule"][1]["default"], "flow_cosmos")
        self.assertEqual(inputs["cosmos_sigma_max"][1]["default"], 80.0)
        self.assertEqual(inputs["cosmos_sigma_min"][1]["default"], 0.002)
        self.assertEqual(inputs["cfg_legacy_progress"][0], "BOOLEAN")
        self.assertFalse(inputs["cfg_legacy_progress"][1]["default"])
        self.assertEqual(inputs["cfg_schedule_mode"][0], CFG_SCHEDULE_MODES)
        self.assertEqual(inputs["cfg_schedule_mode"][1]["default"], "beta_bump")
        self.assertEqual(inputs["denoise_legacy_progress"][0], "BOOLEAN")
        self.assertFalse(inputs["denoise_legacy_progress"][1]["default"])
        self.assertEqual(inputs["cfg_early_scale"][1]["default"], 0.98)
        self.assertEqual(inputs["cfg_early_ramp_end"][1]["default"], 0.10)
        self.assertEqual(inputs["cfg_peak_boost"][1]["default"], 0.60)
        self.assertEqual(inputs["cfg_bump_start"][1]["default"], 0.08)
        self.assertEqual(inputs["cfg_bump_end"][1]["default"], 0.68)
        self.assertEqual(inputs["cfg_beta_alpha"][1]["default"], 4.0)
        self.assertEqual(inputs["cfg_beta_beta"][1]["default"], 7.0)
        self.assertEqual(inputs["cfg_interval_start"][1]["default"], 0.12)
        self.assertEqual(inputs["cfg_interval_rise_end"][1]["default"], 0.24)
        self.assertEqual(inputs["cfg_interval_fall_start"][1]["default"], 0.36)
        self.assertEqual(inputs["cfg_interval_end"][1]["default"], 0.58)
        self.assertEqual(inputs["early_cfg_boost"][1]["default"], 0.5)
        self.assertEqual(inputs["early_cfg_until"][1]["default"], 0.30)
        self.assertEqual(inputs["late_cfg_scale"][1]["default"], 0.92)
        self.assertEqual(inputs["late_cfg_start"][1]["default"], 0.76)
        self.assertFalse(inputs["rf_endpoint_noise_refresh_enabled"][1]["default"])
        self.assertEqual(inputs["rf_endpoint_noise_refresh_strength"][1]["default"], 0.15)
        self.assertEqual(inputs["rf_endpoint_noise_refresh_until"][1]["default"], 0.20)

    def test_flow_settings_builds_connection_object(self):
        settings, summary = AnimaFlowSettings().build(
            steps=35,
            cfg=6.0,
            flow_solver="flow_euler",
            flow_er_order=2,
            flow_pc3_gamma=1.0,
            flow_pc3_tolerance=0.005,
            flow_schedule="flow_cosmos",
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
            denoise_legacy_progress=False,
            cfg_legacy_progress=False,
            cfg_schedule_mode="beta_bump",
            cfg_early_scale=0.98,
            cfg_early_ramp_end=0.10,
            cfg_peak_boost=0.60,
            cfg_bump_start=0.08,
            cfg_bump_end=0.68,
            cfg_beta_alpha=4.0,
            cfg_beta_beta=7.0,
            cfg_interval_start=0.12,
            cfg_interval_rise_end=0.24,
            cfg_interval_fall_start=0.36,
            cfg_interval_end=0.58,
            early_cfg_boost=0.5,
            early_cfg_until=0.30,
            late_cfg_scale=0.92,
            late_cfg_start=0.76,
            rf_endpoint_noise_refresh_enabled=False,
            rf_endpoint_noise_refresh_strength=0.15,
            rf_endpoint_noise_refresh_until=0.20,
        )

        self.assertEqual(settings["cfg"], 6.0)
        self.assertEqual(settings["flow_er_order"], 2)
        self.assertEqual(settings["flow_pc3_gamma"], 1.0)
        self.assertEqual(settings["flow_pc3_tolerance"], 0.005)
        self.assertEqual(settings["flow_schedule"], "flow_cosmos")
        self.assertFalse(settings["denoise_legacy_progress"])
        self.assertFalse(settings["cfg_legacy_progress"])
        self.assertEqual(settings["cfg_schedule_mode"], "beta_bump")
        self.assertEqual(settings["cfg_peak_boost"], 0.60)
        self.assertEqual(settings["cfg_bump_start"], 0.08)
        self.assertEqual(settings["cfg_bump_end"], 0.68)
        self.assertFalse(settings["rf_endpoint_noise_refresh_enabled"])
        self.assertEqual(settings["rf_endpoint_noise_refresh_strength"], 0.15)
        self.assertEqual(settings["rf_endpoint_noise_refresh_until"], 0.20)
        self.assertIn("AnimaFlowSettings", summary)
        self.assertIn("steps_semantics: RF integration intervals", summary)
        self.assertIn("estimated_model_calls: 35", summary)
        self.assertIn("cfg_schedule_mode: beta_bump", summary)
        self.assertIn("cfg_peak_boost: 0.6000", summary)

    def test_estimated_model_calls_covers_new_solver_costs(self):
        self.assertEqual(_estimated_model_calls({"steps": 35, "flow_solver": "flow_heun"}), 69)
        self.assertEqual(_estimated_model_calls({"steps": 35, "flow_solver": "flow_pc3_damped"}), 69)
        self.assertEqual(_estimated_model_calls({"steps": 35, "flow_solver": "flow_pc3_fsal_gated"}), 69)
        self.assertEqual(_estimated_model_calls({"steps": 35, "flow_solver": "flow_3m_sparse_pc3_fsal"}), 43)
        self.assertEqual(_estimated_model_calls({"steps": 35, "flow_solver": "flow_3m_damped"}), 35)

    def test_missing_settings_fields_use_current_baseline_defaults(self):
        settings = _normalize_settings_object({"steps": 24})

        self.assertEqual(settings["steps"], 24)
        self.assertEqual(settings["cfg_schedule_mode"], "beta_bump")
        self.assertFalse(settings["cfg_legacy_progress"])

    def test_flow_sampler_requires_settings_and_keeps_runtime_controls(self):
        inputs = AnimaFlowCorrectiveSampler.INPUT_TYPES()["required"]
        self.assertEqual(inputs["flow_settings"][0], "ANIMA_FLOW_SETTINGS")
        self.assertIn("seed", inputs)
        self.assertIn("denoise", inputs)
        self.assertIn("add_noise", inputs)
        self.assertIn("disable_pbar", inputs)

    def test_flow_sampler_hides_algorithm_controls(self):
        inputs = AnimaFlowCorrectiveSampler.INPUT_TYPES()["required"]
        hidden_controls = {
            "steps",
            "cfg",
            "flow_solver",
            "flow_er_order",
            "flow_pc3_gamma",
            "flow_pc3_tolerance",
            "flow_schedule",
            "cosmos_sigma_max",
            "cosmos_sigma_min",
            "denoise_legacy_progress",
            "cfg_legacy_progress",
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
            "late_cfg_scale",
            "late_cfg_start",
            "rf_endpoint_noise_refresh_enabled",
            "rf_endpoint_noise_refresh_strength",
            "rf_endpoint_noise_refresh_until",
        }
        self.assertTrue(hidden_controls.isdisjoint(inputs))

    def test_parameter_sweep_node_is_registered(self):
        self.assertIn("AnimaFlowParameterSweep", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["AnimaFlowParameterSweep"],
            "Anima Flow Parameter Sweep",
        )

    def test_parameter_sweep_exposes_grid_controls(self):
        inputs = AnimaFlowParameterSweep.INPUT_TYPES()["required"]
        self.assertEqual(inputs["flow_settings"][0], "ANIMA_FLOW_SETTINGS")
        self.assertIn("vae", inputs)
        self.assertIn("sweep_parameter", inputs)
        self.assertEqual(inputs["sweep_parameter"][1]["default"], "flow_schedule")
        self.assertIn("sweep_values", inputs)
        self.assertIn("flow_cosmos_lambda_biased_strong", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos,", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_lambda_biased_light", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_lambda_biased,", inputs["sweep_values"][1]["default"])
        self.assertIn("flow_cosmos_rho7_rf_tail_auto", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7,", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_balanced", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_early", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_late", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_dynamic", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flowmatch_euler", inputs["sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_beta5", inputs["sweep_values"][1]["default"])
        self.assertIn("columns", inputs)
        self.assertEqual(inputs["columns"][1]["default"], 3)
        self.assertEqual(inputs["max_runs"][1]["default"], 2)

    def test_parameter_sweep_hides_algorithm_controls(self):
        inputs = AnimaFlowParameterSweep.INPUT_TYPES()["required"]
        self.assertNotIn("steps", inputs)
        self.assertNotIn("cfg", inputs)
        self.assertNotIn("cosmos_sigma_max", inputs)
        self.assertNotIn("cosmos_sigma_min", inputs)
        self.assertNotIn("cfg_schedule_mode", inputs)
        self.assertNotIn("cfg_peak_boost", inputs)
        self.assertNotIn("rf_endpoint_noise_refresh_enabled", inputs)
        self.assertNotIn("rf_endpoint_noise_refresh_strength", inputs)
        self.assertNotIn("rf_endpoint_noise_refresh_until", inputs)

    def test_matrix_sweep_node_is_registered(self):
        self.assertIn("AnimaFlowMatrixSweep", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["AnimaFlowMatrixSweep"],
            "Anima Flow Matrix Sweep",
        )

    def test_matrix_sweep_defaults_to_schedule_by_solver(self):
        inputs = AnimaFlowMatrixSweep.INPUT_TYPES()["required"]
        self.assertEqual(inputs["flow_settings"][0], "ANIMA_FLOW_SETTINGS")
        self.assertIn("vae", inputs)
        self.assertEqual(inputs["primary_sweep_parameter"][1]["default"], "flow_schedule")
        self.assertIn("flow_cosmos_lambda_biased_strong", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos,", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_lambda_biased_light", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_lambda_biased,", inputs["primary_sweep_values"][1]["default"])
        self.assertIn("flow_cosmos_rho7_rf_tail_auto", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7,", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_balanced", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_early", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_late", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_rho7_rf_tail_dynamic", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flowmatch_euler", inputs["primary_sweep_values"][1]["default"])
        self.assertNotIn("flow_cosmos_beta5", inputs["primary_sweep_values"][1]["default"])
        self.assertEqual(inputs["secondary_sweep_parameter"][1]["default"], "flow_solver")
        self.assertIn("flow_pc3_fsal_gated", inputs["secondary_sweep_values"][1]["default"])
        self.assertIn("flow_3m_damped", inputs["secondary_sweep_values"][1]["default"])
        self.assertIn("flow_3m_sparse_pc3_fsal", inputs["secondary_sweep_values"][1]["default"])
        self.assertIn("flow_heun", inputs["secondary_sweep_values"][1]["default"])
        self.assertNotIn("flow_er", inputs["secondary_sweep_values"][1]["default"])
        self.assertNotIn("flow_rho7_euler", inputs["secondary_sweep_values"][1]["default"])
        self.assertFalse(inputs["include_comfy_er_sde_simple"][1]["default"])
        self.assertEqual(inputs["columns"][1]["default"], 5)
        self.assertEqual(inputs["max_runs"][1]["default"], 10)

    def test_matrix_sweep_hides_algorithm_controls(self):
        inputs = AnimaFlowMatrixSweep.INPUT_TYPES()["required"]
        self.assertNotIn("steps", inputs)
        self.assertNotIn("cfg", inputs)
        self.assertNotIn("cosmos_sigma_max", inputs)
        self.assertNotIn("cosmos_sigma_min", inputs)
        self.assertNotIn("cfg_schedule_mode", inputs)
        self.assertNotIn("cfg_peak_boost", inputs)
        self.assertNotIn("rf_endpoint_noise_refresh_enabled", inputs)

    def test_test_prompt_node_outputs_multi_character_stress_prompt(self):
        self.assertIn("AnimaFlowTestPrompt", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["AnimaFlowTestPrompt"],
            "Anima Flow Test Prompt",
        )

        inputs = AnimaFlowTestPrompt.INPUT_TYPES()["required"]
        self.assertEqual(
            inputs["prompt_case"][1]["default"],
            "yuruyuri_4girls_dynamic_festival",
        )

        positive, negative = AnimaFlowTestPrompt().build("yuruyuri_4girls_dynamic_festival")
        self.assertIn("official art", positive)
        self.assertIn("yuru yuri", positive)
        self.assertIn("4girls", positive)
        self.assertIn("huge bouquet", positive)
        self.assertIn("chromatic aberration", positive)
        self.assertIn("depth of field", positive)
        self.assertIn("Kyouko leaping backward", positive)
        self.assertIn("Yui catching the oversized bouquet", positive)
        self.assertIn("wrong body count", negative)


if __name__ == "__main__":
    unittest.main()
