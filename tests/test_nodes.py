import unittest

from anima_sampler.nodes import (
    ANIMA_FLOW_BASELINE,
    ANIMA_FLOW_SETTINGS,
    AnimaFlowCorrectiveSampler,
    AnimaFlowSettings,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PUBLIC_CFG_MODES,
    _apply_public_cfg_mode,
    _normalize_settings_object,
)


class NodeRegistrationTests(unittest.TestCase):
    def test_release_nodes_are_registered(self):
        self.assertEqual(
            NODE_CLASS_MAPPINGS,
            {
                "AnimaFlowSettings": AnimaFlowSettings,
                "AnimaFlowCorrectiveSampler": AnimaFlowCorrectiveSampler,
            },
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS,
            {
                "AnimaFlowSettings": "Anima Flow Settings",
                "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
            },
        )

    def test_experiment_nodes_are_not_registered(self):
        removed_nodes = {
            "AnimaFlowParameterSweep",
            "AnimaFlowMatrixSweep",
            "AnimaFlowThreeRoundRho7Test",
            "AnimaFlowCFGWindowTest",
            "AnimaFlowShiftRecoveryTest",
            "AnimaFlowTestPrompt",
        }

        self.assertTrue(removed_nodes.isdisjoint(NODE_CLASS_MAPPINGS))
        self.assertTrue(removed_nodes.isdisjoint(NODE_DISPLAY_NAME_MAPPINGS))

    def test_sampler_exposes_daily_controls_and_optional_settings(self):
        input_types = AnimaFlowCorrectiveSampler.INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]

        self.assertEqual(required["steps"][1]["default"], 35)
        self.assertEqual(required["cfg"][1]["default"], 6.0)
        self.assertEqual(required["cfg_mode"][0], PUBLIC_CFG_MODES)
        self.assertEqual(required["cfg_mode"][1]["default"], "bump cfg")
        self.assertEqual(required["flow_solver"][1]["default"], "flow_pc3_damped")
        self.assertEqual(required["flow_schedule"][1]["default"], "flow_cosmos")
        self.assertEqual(required["flow_shift"][1]["default"], 5.0)
        self.assertIn("seed", required)
        self.assertIn("denoise", required)
        self.assertIn("add_noise", required)
        self.assertNotIn("disable_pbar", required)
        self.assertEqual(optional["flow_settings"][0], ANIMA_FLOW_SETTINGS)
        self.assertNotIn("flow_settings", required)

        hidden_controls = {
            "flow_er_order",
            "flow_pc3_gamma",
            "flow_pc3_tolerance",
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
            "late_cfg_scale",
            "late_cfg_start",
            "rf_endpoint_noise_refresh_enabled",
            "rf_endpoint_noise_refresh_strength",
            "rf_endpoint_noise_refresh_until",
        }
        self.assertTrue(hidden_controls.isdisjoint(required))

    def test_settings_exposes_advanced_controls_without_sampler_overrides(self):
        inputs = AnimaFlowSettings.INPUT_TYPES()["required"]

        self.assertNotIn("steps", inputs)
        self.assertNotIn("cfg", inputs)
        self.assertNotIn("flow_solver", inputs)
        self.assertNotIn("flow_schedule", inputs)
        self.assertNotIn("flow_shift", inputs)
        self.assertNotIn("cfg_schedule_mode", inputs)
        self.assertEqual(inputs["cfg_bump_start"][1]["default"], 0.0)
        self.assertEqual(inputs["cfg_bump_end"][1]["default"], 0.27)
        self.assertFalse(inputs["cfg_legacy_progress"][1]["default"])
        self.assertFalse(inputs["denoise_legacy_progress"][1]["default"])

    def test_settings_builds_optional_connection_object(self):
        settings, summary = AnimaFlowSettings().build(
            flow_er_order=2,
            flow_pc3_gamma=0.75,
            flow_pc3_tolerance=0.008,
            cfg_early_scale=1.0,
            cfg_early_ramp_end=0.0,
            cfg_peak_boost=0.60,
            cfg_bump_start=0.0,
            cfg_bump_end=0.27,
            cfg_beta_alpha=2.0,
            cfg_beta_beta=3.0,
            late_cfg_scale=1.0,
            late_cfg_start=0.76,
            cfg_legacy_progress=False,
            denoise_legacy_progress=False,
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
            rf_endpoint_noise_refresh_enabled=False,
            rf_endpoint_noise_refresh_strength=0.15,
            rf_endpoint_noise_refresh_until=0.20,
        )

        self.assertEqual(settings["steps"], ANIMA_FLOW_BASELINE["steps"])
        self.assertEqual(settings["cfg"], ANIMA_FLOW_BASELINE["cfg"])
        self.assertEqual(settings["flow_solver"], ANIMA_FLOW_BASELINE["flow_solver"])
        self.assertEqual(settings["flow_schedule"], "flow_cosmos")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertEqual(settings["flow_pc3_gamma"], 0.75)
        self.assertEqual(settings["flow_pc3_tolerance"], 0.008)
        self.assertIn("optional advanced settings", summary)
        self.assertIn("flow_pc3_gamma: 0.7500", summary)
        self.assertIn("sampler_default_flow_shift: 5.0000", summary)
        self.assertNotIn("\nflow_shift:", summary)

    def test_none_settings_use_rc2_baseline(self):
        settings = _normalize_settings_object(None)

        self.assertEqual(settings["steps"], 35)
        self.assertEqual(settings["cfg"], 6.0)
        self.assertEqual(settings["flow_solver"], "flow_pc3_damped")
        self.assertEqual(settings["flow_schedule"], "flow_cosmos")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertFalse(settings["cfg_legacy_progress"])

    def test_connected_settings_merge_with_baseline(self):
        settings = _normalize_settings_object({"flow_pc3_tolerance": 0.01})

        self.assertEqual(settings["flow_pc3_tolerance"], 0.01)
        self.assertEqual(settings["flow_solver"], "flow_pc3_damped")
        self.assertEqual(settings["flow_schedule"], "flow_cosmos")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertEqual(settings["cfg_bump_start"], 0.0)
        self.assertEqual(settings["cfg_bump_end"], 0.27)

    def test_settings_normalization_rejects_nonfinite_flow_values(self):
        with self.assertRaisesRegex(ValueError, "flow_shift"):
            _normalize_settings_object({"flow_shift": float("nan")})
        with self.assertRaisesRegex(ValueError, "cosmos_sigma"):
            _normalize_settings_object({"cosmos_sigma_max": float("inf")})

    def test_public_cfg_modes_apply_expected_internal_schedule(self):
        base = _normalize_settings_object(None)

        bump = _apply_public_cfg_mode(base, "bump cfg")
        const = _apply_public_cfg_mode(base, "const")

        self.assertEqual(bump["cfg_schedule_mode"], "beta_bump")
        self.assertEqual(bump["cfg_peak_boost"], 0.60)
        self.assertEqual(bump["cfg_bump_start"], 0.0)
        self.assertEqual(bump["cfg_bump_end"], 0.27)
        self.assertEqual(const["cfg_schedule_mode"], "constant")
        self.assertEqual(const["cfg_peak_boost"], 0.0)
        self.assertEqual(const["cfg_early_scale"], 1.0)
        self.assertEqual(const["late_cfg_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
