import unittest
from unittest.mock import patch

import torch

from anima_sampler.nodes import (
    ANIMA_FLOW_BASELINE,
    ANIMA_FLOW_SETTINGS,
    AnimaFlowCorrectiveSampler,
    AnimaFlowSettings,
    AnimaFourWayComparison,
    NODE_CATEGORY,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    PUBLIC_CFG_MODES,
    _apply_disconnected_sampler_defaults,
    _apply_public_cfg_mode,
    _decode_latent_image,
    _normalize_settings_object,
)


class NodeRegistrationTests(unittest.TestCase):
    def test_release_nodes_are_registered(self):
        self.assertEqual(
            NODE_CLASS_MAPPINGS,
            {
                "AnimaFlowSettings": AnimaFlowSettings,
                "AnimaFlowCorrectiveSampler": AnimaFlowCorrectiveSampler,
                "AnimaFourWayComparison": AnimaFourWayComparison,
            },
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS,
            {
                "AnimaFlowSettings": "Anima Flow Settings",
                "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
                "AnimaFourWayComparison": "Anima Four Way Comparison",
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
            "AnimaFlowDiagnosticSampler",
            "AnimaSaveText",
            "AnimaCFGComparisonTest",
            "AnimaBestVsErSdeSimpleComparison",
        }

        self.assertTrue(removed_nodes.isdisjoint(NODE_CLASS_MAPPINGS))
        self.assertTrue(removed_nodes.isdisjoint(NODE_DISPLAY_NAME_MAPPINGS))

    def test_release_nodes_use_single_menu_category(self):
        self.assertEqual(NODE_CATEGORY, "anima sampler")
        for node_class in NODE_CLASS_MAPPINGS.values():
            self.assertEqual(node_class.CATEGORY, "anima sampler")

    def test_sampler_exposes_daily_controls_and_optional_settings(self):
        input_types = AnimaFlowCorrectiveSampler.INPUT_TYPES()
        required = input_types["required"]
        optional = input_types["optional"]

        self.assertEqual(required["steps"][1]["default"], 35)
        self.assertEqual(required["cfg"][1]["default"], 7.0)
        self.assertEqual(required["cfg_mode"][0], PUBLIC_CFG_MODES)
        self.assertEqual(required["cfg_mode"][1]["default"], "const")
        self.assertEqual(required["flow_solver"][1]["default"], "flow_unipc2_x0")
        self.assertEqual(required["flow_schedule"][1]["default"], "flow_rf_linear_shift")
        self.assertIn("flow_rf_linear_s_tail_shift5", required["flow_schedule"][0])
        self.assertEqual(required["flow_shift"][1]["default"], 5.0)
        self.assertIn("seed", required)
        self.assertIn("denoise", required)
        self.assertIn("add_noise", required)
        self.assertNotIn("disable_pbar", required)
        self.assertEqual(optional["flow_settings"][0], ANIMA_FLOW_SETTINGS)
        self.assertEqual(optional["vae"][0], "VAE")
        self.assertNotIn("flow_settings", required)
        self.assertNotIn("vae", required)
        self.assertEqual(AnimaFlowCorrectiveSampler.RETURN_TYPES, ("LATENT", "IMAGE", "STRING"))
        self.assertEqual(AnimaFlowCorrectiveSampler.RETURN_NAMES, ("latent", "image", "log"))

        hidden_controls = {
            "flow_er_order",
            "flow_pc3_gamma",
            "flow_pc3_tolerance",
            "flow_unipc_order",
            "flow_unipc_solver_type",
            "flow_unipc_lower_order_final",
            "flow_unipc_disable_corrector_first",
            "flow_unipc_thresholding",
            "flow_unipc_dynamic_thresholding_ratio",
            "flow_unipc_sample_max_value",
            "cosmos_sigma_max",
            "cosmos_sigma_min",
            "flow_rho7_tail_auto",
            "final_clean_pass",
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
        self.assertEqual(inputs["flow_unipc_order"][1]["default"], 2)
        self.assertEqual(inputs["flow_unipc_solver_type"][1]["default"], "bh2")
        self.assertTrue(inputs["flow_unipc_lower_order_final"][1]["default"])
        self.assertFalse(inputs["flow_unipc_thresholding"][1]["default"])
        self.assertEqual(inputs["cfg_bump_start"][1]["default"], 0.0)
        self.assertEqual(inputs["cfg_bump_end"][1]["default"], 0.27)
        self.assertFalse(inputs["cfg_legacy_progress"][1]["default"])
        self.assertFalse(inputs["denoise_legacy_progress"][1]["default"])
        self.assertFalse(inputs["flow_rho7_tail_auto"][1]["default"])
        self.assertFalse(inputs["final_clean_pass"][1]["default"])

    def test_settings_builds_optional_connection_object(self):
        settings, summary = AnimaFlowSettings().build(
            flow_er_order=2,
            flow_pc3_gamma=0.75,
            flow_pc3_tolerance=0.008,
            flow_unipc_order=2,
            flow_unipc_solver_type="bh2",
            flow_unipc_lower_order_final=True,
            flow_unipc_disable_corrector_first=0,
            flow_unipc_thresholding=False,
            flow_unipc_dynamic_thresholding_ratio=0.995,
            flow_unipc_sample_max_value=1.0,
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
            flow_rho7_tail_auto=True,
            final_clean_pass=True,
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
            rf_endpoint_noise_refresh_enabled=False,
            rf_endpoint_noise_refresh_strength=0.15,
            rf_endpoint_noise_refresh_until=0.20,
        )

        self.assertEqual(settings["steps"], ANIMA_FLOW_BASELINE["steps"])
        self.assertEqual(settings["cfg"], ANIMA_FLOW_BASELINE["cfg"])
        self.assertEqual(settings["flow_solver"], ANIMA_FLOW_BASELINE["flow_solver"])
        self.assertEqual(settings["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertTrue(settings["flow_rho7_tail_auto"])
        self.assertTrue(settings["final_clean_pass"])
        self.assertEqual(settings["flow_pc3_gamma"], 0.75)
        self.assertEqual(settings["flow_pc3_tolerance"], 0.008)
        self.assertEqual(settings["flow_unipc_order"], 2)
        self.assertEqual(settings["flow_unipc_solver_type"], "bh2")
        self.assertTrue(settings["flow_unipc_lower_order_final"])
        self.assertIn("optional advanced settings", summary)
        self.assertIn("flow_pc3_gamma: 0.7500", summary)
        self.assertIn("flow_unipc_solver_type: bh2", summary)
        self.assertIn("flow_rho7_tail_auto: True", summary)
        self.assertIn("final_clean_pass: True", summary)
        self.assertIn("sampler_default_flow_shift: 5.0000", summary)
        self.assertNotIn("\nflow_shift:", summary)

    def test_none_settings_use_rc2_baseline(self):
        settings = _normalize_settings_object(None)

        self.assertEqual(settings["steps"], 35)
        self.assertEqual(settings["cfg"], 7.0)
        self.assertEqual(settings["flow_solver"], "flow_unipc2_x0")
        self.assertEqual(settings["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertFalse(settings["flow_rho7_tail_auto"])
        self.assertFalse(settings["final_clean_pass"])
        self.assertFalse(settings["cfg_legacy_progress"])

    def test_connected_settings_merge_with_baseline(self):
        settings = _normalize_settings_object({"flow_pc3_tolerance": 0.01})

        self.assertEqual(settings["flow_pc3_tolerance"], 0.01)
        self.assertEqual(settings["flow_solver"], "flow_unipc2_x0")
        self.assertEqual(settings["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(settings["flow_shift"], 5.0)
        self.assertFalse(settings["flow_rho7_tail_auto"])
        self.assertFalse(settings["final_clean_pass"])
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
        ramp = _apply_public_cfg_mode(base, "ramp cfg")

        self.assertEqual(bump["cfg_schedule_mode"], "beta_bump")
        self.assertEqual(bump["cfg_peak_boost"], 0.60)
        self.assertEqual(bump["cfg_bump_start"], 0.0)
        self.assertEqual(bump["cfg_bump_end"], 0.27)
        self.assertEqual(ramp["cfg_schedule_mode"], "low_to_high")
        self.assertAlmostEqual(ramp["cfg_early_scale"], 4.5 / 7.0)
        self.assertEqual(ramp["cfg_interval_start"], 0.24)
        self.assertEqual(ramp["cfg_interval_rise_end"], 0.66)
        self.assertEqual(ramp["cfg_interval_fall_start"], 1.0)
        self.assertEqual(ramp["cfg_interval_end"], 1.0)
        self.assertEqual(const["cfg_schedule_mode"], "constant")
        self.assertEqual(const["cfg_peak_boost"], 0.0)
        self.assertEqual(const["cfg_early_scale"], 1.0)
        self.assertEqual(const["late_cfg_scale"], 1.0)

    def test_disconnected_linear_shift_defaults_to_cosmos25_no_final_clean(self):
        params = _normalize_settings_object(None)
        params["flow_schedule"] = "flow_rf_linear_shift"

        out = _apply_disconnected_sampler_defaults(params, None)

        self.assertFalse(out["final_clean_pass"])

    def test_connected_settings_can_force_linear_shift_final_clean(self):
        params = _normalize_settings_object(None)
        params["flow_schedule"] = "flow_rf_linear_shift"
        params["final_clean_pass"] = True

        out = _apply_disconnected_sampler_defaults(params, params)

        self.assertTrue(out["final_clean_pass"])

    def test_disconnected_linear_s_tail_shift5_defaults_to_no_final_clean(self):
        params = _normalize_settings_object(None)
        params["flow_schedule"] = "flow_rf_linear_s_tail_shift5"

        out = _apply_disconnected_sampler_defaults(params, None)

        self.assertFalse(out["final_clean_pass"])

    def test_decode_latent_image_is_optional_and_decodes_4d_latent(self):
        samples = torch.zeros(1, 16, 8, 8)
        latent = {"samples": samples}
        vae = _DummyVAE()

        self.assertIsNone(_decode_latent_image(None, latent))
        image = _decode_latent_image(vae, latent)

        self.assertIs(image, vae.image)
        self.assertIs(vae.samples, samples)

    def test_decode_latent_image_squeezes_single_frame_latent_for_2d_vae(self):
        samples = torch.zeros(1, 16, 1, 8, 8)
        vae = _DummyVAE()

        _decode_latent_image(vae, {"samples": samples})

        self.assertEqual(tuple(vae.samples.shape), (1, 16, 8, 8))

    def test_decode_latent_image_expands_4d_latent_for_3d_vae(self):
        samples = torch.zeros(1, 16, 8, 8)
        vae = _DummyVAE(latent_dim=3, image=torch.zeros(1, 1, 8, 8, 3))

        image = _decode_latent_image(vae, {"samples": samples})

        self.assertEqual(tuple(vae.samples.shape), (1, 16, 1, 8, 8))
        self.assertEqual(tuple(image.shape), (1, 8, 8, 3))

    def test_decode_latent_image_rejects_multi_frame_latent(self):
        with self.assertRaisesRegex(ValueError, "single-frame"):
            _decode_latent_image(_DummyVAE(), {"samples": torch.zeros(1, 16, 2, 8, 8)})

    def test_decode_latent_image_rejects_multi_frame_image(self):
        vae = _DummyVAE(latent_dim=3, image=torch.zeros(1, 2, 8, 8, 3))

        with self.assertRaisesRegex(ValueError, "single-frame image"):
            _decode_latent_image(vae, {"samples": torch.zeros(1, 16, 8, 8)})

    def test_sampler_returns_latent_image_and_log_when_vae_is_connected(self):
        latent = {"samples": torch.zeros(1, 16, 8, 8)}
        vae = _DummyVAE()

        with patch("anima_sampler.nodes._run_sampler_with_params", return_value=(latent, "log")):
            latent_out, image, log = AnimaFlowCorrectiveSampler().sample(
                model=object(),
                positive=[],
                negative=[],
                latent_image=latent,
                seed=1,
                steps=35,
                cfg=6.0,
                cfg_mode="bump cfg",
                flow_solver="flow_pc3_damped",
                flow_schedule="flow_cosmos",
                flow_shift=5.0,
                denoise=1.0,
                add_noise=True,
                vae=vae,
            )

        self.assertIs(latent_out, latent)
        self.assertIs(image, vae.image)
        self.assertIn("image_output: decoded", log)

    def test_sampler_uses_cosmos25_linear_shift_default_without_settings(self):
        latent = {"samples": torch.zeros(1, 16, 8, 8)}

        with patch("anima_sampler.nodes._run_sampler_with_params", return_value=(latent, "log")) as run:
            AnimaFlowCorrectiveSampler().sample(
                model=object(),
                positive=[],
                negative=[],
                latent_image=latent,
                seed=1,
                steps=35,
                cfg=6.0,
                cfg_mode="const",
                flow_solver="flow_unipc2_x0",
                flow_schedule="flow_rf_linear_shift",
                flow_shift=5.0,
                denoise=1.0,
                add_noise=True,
            )

        params = run.call_args.kwargs["params"]
        self.assertEqual(params["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(params["flow_solver"], "flow_unipc2_x0")
        self.assertEqual(params["cfg_schedule_mode"], "constant")
        self.assertFalse(params["final_clean_pass"])

    def test_four_way_comparison_node_exposes_expected_interface(self):
        inputs = AnimaFourWayComparison.INPUT_TYPES()["required"]

        self.assertEqual(inputs["steps"][1]["default"], 35)
        self.assertEqual(inputs["unipc_cfg"][1]["default"], 7.0)
        self.assertEqual(inputs["pc3_cfg"][1]["default"], 7.0)
        self.assertEqual(inputs["er_official_cfg"][1]["default"], 4.5)
        self.assertEqual(inputs["er_high_cfg"][1]["default"], 7.0)
        self.assertEqual(
            AnimaFourWayComparison.RETURN_TYPES,
            ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING"),
        )
        self.assertEqual(
            AnimaFourWayComparison.RETURN_NAMES,
            (
                "comparison",
                "unipc_image",
                "pc3_image",
                "er_sde_simple_cfg45_image",
                "er_sde_simple_cfg7_image",
                "log",
            ),
        )

    def test_four_way_comparison_uses_expected_profiles(self):
        unipc_latent = {"samples": torch.zeros(1, 16, 8, 8)}
        pc3_latent = {"samples": torch.ones(1, 16, 8, 8)}
        er45_latent = {"samples": torch.full((1, 16, 8, 8), 2.0)}
        er7_latent = {"samples": torch.full((1, 16, 8, 8), 3.0)}
        comparison = torch.full((1, 16, 16, 3), 0.5)
        vae = _DummyVAE(latent_dim=3)

        with (
            patch(
                "anima_sampler.nodes._run_sampler_with_params",
                side_effect=[(unipc_latent, "unipc log"), (pc3_latent, "pc3 log")],
            ) as flow_run,
            patch(
                "anima_sampler.nodes.run_comfy_native_sampler",
                side_effect=[(er45_latent, "er45 log"), (er7_latent, "er7 log")],
            ) as native_run,
            patch(
                "anima_sampler.nodes.build_labeled_comparison_grid",
                return_value=comparison,
            ) as grid,
        ):
            out = AnimaFourWayComparison().compare(
                model=object(),
                positive=[],
                negative=[],
                latent_image={"samples": torch.zeros(1, 16, 8, 8)},
                vae=vae,
                seed=7,
                steps=35,
                unipc_cfg=7.0,
                pc3_cfg=7.0,
                er_official_cfg=4.5,
                er_high_cfg=7.0,
                denoise=1.0,
                add_noise=True,
            )

        unipc_params = flow_run.call_args_list[0].kwargs["params"]
        pc3_params = flow_run.call_args_list[1].kwargs["params"]
        self.assertEqual(unipc_params["flow_solver"], "flow_unipc2_x0")
        self.assertEqual(pc3_params["flow_solver"], "flow_pc3_damped")
        self.assertEqual(unipc_params["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(pc3_params["flow_schedule"], "flow_rf_linear_shift")
        self.assertEqual(unipc_params["flow_shift"], 5.0)
        self.assertEqual(pc3_params["flow_shift"], 5.0)
        self.assertFalse(unipc_params["final_clean_pass"])
        self.assertFalse(pc3_params["final_clean_pass"])
        self.assertEqual(unipc_params["cfg"], 7.0)
        self.assertEqual(pc3_params["cfg"], 7.0)
        self.assertEqual(unipc_params["cfg_schedule_mode"], "constant")
        self.assertEqual(pc3_params["cfg_schedule_mode"], "constant")
        self.assertEqual(native_run.call_args_list[0].kwargs["sampler_name"], "er_sde")
        self.assertEqual(native_run.call_args_list[0].kwargs["scheduler"], "simple")
        self.assertEqual(native_run.call_args_list[0].kwargs["cfg"], 4.5)
        self.assertEqual(native_run.call_args_list[1].kwargs["cfg"], 7.0)
        self.assertEqual(len(grid.call_args.args[0]), 4)
        self.assertIn("UniPC + linear shift5", grid.call_args.args[1][0])
        self.assertIn("PC3 + linear shift5", grid.call_args.args[1][1])
        self.assertIn("er_sde + simple cfg 4.50", grid.call_args.args[1][2])
        self.assertIs(out[0], comparison)
        self.assertIn("AnimaFourWayComparison", out[5])
        self.assertIn("profile_b: flow_pc3_damped", out[5])


class _DummyVAE:
    def __init__(self, latent_dim=2, image=None):
        self.latent_dim = latent_dim
        self.samples = None
        self.image = image if image is not None else torch.zeros(1, 8, 8, 3)

    def decode(self, samples):
        self.samples = samples
        return self.image


if __name__ == "__main__":
    unittest.main()
