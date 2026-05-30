import sys
import types
import unittest
from unittest.mock import patch

import torch

from anima_sampler.nodes import (
    ANIMA_FLOW_BASELINE,
    ANIMA_FLOW_SETTINGS,
    AnimaCosmosReferenceLatent,
    AnimaCosmosReferenceModelPatch,
    AnimaCosmosRepaintPrepare,
    AnimaFlowCorrectiveSampler,
    AnimaFlowSettings,
    AnimaInpaintLatentPrepare,
    AnimaRepaintComposite,
    AnimaTReferenceControlRepaintRoute,
    AnimaTReferenceEditRoute,
    AnimaTReferenceRepaintRoute,
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
                "AnimaInpaintLatentPrepare": AnimaInpaintLatentPrepare,
                "AnimaTReferenceEditRoute": AnimaTReferenceEditRoute,
                "AnimaTReferenceRepaintRoute": AnimaTReferenceRepaintRoute,
                "AnimaTReferenceControlRepaintRoute": AnimaTReferenceControlRepaintRoute,
                "AnimaCosmosRepaintPrepare": AnimaCosmosRepaintPrepare,
                "AnimaRepaintComposite": AnimaRepaintComposite,
                "AnimaCosmosReferenceModelPatch": AnimaCosmosReferenceModelPatch,
                "AnimaCosmosReferenceLatent": AnimaCosmosReferenceLatent,
            },
        )
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS,
            {
                "AnimaFlowSettings": "Anima Flow Settings",
                "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
                "AnimaInpaintLatentPrepare": "Anima Inpaint Latent Prepare",
                "AnimaTReferenceEditRoute": "Anima T-Reference Edit Route",
                "AnimaTReferenceRepaintRoute": "Anima T-Reference Repaint Route",
                "AnimaTReferenceControlRepaintRoute": "Anima T-Reference Control Repaint Route",
                "AnimaCosmosRepaintPrepare": "Anima Cosmos Repaint Prepare",
                "AnimaRepaintComposite": "Anima Repaint Composite",
                "AnimaCosmosReferenceModelPatch": "Anima Cosmos Reference Model Patch",
                "AnimaCosmosReferenceLatent": "Anima Cosmos Reference Latent",
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

    def test_decode_latent_image_composites_repaint_output_over_source(self):
        samples = torch.zeros(1, 16, 4, 4)
        source = torch.zeros(1, 4, 4, 3)
        mask = torch.zeros(1, 4, 4)
        mask[:, 1:3, 1:3] = 1.0
        vae = _DummyVAE(image=torch.ones(1, 4, 4, 3))

        image = _decode_latent_image(
            vae,
            {
                "samples": samples,
                "anima_repaint_source_image": source,
                "anima_repaint_mask": mask,
            },
        )

        self.assertEqual(float(image[:, 0, 0].max()), 0.0)
        self.assertEqual(float(image[:, 1:3, 1:3].min()), 1.0)

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

    def test_repaint_prepare_outputs_latent_mask_and_control_image(self):
        image = torch.ones(1, 8, 8, 3)
        mask = torch.zeros(1, 8, 8)
        mask[:, 3:5, 3:5] = 1.0
        vae = _DummyEncodeVAE(samples=torch.zeros(1, 16, 2, 2))

        latent, control, out_mask, preview, log = AnimaCosmosRepaintPrepare().prepare(
            image=image,
            mask=mask,
            vae=vae,
            mode="structure repaint",
            mask_threshold=0.5,
            mask_grow=1,
            mask_feather=1,
            latent_fill="original",
            noise_seed=1,
            control_fill="masked black",
            invert_mask=False,
        )

        self.assertIs(latent["samples"], vae.samples)
        self.assertEqual(tuple(latent["noise_mask"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(control.shape), (1, 8, 8, 3))
        self.assertEqual(tuple(out_mask.shape), (1, 8, 8))
        self.assertEqual(tuple(preview.shape), (1, 8, 8, 3))
        self.assertLess(float(control[:, 3:5, 3:5].max()), 1.0)
        self.assertIn("recommended_denoise: 0.90", log)
        self.assertIn("noise_mask_shape: [1, 1, 2, 2]", log)

    def test_repaint_feather_preserves_mask_core_without_coloring_control_edge(self):
        image = torch.ones(1, 7, 7, 3)
        mask = torch.zeros(1, 7, 7)
        mask[:, 3, 3] = 1.0
        vae = _DummyEncodeVAE(samples=torch.zeros(1, 16, 2, 2))

        latent, control, out_mask, _preview, _log = AnimaCosmosRepaintPrepare().prepare(
            image=image,
            mask=mask,
            vae=vae,
            mode="edge repair",
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=2,
            latent_fill="original",
            noise_seed=1,
            control_fill="masked black",
            invert_mask=False,
        )

        self.assertEqual(float(out_mask[0, 3, 3]), 1.0)
        self.assertEqual(float(out_mask[0, 3, 4]), 0.0)
        self.assertGreater(float(latent["anima_repaint_mask"][0, 3, 4]), 0.0)
        self.assertEqual(float(control[0, 3, 4, 0]), 1.0)

    def test_inpaint_latent_prepare_exposes_image_mask_workflow(self):
        inputs = AnimaInpaintLatentPrepare.INPUT_TYPES()["required"]

        self.assertIn("image", inputs)
        self.assertIn("mask", inputs)
        self.assertIn("vae", inputs)
        self.assertNotIn("model", inputs)
        self.assertNotIn("positive", inputs)
        self.assertNotIn("negative", inputs)
        self.assertEqual(inputs["latent_fill"][1]["default"], "neutral gray")
        self.assertIn("noise_seed", inputs)
        self.assertEqual(
            AnimaInpaintLatentPrepare.RETURN_TYPES,
            ("LATENT", "MASK", "IMAGE", "STRING"),
        )

    def test_repaint_prepare_uses_hard_sample_mask_and_soft_composite_mask(self):
        image = torch.ones(1, 7, 7, 3)
        mask = torch.zeros(1, 7, 7)
        mask[:, 3, 3] = 1.0
        vae = _DummyEncodeVAE(samples=torch.zeros(1, 16, 7, 7))

        latent, _control, out_mask, _preview, log = AnimaCosmosRepaintPrepare().prepare(
            image=image,
            mask=mask,
            vae=vae,
            mode="edge repair",
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=2,
            latent_fill="neutral gray",
            noise_seed=1,
            control_fill="neutral gray",
            invert_mask=False,
        )

        self.assertEqual(float(latent["noise_mask"][0, 0, 3, 3]), 1.0)
        self.assertEqual(float(latent["noise_mask"][0, 0, 3, 4]), 0.0)
        self.assertEqual(float(out_mask[0, 3, 4]), 0.0)
        self.assertGreater(float(latent["anima_repaint_mask"][0, 3, 4]), 0.0)
        self.assertEqual(float(vae.encoded_image[0, 3, 4, 0]), 1.0)
        self.assertAlmostEqual(float(vae.encoded_image[0, 3, 3, 0]), 0.5)
        self.assertIn("hard grown mask", log)
        self.assertIn("output_mask: hard grown mask", log)

    def test_inpaint_latent_prepare_outputs_sampler_ready_latent(self):
        image = torch.ones(1, 8, 8, 3)
        mask = torch.zeros(1, 8, 8)
        mask[:, 2:6, 2:6] = 1.0
        samples = torch.zeros(1, 16, 2, 2)
        vae = _DummyEncodeVAE(samples=samples)

        latent, out_mask, preview, log = (
            AnimaInpaintLatentPrepare().prepare(
                image=image,
                mask=mask,
                vae=vae,
                mask_threshold=0.5,
                mask_grow=0,
                mask_feather=0,
                latent_fill="latent noise",
                noise_seed=123,
                invert_mask=False,
            )
        )

        self.assertFalse(torch.equal(latent["samples"], samples))
        self.assertIn("noise_mask", latent)
        self.assertEqual(tuple(latent["noise_mask"].shape), (1, 1, 2, 2))
        self.assertEqual(tuple(out_mask.shape), (1, 8, 8))
        self.assertEqual(tuple(preview.shape), (1, 8, 8, 3))
        self.assertTrue(torch.equal(vae.encoded_image, image))
        self.assertIn("next_node: connect latent to Anima Flow Corrective Sampler", log)
        self.assertIn("controlnet: disabled", log)

    def test_inpaint_latent_noise_is_seeded_and_mask_limited(self):
        image = torch.ones(1, 2, 2, 3)
        mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        samples = torch.zeros(1, 16, 2, 2)

        latent_a, _out_mask, _preview, _log = AnimaInpaintLatentPrepare().prepare(
            image=image,
            mask=mask,
            vae=_DummyEncodeVAE(samples=samples),
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=0,
            latent_fill="latent noise",
            noise_seed=99,
            invert_mask=False,
        )
        latent_b, _out_mask, _preview, _log = AnimaInpaintLatentPrepare().prepare(
            image=image,
            mask=mask,
            vae=_DummyEncodeVAE(samples=samples),
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=0,
            latent_fill="latent noise",
            noise_seed=99,
            invert_mask=False,
        )

        self.assertTrue(torch.equal(latent_a["samples"], latent_b["samples"]))
        self.assertGreater(float(latent_a["samples"][..., 0, 0].abs().sum()), 0.0)
        self.assertEqual(float(latent_a["samples"][..., 0, 1].abs().sum()), 0.0)
        self.assertEqual(float(latent_a["samples"][..., 1, 0].abs().sum()), 0.0)
        self.assertEqual(float(latent_a["samples"][..., 1, 1].abs().sum()), 0.0)

    def test_zero_mask_threshold_selects_only_nonzero_mask_pixels(self):
        image = torch.ones(1, 2, 2, 3)
        mask = torch.tensor([[[0.0, 0.25], [0.0, 0.0]]])
        vae = _DummyEncodeVAE(samples=torch.zeros(1, 16, 2, 2))

        AnimaCosmosRepaintPrepare().prepare(
            image=image,
            mask=mask,
            vae=vae,
            mode="edge repair",
            mask_threshold=0.0,
            mask_grow=0,
            mask_feather=0,
            latent_fill="masked black",
            noise_seed=1,
            control_fill="masked black",
            invert_mask=False,
        )

        self.assertEqual(float(vae.encoded_image[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(vae.encoded_image[0, 0, 1, 0]), 0.0)
        self.assertEqual(float(vae.encoded_image[0, 1, 0, 0]), 1.0)

    def test_repaint_prepare_can_invert_mask_and_fill_latent(self):
        image = torch.ones(1, 4, 4, 3)
        mask = torch.ones(1, 4, 4)
        mask[:, 1:3, 1:3] = 0.0
        vae = _DummyEncodeVAE(samples=torch.zeros(1, 16, 1, 1))

        AnimaCosmosRepaintPrepare().prepare(
            image=image,
            mask=mask,
            vae=vae,
            mode="edge repair",
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=0,
            latent_fill="neutral gray",
            noise_seed=1,
            control_fill="neutral gray",
            invert_mask=True,
        )

        self.assertAlmostEqual(float(vae.encoded_image[:, 1:3, 1:3].mean()), 0.5)

    def test_repaint_composite_softens_native_sampler_output_edge(self):
        source = torch.ones(1, 7, 7, 3)
        repaint = torch.zeros(1, 7, 7, 3)
        mask = torch.zeros(1, 7, 7)
        mask[:, 3, 3] = 1.0

        image, composite_mask, log = AnimaRepaintComposite().composite(
            source_image=source,
            repaint_image=repaint,
            mask=mask,
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=2,
            invert_mask=False,
        )

        self.assertEqual(tuple(image.shape), (1, 7, 7, 3))
        self.assertEqual(tuple(composite_mask.shape), (1, 7, 7))
        self.assertEqual(float(image[0, 3, 3, 0]), 0.0)
        self.assertGreater(float(image[0, 3, 4, 0]), 0.0)
        self.assertLess(float(image[0, 3, 4, 0]), 1.0)
        self.assertEqual(float(image[0, 0, 0, 0]), 1.0)
        self.assertIn("native KSampler/VAEDecode repaint", log)

    def test_t_reference_repaint_route_builds_patched_model_and_latent(self):
        image = torch.full((1, 8, 8, 3), 0.25)
        image[:, 2:6, 2:6] = 1.0
        mask = torch.zeros(1, 8, 8)
        mask[:, 2:6, 2:6] = 1.0
        reference_samples = torch.full((1, 16, 2, 2), 2.0)
        latent_samples = torch.zeros(1, 16, 2, 2)
        vae = _SequenceEncodeVAE([latent_samples, reference_samples])
        model = _DummyModelPatcher()

        patched, latent, out_mask, preview, log = AnimaTReferenceRepaintRoute().build(
            model=model,
            image=image,
            mask=mask,
            vae=vae,
            mode="structure repaint",
            mask_threshold=0.5,
            mask_grow=0,
            mask_feather=0,
            latent_fill="original",
            reference_fill="neutral gray",
            noise_seed=1,
            invert_mask=False,
        )

        self.assertIsNot(patched, model)
        self.assertIs(latent["samples"], latent_samples)
        self.assertEqual(len(patched.model_options["anima_ref_latents"]), 1)
        self.assertIs(patched.model_options["anima_ref_latents"][0], reference_samples)
        self.assertEqual(tuple(out_mask.shape), (1, 8, 8))
        self.assertEqual(tuple(preview.shape), (1, 8, 8, 3))
        reference_pixels = vae.encoded_images[1]
        self.assertAlmostEqual(float(reference_pixels[:, 2:6, 2:6].mean()), 0.5)
        self.assertAlmostEqual(float(reference_pixels[:, :2, :, :].mean()), 0.25)
        self.assertIn("no-controlnet t-reference repaint", log)
        self.assertIn("reference_masking", log)
        self.assertIn("reference_fill: neutral gray", log)
        self.assertIn("connect_model: route.model", log)

    def test_t_reference_repaint_route_defaults_avoid_source_and_black_leakage(self):
        inputs = AnimaTReferenceRepaintRoute.INPUT_TYPES()["required"]

        self.assertEqual(inputs["latent_fill"][1]["default"], "neutral gray")
        self.assertEqual(inputs["reference_fill"][1]["default"], "neutral gray")

    def test_t_reference_edit_route_matches_discussion_workflow_shape(self):
        image = torch.ones(1, 8, 8, 3)
        samples = torch.full((1, 16, 2, 2), 2.0)
        vae = _DummyEncodeVAE(samples=samples)
        model = _DummyModelPatcher()

        patched, latent, reference_latent, log = AnimaTReferenceEditRoute().build(
            model=model,
            image=image,
            vae=vae,
        )

        self.assertIsNot(patched, model)
        self.assertIs(latent["samples"], samples)
        self.assertIs(reference_latent["samples"], samples)
        self.assertEqual(len(patched.model_options["anima_ref_latents"]), 1)
        self.assertIs(patched.model_options["anima_ref_latents"][0], samples)
        self.assertNotIn("noise_mask", latent)
        self.assertIn("full-image t-reference edit", log)
        self.assertIn("AnimaEdit LoRA", log)
        self.assertIn("native KSampler er_sde/simple", log)

    def test_t_reference_control_route_outputs_reference_control_and_latent(self):
        image = torch.full((1, 8, 8, 3), 0.25)
        image[:, 2:6, 2:6] = 1.0
        mask = torch.zeros(1, 8, 8)
        mask[:, 2:6, 2:6] = 1.0
        reference_samples = torch.full((1, 16, 2, 2), 2.0)
        latent_samples = torch.zeros(1, 16, 2, 2)
        vae = _SequenceEncodeVAE([latent_samples, reference_samples])

        reference_latent, latent, control, out_mask, preview, log = (
            AnimaTReferenceControlRepaintRoute().prepare(
                image=image,
                mask=mask,
                vae=vae,
                mode="structure repaint",
                mask_threshold=0.5,
                mask_grow=0,
                mask_feather=0,
                latent_fill="original",
                noise_seed=1,
                control_fill="masked black",
                reference_fill="neutral gray",
                invert_mask=False,
            )
        )

        self.assertIs(reference_latent["samples"], reference_samples)
        self.assertIs(latent["samples"], latent_samples)
        self.assertEqual(tuple(control.shape), (1, 8, 8, 3))
        self.assertEqual(tuple(out_mask.shape), (1, 8, 8))
        self.assertEqual(tuple(preview.shape), (1, 8, 8, 3))
        self.assertLess(float(control[:, 2:6, 2:6].max()), 1.0)
        self.assertEqual(float(control[:, 2:6, 2:6].max()), 0.0)
        reference_pixels = vae.encoded_images[1]
        self.assertAlmostEqual(float(reference_pixels[:, 2:6, 2:6].mean()), 0.5)
        self.assertIn("external ControlNet/LLLite", log)
        self.assertIn("reference_masking", log)
        self.assertIn("reference_fill: neutral gray", log)
        self.assertIn("connect_reference: reference_latent", log)

    def test_t_reference_control_route_uses_hard_mask_for_reference_fill(self):
        image = torch.ones(1, 7, 7, 3)
        mask = torch.zeros(1, 7, 7)
        mask[:, 3, 3] = 1.0
        reference_samples = torch.full((1, 16, 2, 2), 2.0)
        latent_samples = torch.zeros(1, 16, 2, 2)
        vae = _SequenceEncodeVAE([latent_samples, reference_samples])

        reference_latent, latent, control, out_mask, _preview, _log = (
            AnimaTReferenceControlRepaintRoute().prepare(
                image=image,
                mask=mask,
                vae=vae,
                mode="edge repair",
                mask_threshold=0.5,
                mask_grow=0,
                mask_feather=2,
                latent_fill="original",
                noise_seed=1,
                control_fill="masked black",
                reference_fill="neutral gray",
                invert_mask=False,
            )
        )

        self.assertIs(reference_latent["samples"], reference_samples)
        self.assertIs(latent["samples"], latent_samples)
        self.assertEqual(float(out_mask[0, 3, 3]), 1.0)
        self.assertEqual(float(out_mask[0, 3, 4]), 0.0)
        self.assertGreater(float(latent["anima_repaint_mask"][0, 3, 4]), 0.0)
        self.assertEqual(float(control[0, 3, 3, 0]), 0.0)
        self.assertEqual(float(control[0, 3, 4, 0]), 1.0)

        reference_pixels = vae.encoded_images[1]
        self.assertAlmostEqual(float(reference_pixels[0, 3, 3, 0]), 0.5)
        self.assertEqual(float(reference_pixels[0, 3, 4, 0]), 1.0)

    def test_control_route_has_no_model_input_to_avoid_controlnet_cycles(self):
        inputs = AnimaTReferenceControlRepaintRoute.INPUT_TYPES()["required"]

        self.assertNotIn("model", inputs)
        self.assertIn("control_fill", inputs)
        self.assertEqual(
            AnimaTReferenceControlRepaintRoute.RETURN_NAMES,
            (
                "reference_latent",
                "latent",
                "control_image",
                "mask",
                "mask_preview",
                "log",
            ),
        )

    def test_reference_latent_patches_model_and_appends_time_frames(self):
        model = _DummyModelPatcher()
        reference = torch.full((1, 16, 1, 2, 2), 2.0)

        patched, = AnimaCosmosReferenceLatent().apply(
            model=model,
            latent={"samples": reference},
            enabled=True,
        )

        self.assertIsNot(patched, model)
        self.assertTrue(patched.model_options["anima_cosmos_reference_patch_installed"])
        self.assertEqual(len(patched.model_options["anima_ref_latents"]), 1)

        wrapper = patched.model_options["model_function_wrapper"]
        x = torch.ones(1, 16, 1, 2, 2)
        out = wrapper(
            patched.model.apply_model,
            {"input": x, "timestep": torch.ones(1), "c": {}},
        )

        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(patched.model.last_input_shape, (1, 16, 2, 2, 2))

    def test_reference_latent_disabled_is_pass_through(self):
        model = _DummyModelPatcher()

        out, = AnimaCosmosReferenceLatent().apply(
            model=model,
            latent={"samples": torch.zeros(1, 16, 1, 2, 2)},
            enabled=False,
        )

        self.assertIs(out, model)
        self.assertNotIn("model_function_wrapper", model.model_options)

    def test_reference_patch_accepts_comfy_cond_list_references(self):
        class FakeCONDList:
            def __init__(self, items):
                self.cond = list(items)

        comfy_module = types.ModuleType("comfy")
        conds_module = types.ModuleType("comfy.conds")
        conds_module.CONDList = FakeCONDList
        comfy_module.conds = conds_module
        previous_comfy = sys.modules.get("comfy")
        previous_conds = sys.modules.get("comfy.conds")
        sys.modules["comfy"] = comfy_module
        sys.modules["comfy.conds"] = conds_module
        try:
            model = _DummyModelPatcher()
            patched, = AnimaCosmosReferenceModelPatch().patch(model)
            wrapper = patched.model_options["model_function_wrapper"]
            x = torch.ones(1, 16, 1, 2, 2)
            reference = torch.full((1, 16, 1, 2, 2), 2.0)

            wrapper(
                patched.model.apply_model,
                {
                    "input": x,
                    "timestep": torch.ones(1),
                    "c": {"ref_latents": FakeCONDList([reference])},
                },
            )
        finally:
            if previous_comfy is None:
                sys.modules.pop("comfy", None)
            else:
                sys.modules["comfy"] = previous_comfy
            if previous_conds is None:
                sys.modules.pop("comfy.conds", None)
            else:
                sys.modules["comfy.conds"] = previous_conds

        self.assertEqual(patched.model.last_input_shape, (1, 16, 2, 2, 2))

    def test_reference_patch_accepts_cond_object_with_list_payload(self):
        class CondObject:
            def __init__(self, items):
                self.cond = list(items)

        model = _DummyModelPatcher()
        patched, = AnimaCosmosReferenceModelPatch().patch(model)
        wrapper = patched.model_options["model_function_wrapper"]
        x = torch.ones(1, 16, 1, 2, 2)
        reference = torch.full((1, 16, 1, 2, 2), 2.0)

        wrapper(
            patched.model.apply_model,
            {
                "input": x,
                "timestep": torch.ones(1),
                "c": {"ref_latents": CondObject([reference])},
            },
        )

        self.assertEqual(patched.model.last_input_shape, (1, 16, 2, 2, 2))

    def test_reference_patch_does_not_rewrap_wrapper_chain_that_preserves_reference(self):
        model = _DummyModelPatcher()
        ref_a = torch.full((1, 16, 1, 2, 2), 2.0)
        ref_b = torch.full((1, 16, 1, 2, 2), 3.0)
        patched, = AnimaCosmosReferenceLatent().apply(model, {"samples": ref_a}, True)
        previous_wrapper = patched.model_options["model_function_wrapper"]

        def external_wrapper(model_apply, model_kwargs):
            return previous_wrapper(model_apply, model_kwargs)

        patched.model_options["model_function_wrapper"] = external_wrapper
        patched, = AnimaCosmosReferenceLatent().apply(patched, {"samples": ref_b}, True)
        x = torch.ones(1, 16, 1, 2, 2)

        patched.model_options["model_function_wrapper"](
            patched.model.apply_model,
            {"input": x, "timestep": torch.ones(1), "c": {}},
        )

        self.assertIs(patched.model_options["model_function_wrapper"], external_wrapper)
        self.assertEqual(patched.model.last_input_shape, (1, 16, 3, 2, 2))

    def test_reference_patch_rewraps_when_installed_flag_is_stale(self):
        model = _DummyModelPatcher()
        ref_a = torch.full((1, 16, 1, 2, 2), 2.0)
        ref_b = torch.full((1, 16, 1, 2, 2), 3.0)
        patched, = AnimaCosmosReferenceLatent().apply(model, {"samples": ref_a}, True)

        def replacement_wrapper(model_apply, model_kwargs):
            kwargs = dict(model_kwargs)
            x = kwargs.pop("input")
            timestep = kwargs.pop("timestep")
            cond = kwargs.pop("c", {}) or {}
            return model_apply(x, timestep, **cond, **kwargs)

        patched.model_options["model_function_wrapper"] = replacement_wrapper
        patched, = AnimaCosmosReferenceLatent().apply(patched, {"samples": ref_b}, True)
        x = torch.ones(1, 16, 1, 2, 2)

        patched.model_options["model_function_wrapper"](
            patched.model.apply_model,
            {"input": x, "timestep": torch.ones(1), "c": {}},
        )

        self.assertIsNot(patched.model_options["model_function_wrapper"], replacement_wrapper)
        self.assertEqual(patched.model.last_input_shape, (1, 16, 3, 2, 2))

    def test_reference_patch_detects_previous_wrapper_in_default_argument(self):
        model = _DummyModelPatcher()
        ref_a = torch.full((1, 16, 1, 2, 2), 2.0)
        ref_b = torch.full((1, 16, 1, 2, 2), 3.0)
        patched, = AnimaCosmosReferenceLatent().apply(model, {"samples": ref_a}, True)
        previous_wrapper = patched.model_options["model_function_wrapper"]

        def external_wrapper(model_apply, model_kwargs, previous=previous_wrapper):
            return previous(model_apply, model_kwargs)

        patched.model_options["model_function_wrapper"] = external_wrapper
        patched, = AnimaCosmosReferenceLatent().apply(patched, {"samples": ref_b}, True)
        x = torch.ones(1, 16, 1, 2, 2)

        patched.model_options["model_function_wrapper"](
            patched.model.apply_model,
            {"input": x, "timestep": torch.ones(1), "c": {}},
        )

        self.assertIs(patched.model_options["model_function_wrapper"], external_wrapper)
        self.assertEqual(patched.model.last_input_shape, (1, 16, 3, 2, 2))


class _DummyVAE:
    def __init__(self, latent_dim=2, image=None):
        self.latent_dim = latent_dim
        self.samples = None
        self.image = image if image is not None else torch.zeros(1, 8, 8, 3)

    def decode(self, samples):
        self.samples = samples
        return self.image


class _DummyEncodeVAE:
    def __init__(self, samples):
        self.samples = samples
        self.encoded_image = None

    def encode(self, image):
        self.encoded_image = image
        return self.samples


class _SequenceEncodeVAE:
    def __init__(self, samples):
        self.samples = list(samples)
        self.encoded_images = []

    def encode(self, image):
        self.encoded_images.append(image)
        if not self.samples:
            raise AssertionError("VAE encode called more times than expected")
        return self.samples.pop(0)


class _DummyInnerModel:
    def __init__(self):
        self.last_input_shape = None
        self.anima_ref_latents = []

    def extra_conds(self, **kwargs):
        return {}

    def process_latent_in(self, latent):
        return latent

    def apply_model(self, x, timestep, **kwargs):
        self.last_input_shape = tuple(x.shape)
        return x


class _DummyModelPatcher:
    def __init__(self):
        self.model = _DummyInnerModel()
        self.model_options = {}

    def clone(self):
        clone = _DummyModelPatcher()
        clone.model = self.model
        clone.model_options = dict(self.model_options)
        return clone

    def add_object_patch(self, name, value):
        setattr(self.model, name, value)

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper


if __name__ == "__main__":
    unittest.main()
