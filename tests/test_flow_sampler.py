import unittest

import torch

from anima_sampler.flow_sampler import (
    CFG_SCHEDULE_DOMAINS,
    CFG_SCHEDULE_MODES,
    AnimaSamplerLog,
    FLOW_SOLVERS,
    FlowERState,
    FlowPC3State,
    _describe_model_sampling_shift,
    _hybrid_tail_start_step,
    _infer_cosmos_latent_channels,
    _pc3_fsal_cache_score,
    _randn_like,
    _restore_sampler_channels,
    build_anima_sigmas,
    cfg_schedule_position,
    flow_er_step,
    flow_euler_step,
    flow_3m_damped_step,
    flow_pc3_damped_step,
    flow_pc3_damped_step_result,
    flow_pc3_predictor_step,
    flow_heun_step,
    rf_endpoint_noise_refresh,
    flow_velocity,
    _normalize_cosmos_latent,
    cfg_at_progress,
)


class FlowSamplerScheduleTests(unittest.TestCase):
    def test_flow_er_is_available_as_solver(self):
        self.assertIn("flow_er", FLOW_SOLVERS)
        self.assertIn("flow_heun", FLOW_SOLVERS)
        self.assertIn("flow_pc3_damped", FLOW_SOLVERS)
        self.assertIn("flow_pc3_fsal_gated", FLOW_SOLVERS)
        self.assertIn("flow_3m_damped", FLOW_SOLVERS)
        self.assertIn("flow_3m_sparse_pc3_fsal", FLOW_SOLVERS)
        self.assertNotIn("flow_rho7_euler", FLOW_SOLVERS)

    def test_sampler_log_explains_steps_and_estimated_calls(self):
        log = _dummy_sampler_log(actual_steps=12, sampler_core="flow_heun")

        self.assertEqual(log.estimated_model_calls(), 23)
        self.assertIn("steps_semantics: RF integration intervals", log.as_text())
        self.assertIn("estimated_model_calls: 23", log.as_text())

        actual_log = _dummy_sampler_log(
            actual_steps=12,
            sampler_core="flow_pc3_fsal_gated",
            actual_model_calls=43,
            cache_candidates=10,
            cache_accepts=6,
            cache_rejects=4,
            pc3_used_total=11,
        )
        self.assertIn("actual_model_calls: 43", actual_log.as_text())
        self.assertIn("cache_accept_rate: 0.6000", actual_log.as_text())

        sparse_log = _dummy_sampler_log(
            actual_steps=35,
            sampler_core="flow_3m_sparse_pc3_fsal",
        )
        self.assertEqual(sparse_log.estimated_model_calls(), 43)

        three_m_log = _dummy_sampler_log(
            actual_steps=12,
            sampler_core="flow_3m_damped",
            mean_gamma3=0.42,
        )
        self.assertIn("mean_gamma3: 0.4200", three_m_log.as_text())

    def test_lambda_cfg_schedule_domain_is_available(self):
        self.assertEqual(CFG_SCHEDULE_DOMAINS[0], "lambda")
        self.assertIn("rf_t", CFG_SCHEDULE_DOMAINS)
        self.assertIn("progress", CFG_SCHEDULE_DOMAINS)

    def test_beta_bump_cfg_schedule_mode_is_default_option(self):
        self.assertEqual(CFG_SCHEDULE_MODES[0], "beta_bump")
        self.assertIn("legacy_boost", CFG_SCHEDULE_MODES)
        self.assertIn("constant", CFG_SCHEDULE_MODES)

    def test_build_anima_sigmas_exposes_schedule_variants(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        simple = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="simple",
        )
        flow_cosmos = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="flow_cosmos",
        )
        flow_cosmos_lambda_biased_strong = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="flow_cosmos_lambda_biased_strong",
        )
        flow_cosmos_shift5 = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="flow_cosmos",
            flow_shift=5.0,
        )
        flow_cosmos_rho7_rf_tail_auto = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="flow_cosmos_rho7_rf_tail_auto",
        )
        flow_cosmos_rho7_with_shift = build_anima_sigmas(
            model,
            4,
            denoise=1.0,
            flow_schedule="flow_cosmos_rho7_rf_tail_auto",
            flow_shift=5.0,
        )

        self.assertEqual(simple.tolist(), [1.0, 0.75, 0.5, 0.25, 0.0])
        self.assertAlmostEqual(float(flow_cosmos[0]), 80.0 / 81.0, places=5)
        self.assertAlmostEqual(float(flow_cosmos[-2]), 0.002 / 1.002, places=5)
        self.assertGreater(float(flow_cosmos[1]), float(flow_cosmos[2]))
        self.assertEqual(float(flow_cosmos[-1]), 0.0)
        self.assertAlmostEqual(float(flow_cosmos_lambda_biased_strong[0]), 80.0 / 81.0, places=5)
        self.assertAlmostEqual(float(flow_cosmos_lambda_biased_strong[-2]), 0.002 / 1.002, places=5)
        self.assertEqual(float(flow_cosmos_lambda_biased_strong[-1]), 0.0)
        self.assertAlmostEqual(float(flow_cosmos_shift5[0]), 400.0 / 401.0, places=5)
        self.assertAlmostEqual(float(flow_cosmos_shift5[-2]), 0.002 / 1.002, places=5)
        self.assertGreater(float(flow_cosmos_shift5[0]), float(flow_cosmos[0]))
        self.assertEqual(float(flow_cosmos_shift5[-1]), 0.0)
        self.assertAlmostEqual(float(flow_cosmos_rho7_rf_tail_auto[0]), 80.0 / 81.0, places=5)
        self.assertAlmostEqual(float(flow_cosmos_rho7_rf_tail_auto[-2]), 0.002 / 1.002, places=5)
        self.assertEqual(float(flow_cosmos_rho7_rf_tail_auto[-1]), 0.0)
        self.assertEqual(flow_cosmos_rho7_with_shift.tolist(), flow_cosmos_rho7_rf_tail_auto.tolist())

    def test_build_anima_sigmas_rejects_unknown_schedule(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        with self.assertRaises(ValueError):
            build_anima_sigmas(
                model,
                4,
                denoise=1.0,
                flow_schedule="not_a_schedule",
            )

    def test_build_anima_sigmas_rejects_nonfinite_flow_values(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        with self.assertRaisesRegex(ValueError, "flow_shift"):
            build_anima_sigmas(
                model,
                4,
                denoise=1.0,
                flow_schedule="flow_cosmos",
                flow_shift=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "cosmos_sigma"):
            build_anima_sigmas(
                model,
                4,
                denoise=1.0,
                flow_schedule="flow_cosmos",
                cosmos_sigma_max=float("inf"),
            )

    def test_hybrid_tail_start_detects_uniform_ell_tail_for_history_reset(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))
        sigmas = build_anima_sigmas(
            model,
            35,
            denoise=1.0,
            flow_schedule="flow_cosmos_rho7_rf_tail_auto",
        )

        start = _hybrid_tail_start_step(torch, sigmas, "flow_cosmos_rho7_rf_tail_auto")

        self.assertIsNotNone(start)
        self.assertGreater(start, 0)
        self.assertIsNone(_hybrid_tail_start_step(torch, sigmas, "flow_cosmos"))

    def test_hybrid_tail_start_detects_shift_rf_tail_for_history_reset(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))
        sigmas = build_anima_sigmas(
            model,
            35,
            denoise=1.0,
            flow_schedule="flow_cosmos",
            flow_shift=5.0,
        )

        start = _hybrid_tail_start_step(torch, sigmas, "flow_cosmos", flow_shift=5.0)

        self.assertIsNotNone(start)
        self.assertGreater(start, 0)
        self.assertIsNone(_hybrid_tail_start_step(torch, sigmas, "flow_cosmos", flow_shift=1.0))

    def test_flow_cosmos_partial_denoise_uses_rf_lambda_start(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        sigmas = build_anima_sigmas(
            model,
            4,
            denoise=0.5,
            flow_schedule="flow_cosmos",
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
        )

        expected_start_sigma = (80.0 * 0.002) ** 0.5
        self.assertEqual(len(sigmas), 5)
        self.assertAlmostEqual(
            float(sigmas[0]),
            expected_start_sigma / (1.0 + expected_start_sigma),
            places=6,
        )
        self.assertAlmostEqual(float(sigmas[-2]), 0.002 / 1.002, places=6)
        self.assertEqual(float(sigmas[-1]), 0.0)

    def test_flow_cosmos_partial_denoise_applies_shifted_start_with_unshifted_tail(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        sigmas = build_anima_sigmas(
            model,
            4,
            denoise=0.5,
            flow_schedule="flow_cosmos",
            flow_shift=5.0,
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
        )

        expected_start_sigma = (80.0 * 5.0 * 0.002) ** 0.5
        self.assertEqual(len(sigmas), 5)
        self.assertAlmostEqual(
            float(sigmas[0]),
            expected_start_sigma / (1.0 + expected_start_sigma),
            places=6,
        )
        self.assertAlmostEqual(float(sigmas[-2]), 0.002 / 1.002, places=6)
        self.assertEqual(float(sigmas[-1]), 0.0)

    def test_flow_cosmos_partial_denoise_can_use_legacy_step_tail(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101))

        rf_sigmas = build_anima_sigmas(
            model,
            4,
            denoise=0.5,
            flow_schedule="flow_cosmos",
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
        )
        legacy_sigmas = build_anima_sigmas(
            model,
            4,
            denoise=0.5,
            flow_schedule="flow_cosmos",
            cosmos_sigma_max=80.0,
            cosmos_sigma_min=0.002,
            denoise_legacy_progress=True,
        )

        self.assertEqual(len(legacy_sigmas), 5)
        self.assertNotAlmostEqual(float(legacy_sigmas[0]), float(rf_sigmas[0]), places=4)

    def test_cfg_boost_decays_to_base(self):
        cfg = cfg_at_progress(
            0.0,
            base_cfg=7.0,
            early_cfg_boost=1.5,
            early_cfg_until=0.3,
            late_cfg_scale=0.9,
            late_cfg_start=0.75,
        )
        self.assertAlmostEqual(cfg, 8.5)

        cfg = cfg_at_progress(
            0.3,
            base_cfg=7.0,
            early_cfg_boost=1.5,
            early_cfg_until=0.3,
            late_cfg_scale=0.9,
            late_cfg_start=0.75,
        )
        self.assertAlmostEqual(cfg, 7.0)

    def test_late_cfg_scales_down(self):
        cfg = cfg_at_progress(
            1.0,
            base_cfg=7.0,
            early_cfg_boost=1.5,
            early_cfg_until=0.3,
            late_cfg_scale=0.8,
            late_cfg_start=0.75,
        )
        self.assertAlmostEqual(cfg, 5.6)

    def test_beta_bump_cfg_starts_mild_peaks_then_softens(self):
        cfg_start = cfg_at_progress(
            0.0,
            base_cfg=6.0,
            cfg_schedule_mode="beta_bump",
            cfg_early_scale=0.98,
            cfg_early_ramp_end=0.10,
            cfg_peak_boost=0.60,
            cfg_bump_start=0.08,
            cfg_bump_end=0.68,
            cfg_beta_alpha=4.0,
            cfg_beta_beta=7.0,
            late_cfg_scale=0.92,
            late_cfg_start=0.76,
        )
        cfg_peak = cfg_at_progress(
            0.28,
            base_cfg=6.0,
            cfg_schedule_mode="beta_bump",
            cfg_early_scale=0.98,
            cfg_early_ramp_end=0.10,
            cfg_peak_boost=0.60,
            cfg_bump_start=0.08,
            cfg_bump_end=0.68,
            cfg_beta_alpha=4.0,
            cfg_beta_beta=7.0,
            late_cfg_scale=0.92,
            late_cfg_start=0.76,
        )
        cfg_end = cfg_at_progress(
            1.0,
            base_cfg=6.0,
            cfg_schedule_mode="beta_bump",
            cfg_early_scale=0.98,
            cfg_early_ramp_end=0.10,
            cfg_peak_boost=0.60,
            cfg_bump_start=0.08,
            cfg_bump_end=0.68,
            cfg_beta_alpha=4.0,
            cfg_beta_beta=7.0,
            late_cfg_scale=0.92,
            late_cfg_start=0.76,
        )

        self.assertAlmostEqual(cfg_start, 5.88, places=4)
        self.assertAlmostEqual(cfg_peak, 6.6, places=4)
        self.assertAlmostEqual(cfg_end, 5.52, places=4)
        self.assertGreater(cfg_peak, cfg_start)
        self.assertGreater(cfg_peak, cfg_end)

    def test_limited_interval_cfg_boosts_inside_window(self):
        cfg_before = cfg_at_progress(
            0.05,
            base_cfg=6.0,
            cfg_schedule_mode="limited_interval",
            cfg_peak_boost=0.8,
            cfg_interval_start=0.10,
            cfg_interval_rise_end=0.20,
            cfg_interval_fall_start=0.40,
            cfg_interval_end=0.60,
        )
        cfg_plateau = cfg_at_progress(
            0.30,
            base_cfg=6.0,
            cfg_schedule_mode="limited_interval",
            cfg_peak_boost=0.8,
            cfg_interval_start=0.10,
            cfg_interval_rise_end=0.20,
            cfg_interval_fall_start=0.40,
            cfg_interval_end=0.60,
        )
        cfg_after = cfg_at_progress(
            0.70,
            base_cfg=6.0,
            cfg_schedule_mode="limited_interval",
            cfg_peak_boost=0.8,
            cfg_interval_start=0.10,
            cfg_interval_rise_end=0.20,
            cfg_interval_fall_start=0.40,
            cfg_interval_end=0.60,
        )

        self.assertAlmostEqual(cfg_before, 6.0)
        self.assertAlmostEqual(cfg_plateau, 6.8)
        self.assertAlmostEqual(cfg_after, 6.0)

    def test_constant_cfg_ignores_curve_parameters(self):
        cfg = cfg_at_progress(
            0.28,
            base_cfg=6.0,
            cfg_schedule_mode="constant",
            early_cfg_boost=4.0,
            early_cfg_until=0.5,
            late_cfg_scale=0.5,
            late_cfg_start=0.2,
            cfg_early_scale=0.5,
            cfg_peak_boost=5.0,
        )

        self.assertAlmostEqual(cfg, 6.0)

    def test_cfg_schedule_position_can_use_step_progress(self):
        sigmas = torch.tensor([0.8, 0.5, 0.2, 0.0])

        self.assertAlmostEqual(
            cfg_schedule_position(
                torch,
                sigmas[1],
                sigmas,
                1,
                domain="progress",
                total_steps=3,
            ),
            0.5,
        )

    def test_cfg_schedule_position_can_use_rf_time(self):
        sigmas = torch.tensor([0.8, 0.5, 0.2, 0.0])

        self.assertAlmostEqual(
            cfg_schedule_position(
                torch,
                sigmas[1],
                sigmas,
                1,
                domain="rf_t",
                total_steps=3,
            ),
            0.5,
        )

    def test_cfg_schedule_position_can_use_lambda(self):
        sigmas = torch.tensor([0.9, 0.7, 0.2, 0.0])

        actual = cfg_schedule_position(
            torch,
            sigmas[1],
            sigmas,
            1,
            domain="lambda",
            total_steps=3,
        )
        lambda_start = torch.log(torch.tensor((1.0 - 0.9) / 0.9))
        lambda_mid = torch.log(torch.tensor((1.0 - 0.7) / 0.7))
        lambda_end = torch.log(torch.tensor((1.0 - 0.2) / 0.2))
        expected = float((lambda_mid - lambda_start) / (lambda_end - lambda_start))

        self.assertAlmostEqual(actual, expected, places=6)
        self.assertNotAlmostEqual(actual, 0.5, places=3)

    def test_model_sampling_shift_description_marks_flow_cosmos_bypass(self):
        model = _DummyModel(torch.linspace(0.0, 1.0, 101), sampling_class_name="ModelSamplingDiscreteFlow")
        model.sampling.shift = 3.0

        self.assertIn("bypassed by flow_cosmos", _describe_model_sampling_shift(model, flow_schedule="flow_cosmos"))

    def test_4d_image_latent_gets_temporal_axis_for_cosmos(self):
        latent = {"samples": torch.zeros(1, 16, 64, 64)}

        normalized, added_temporal_dim, original_shape, channel_adapter = _normalize_cosmos_latent(
            latent,
            expected_latent_channels=16,
        )

        self.assertTrue(added_temporal_dim)
        self.assertEqual(original_shape, "[1, 16, 64, 64]")
        self.assertEqual(channel_adapter, "")
        self.assertEqual(tuple(normalized["samples"].shape), (1, 16, 1, 64, 64))

    def test_4_channel_empty_latent_is_promoted_to_anima_latent_channels(self):
        latent = {"samples": torch.ones(1, 4, 64, 64)}

        normalized, added_temporal_dim, original_shape, channel_adapter = _normalize_cosmos_latent(
            latent,
            expected_latent_channels=16,
        )

        self.assertTrue(added_temporal_dim)
        self.assertEqual(original_shape, "[1, 4, 64, 64]")
        self.assertIn("4->16", channel_adapter)
        self.assertEqual(tuple(normalized["samples"].shape), (1, 16, 1, 64, 64))
        self.assertTrue(torch.equal(normalized["samples"][:, :4], latent["samples"].unsqueeze(2)))
        self.assertEqual(float(normalized["samples"][:, 4:].sum()), 0.0)

    def test_anima_x_embedder_width_infers_16_latent_channels(self):
        self.assertEqual(_infer_cosmos_latent_channels(68, 4), 16)

    def test_sampler_output_restores_to_sampler_channel_count(self):
        state = torch.ones(1, 16, 1, 8, 8)
        denoised = torch.zeros(1, 20, 1, 8, 8)

        restored = _restore_sampler_channels(denoised, state)
        self.assertEqual(tuple(restored.shape), tuple(state.shape))

    def test_flow_euler_step_integrates_toward_denoised(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        sigma = torch.tensor(1.0)
        sigma_next = torch.tensor(0.5)

        self.assertTrue(torch.equal(flow_euler_step(x, denoised, sigma, sigma_next), torch.tensor([6.0])))

    def test_flow_heun_step_matches_euler_for_constant_x0(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)

        self.assertTrue(
            torch.allclose(
                flow_heun_step(x, denoised, denoised, sigma, sigma_next),
                flow_euler_step(x, denoised, sigma, sigma_next),
            )
        )

    def test_flow_heun_step_uses_endpoint_x0_integral(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        denoised_pred = torch.tensor([4.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)

        out = flow_heun_step(x, denoised, denoised_pred, sigma, sigma_next)

        lambda_current = torch.log(torch.tensor((1.0 - 0.5) / 0.5))
        lambda_next = torch.log(torch.tensor((1.0 - 0.25) / 0.25))
        h = lambda_next - lambda_current
        k0 = torch.expm1(h)
        k1 = torch.exp(h) * h - k0
        endpoint_weight = k1 / h
        current_weight = k0 - endpoint_weight
        expected = torch.tensor([5.0]) + torch.tensor(0.25) * torch.exp(lambda_current) * (
            current_weight * denoised + endpoint_weight * denoised_pred
        )

        self.assertTrue(torch.allclose(out, expected))

    def test_flow_pc3_predictor_uses_previous_x0_history(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)
        previous_denoised = torch.tensor([1.0])
        previous_lambda = torch.log(torch.tensor(0.25 / 0.75))

        out = flow_pc3_predictor_step(
            x,
            denoised,
            sigma,
            sigma_next,
            state=FlowPC3State(
                previous_denoised=previous_denoised,
                previous_lambda=previous_lambda,
            ),
        )

        lambda_current = torch.log(torch.tensor((1.0 - 0.5) / 0.5))
        lambda_next = torch.log(torch.tensor((1.0 - 0.25) / 0.25))
        h = lambda_next - lambda_current
        h_previous = lambda_current - previous_lambda
        k0 = torch.expm1(h)
        k1 = torch.exp(h) * h - k0
        current_weight = k0 + k1 / h_previous
        previous_weight = -k1 / h_previous
        expected = torch.tensor([5.0]) + torch.tensor(0.25) * torch.exp(lambda_current) * (
            current_weight * denoised + previous_weight * previous_denoised
        )

        self.assertTrue(torch.allclose(out, expected))
        self.assertFalse(torch.allclose(out, flow_euler_step(x, denoised, sigma, sigma_next)))

    def test_flow_pc3_without_history_matches_heun_and_stores_current_x0(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        denoised_pred = torch.tensor([4.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)

        out, state = flow_pc3_damped_step(
            x,
            denoised,
            denoised_pred,
            sigma,
            sigma_next,
            state=FlowPC3State(),
            max_gamma=1.0,
            tolerance=1.0,
        )

        self.assertTrue(torch.allclose(out, flow_heun_step(x, denoised, denoised_pred, sigma, sigma_next)))
        self.assertTrue(torch.equal(state.previous_denoised, denoised))
        self.assertTrue(torch.allclose(state.previous_lambda, torch.log(torch.tensor((1.0 - 0.5) / 0.5))))

    def test_flow_pc3_zero_gamma_matches_heun_with_history(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        denoised_pred = torch.tensor([4.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)
        state = FlowPC3State(
            previous_denoised=torch.tensor([1.0]),
            previous_lambda=torch.log(torch.tensor(0.25 / 0.75)),
        )

        out, _state = flow_pc3_damped_step(
            x,
            denoised,
            denoised_pred,
            sigma,
            sigma_next,
            state=state,
            max_gamma=0.0,
            tolerance=1.0,
        )

        self.assertTrue(torch.allclose(out, flow_heun_step(x, denoised, denoised_pred, sigma, sigma_next)))

    def test_flow_pc3_applies_damped_am3_with_history(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        denoised_pred = torch.tensor([4.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)
        previous_denoised = torch.tensor([1.0])
        previous_lambda = torch.log(torch.tensor(0.25 / 0.75))

        out, state = flow_pc3_damped_step(
            x,
            denoised,
            denoised_pred,
            sigma,
            sigma_next,
            state=FlowPC3State(
                previous_denoised=previous_denoised,
                previous_lambda=previous_lambda,
            ),
            max_gamma=1.0,
            tolerance=1.0,
        )

        lambda_current = torch.log(torch.tensor((1.0 - 0.5) / 0.5))
        lambda_next = torch.log(torch.tensor((1.0 - 0.25) / 0.25))
        h = lambda_next - lambda_current
        previous_node = previous_lambda - lambda_current
        k0 = torch.expm1(h)
        k1 = torch.exp(h) * h - k0
        k2 = torch.exp(h) * h * h - 2.0 * k1
        previous_weight = (k2 - h * k1) / (previous_node * (previous_node - h))
        current_weight = (k2 - (previous_node + h) * k1 + previous_node * h * k0) / (
            previous_node * h
        )
        endpoint_weight = (k2 - previous_node * k1) / ((h - previous_node) * h)
        x_heun = flow_heun_step(x, denoised, denoised_pred, sigma, sigma_next)
        x_am3 = torch.tensor([5.0]) + torch.tensor(0.25) * torch.exp(lambda_current) * (
            previous_weight * previous_denoised
            + current_weight * denoised
            + endpoint_weight * denoised_pred
        )
        error = torch.sqrt(torch.mean((x_am3 - x_heun).float() ** 2)) / (
            torch.sqrt(torch.mean(x_heun.float() ** 2)) + 1e-6
        )
        gamma_error = torch.clamp(torch.sqrt(torch.tensor(1.0) / (error + 1e-6)), min=0.0, max=1.0)
        gamma_lambda = torch.sigmoid((lambda_current + 2.5) / 0.5) * torch.sigmoid(
            (4.5 - lambda_next) / 0.8
        )
        expected = x_heun + gamma_lambda * gamma_error * (x_am3 - x_heun)

        self.assertTrue(torch.allclose(out, expected))
        self.assertFalse(torch.allclose(out, x_heun))
        self.assertTrue(torch.equal(state.previous_denoised, denoised))

    def test_flow_pc3_result_exposes_cache_metrics_inputs(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        denoised_pred = torch.tensor([4.0])
        sigma = torch.tensor(0.5)
        sigma_next = torch.tensor(0.25)
        result = flow_pc3_damped_step_result(
            x,
            denoised,
            denoised_pred,
            sigma,
            sigma_next,
            state=FlowPC3State(
                previous_denoised=torch.tensor([1.0]),
                previous_lambda=torch.log(torch.tensor(0.25 / 0.75)),
            ),
            max_gamma=1.0,
            tolerance=1.0,
        )

        out, state = flow_pc3_damped_step(
            x,
            denoised,
            denoised_pred,
            sigma,
            sigma_next,
            state=FlowPC3State(
                previous_denoised=torch.tensor([1.0]),
                previous_lambda=torch.log(torch.tensor(0.25 / 0.75)),
            ),
            max_gamma=1.0,
            tolerance=1.0,
        )

        self.assertTrue(torch.allclose(result.x, out))
        self.assertTrue(torch.equal(result.state.previous_denoised, state.previous_denoised))
        self.assertFalse(torch.allclose(result.x_heun, result.x_am3))
        self.assertGreater(float(result.gamma), 0.0)

    def test_flow_3m_damped_zero_gamma_matches_2m(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([3.0])
        sigma = 1.0 / (1.0 + torch.exp(torch.tensor(0.6)))
        sigma_next = 1.0 / (1.0 + torch.exp(torch.tensor(0.9)))
        state = FlowERState(
            previous_denoised=torch.tensor([2.0]),
            previous_lambda=torch.tensor(0.3),
            previous_previous_denoised=torch.tensor([1.0]),
            previous_previous_lambda=torch.tensor(0.0),
        )

        result = flow_3m_damped_step(
            x,
            denoised,
            sigma,
            sigma_next,
            state=state,
            max_gamma=0.0,
            tolerance=1.0,
        )
        er_2m, _ = flow_er_step(x, denoised, sigma, sigma_next, state=state, max_order=2)

        self.assertEqual(result.order, 2)
        self.assertTrue(torch.allclose(result.x, er_2m))

    def test_flow_3m_damped_blends_toward_er_order3(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([3.0])
        sigma = 1.0 / (1.0 + torch.exp(torch.tensor(0.6)))
        sigma_next = 1.0 / (1.0 + torch.exp(torch.tensor(0.9)))
        state = FlowERState(
            previous_denoised=torch.tensor([2.0]),
            previous_lambda=torch.tensor(0.3),
            previous_previous_denoised=torch.tensor([1.0]),
            previous_previous_lambda=torch.tensor(0.0),
        )

        result = flow_3m_damped_step(
            x,
            denoised,
            sigma,
            sigma_next,
            state=state,
            max_gamma=1.0,
            tolerance=100.0,
        )
        er_2m, _ = flow_er_step(x, denoised, sigma, sigma_next, state=state, max_order=2)
        er_3m, _ = flow_er_step(x, denoised, sigma, sigma_next, state=state, max_order=3)
        expected = er_2m + result.gamma3 * (er_3m - er_2m)

        self.assertEqual(result.order, 3)
        self.assertGreater(float(result.gamma3), 0.0)
        self.assertLess(float(result.gamma3), 1.0)
        self.assertTrue(torch.allclose(result.x, expected, atol=1e-5))

    def test_pc3_fsal_cache_score_accepts_small_endpoint_correction(self):
        x_pred = torch.ones(1, 4, 2, 2)
        x_heun = x_pred + 0.0001
        x_am3 = x_heun + 0.00005
        x_next = x_heun + 0.00002

        score, e_x, e_pc3 = _pc3_fsal_cache_score(
            torch,
            x_pred=x_pred,
            x_next=x_next,
            x_heun=x_heun,
            x_am3=x_am3,
            t_next=torch.tensor(0.5),
            tolerance=0.005,
        )

        self.assertLess(score, 1.0)
        self.assertLess(e_x, 0.001)
        self.assertLess(e_pc3, 0.001)

    def test_pc3_fsal_cache_score_rejects_large_endpoint_correction(self):
        x_pred = torch.ones(1, 4, 2, 2)
        x_heun = x_pred + 0.2
        x_am3 = x_heun + 0.2
        x_next = x_am3

        score, _e_x, _e_pc3 = _pc3_fsal_cache_score(
            torch,
            x_pred=x_pred,
            x_next=x_next,
            x_heun=x_heun,
            x_am3=x_am3,
            t_next=torch.tensor(0.1),
            tolerance=0.005,
        )

        self.assertGreaterEqual(score, 1.0)

    def test_rf_endpoint_noise_refresh_refreshes_endpoint_noise(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        sigma = torch.tensor(1.0)
        sigma_next = torch.tensor(0.5)
        euler_next = flow_euler_step(x, denoised, sigma, sigma_next)
        generator = torch.Generator().manual_seed(123)
        expected_generator = torch.Generator().manual_seed(123)

        out, applied = rf_endpoint_noise_refresh(
            torch,
            euler_next,
            x,
            denoised,
            sigma,
            sigma_next,
            generator,
            refresh_strength=0.5,
            refresh_until=0.0,
            refresh_from=None,
        )
        expected_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=expected_generator)
        endpoint_noise = (x - (1.0 - sigma) * denoised) / sigma
        refreshed_noise = (1.0 - 0.5**2) ** 0.5 * endpoint_noise + 0.5 * expected_noise
        expected = euler_next + sigma_next * (refreshed_noise - endpoint_noise)

        self.assertTrue(applied)
        self.assertTrue(torch.allclose(out, expected))

    def test_rf_endpoint_noise_refresh_strength_zero_matches_deterministic_flow(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        euler_next = flow_euler_step(x, denoised, torch.tensor(1.0), torch.tensor(0.5))

        out, applied = rf_endpoint_noise_refresh(
            torch,
            euler_next,
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.5),
            None,
            refresh_strength=0.0,
        )
        self.assertFalse(applied)
        self.assertTrue(torch.allclose(out, euler_next))

    def test_rf_endpoint_noise_refresh_respects_until_gate(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        euler_next = flow_euler_step(x, denoised, torch.tensor(1.0), torch.tensor(0.1))

        out, applied = rf_endpoint_noise_refresh(
            torch,
            euler_next,
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.1),
            torch.Generator().manual_seed(123),
            refresh_strength=1.0,
            refresh_until=0.2,
        )
        self.assertFalse(applied)
        self.assertTrue(torch.allclose(out, euler_next))

    def test_rf_endpoint_noise_refresh_preserves_non_euler_solver_when_strength_zero(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        heun_next = torch.tensor([7.0])

        out, applied = rf_endpoint_noise_refresh(
            torch,
            heun_next,
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.5),
            None,
            enabled=True,
            refresh_strength=0.0,
            refresh_until=0.2,
        )

        self.assertFalse(applied)
        self.assertTrue(torch.equal(out, heun_next))

    def test_rf_endpoint_noise_refresh_adds_delta_directly_to_solver_output(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        heun_next = torch.tensor([7.0])
        sigma = torch.tensor(0.9)
        sigma_next = torch.tensor(0.5)
        generator = torch.Generator().manual_seed(123)
        expected_generator = torch.Generator().manual_seed(123)

        out, applied = rf_endpoint_noise_refresh(
            torch,
            heun_next,
            x,
            denoised,
            sigma,
            sigma_next,
            generator,
            enabled=True,
            refresh_strength=0.5,
            refresh_until=0.0,
        )
        expected_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=expected_generator)
        endpoint_noise = (x - (1.0 - sigma) * denoised) / sigma
        refreshed_noise = (1.0 - 0.5**2) ** 0.5 * endpoint_noise + 0.5 * expected_noise
        expected = heun_next + sigma_next * (refreshed_noise - endpoint_noise)

        self.assertTrue(applied)
        self.assertTrue(torch.allclose(out, expected))

    def test_rf_endpoint_noise_refresh_returns_deterministic_at_terminal_sigma(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])

        out, applied = rf_endpoint_noise_refresh(
            torch,
            denoised,
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.0),
            None,
        )
        self.assertFalse(applied)
        self.assertTrue(torch.equal(out, denoised))

    def test_randn_like_falls_back_when_generator_device_is_rejected(self):
        class RejectingTorch:
            def __init__(self):
                self.calls = 0

            def randn(self, shape, *, dtype, layout, device, generator=None):
                self.calls += 1
                if generator is not None:
                    raise RuntimeError("generator device mismatch")
                return torch.zeros(shape, dtype=dtype, device=device)

        fake_torch = RejectingTorch()
        x = torch.ones(1, 2)

        out = _randn_like(fake_torch, x, object())

        self.assertEqual(fake_torch.calls, 2)
        self.assertTrue(torch.equal(out, torch.zeros_like(x)))

    def test_flow_velocity_uses_denoised_x0(self):
        self.assertTrue(
            torch.equal(
                flow_velocity(torch.tensor([10.0]), torch.tensor([2.0]), torch.tensor(2.0)),
                torch.tensor([4.0]),
            )
        )

    def test_flow_euler_step_returns_denoised_at_terminal_sigma(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])

        self.assertTrue(
            torch.equal(
                flow_euler_step(x, denoised, torch.tensor(1.0), torch.tensor(0.0)),
                denoised,
            )
        )

    def test_flow_er_step_uses_data_prediction_update(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])

        out, state = flow_er_step(
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.5),
            state=FlowERState(),
        )

        self.assertTrue(torch.equal(out, torch.tensor([6.0])))
        self.assertTrue(torch.equal(state.previous_denoised, denoised))

    def test_flow_er_step_can_downshift_to_first_order(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        contaminated_state = FlowERState(
            previous_denoised=torch.tensor([100.0]),
            previous_lambda=torch.tensor(2.0),
        )

        out, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.5),
            state=contaminated_state,
            max_order=1,
        )

        self.assertTrue(torch.equal(out, torch.tensor([6.0])))

    def test_flow_er_lms2_uses_previous_x0_slope(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])
        previous_denoised = torch.tensor([1.0])
        lambda_previous = torch.log(torch.tensor(0.25 / 0.75))

        out, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(0.5),
            torch.tensor(0.25),
            state=FlowERState(
                previous_denoised=previous_denoised,
                previous_lambda=lambda_previous,
            ),
        )

        lambda_current = torch.log(torch.tensor((1.0 - 0.5) / 0.5))
        lambda_next = torch.log(torch.tensor((1.0 - 0.25) / 0.25))
        h = lambda_next - lambda_current
        k0 = torch.expm1(h)
        k1 = torch.exp(h) * h - k0
        slope = (denoised - previous_denoised) / (lambda_current - lambda_previous)
        expected = torch.tensor([6.0]) + torch.tensor(0.25) * torch.exp(lambda_current) * k1 * slope

        self.assertTrue(torch.allclose(out, expected))

    def test_flow_er_lms3_uses_two_previous_x0_predictions(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([4.0])
        previous_denoised = torch.tensor([1.0])
        previous_previous_denoised = torch.tensor([3.0])
        lambda_previous = torch.log(torch.tensor(0.25 / 0.75))
        lambda_previous_previous = torch.log(torch.tensor(0.1 / 0.9))

        out, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(0.5),
            torch.tensor(0.25),
            state=FlowERState(
                previous_denoised=previous_denoised,
                previous_lambda=lambda_previous,
                previous_previous_denoised=previous_previous_denoised,
                previous_previous_lambda=lambda_previous_previous,
            ),
            max_order=3,
        )
        out_order2, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(0.5),
            torch.tensor(0.25),
            state=FlowERState(
                previous_denoised=previous_denoised,
                previous_lambda=lambda_previous,
                previous_previous_denoised=previous_previous_denoised,
                previous_previous_lambda=lambda_previous_previous,
            ),
            max_order=2,
        )

        lambda_current = torch.log(torch.tensor((1.0 - 0.5) / 0.5))
        lambda_next = torch.log(torch.tensor((1.0 - 0.25) / 0.25))
        h = lambda_next - lambda_current
        k0 = torch.expm1(h)
        k1 = torch.exp(h) * h - k0
        k2 = torch.exp(h) * h * h - 2.0 * k1
        a = lambda_previous - lambda_current
        b = lambda_previous_previous - lambda_current
        w0 = (k2 - (a + b) * k1 + a * b * k0) / (a * b)
        w1 = (k2 - b * k1) / (a * (a - b))
        w2 = (k2 - a * k1) / (b * (b - a))
        expected = torch.tensor([5.0]) + torch.tensor(0.25) * torch.exp(lambda_current) * (
            w0 * denoised + w1 * previous_denoised + w2 * previous_previous_denoised
        )

        self.assertTrue(torch.allclose(out, expected))
        self.assertFalse(torch.allclose(out, out_order2))

    def test_flow_er_without_history_matches_flow_ode_step(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])

        out, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.5),
            state=FlowERState(),
        )

        self.assertTrue(torch.equal(out, torch.tensor([6.0])))

    def test_flow_er_step_returns_denoised_at_terminal_sigma(self):
        x = torch.tensor([10.0])
        denoised = torch.tensor([2.0])

        out, _state = flow_er_step(
            x,
            denoised,
            torch.tensor(1.0),
            torch.tensor(0.0),
            state=FlowERState(),
        )

        self.assertTrue(torch.equal(out, denoised))


class _DummySampling:
    def __init__(self, sigmas):
        self.sigmas = sigmas


class _DummyModel:
    def __init__(self, sigmas, *, sampling_class_name="_DummySampling"):
        sampling_type = type(sampling_class_name, (_DummySampling,), {})
        self.sampling = sampling_type(sigmas)

    def get_model_object(self, name):
        if name != "model_sampling":
            raise KeyError(name)
        return self.sampling


def _dummy_sampler_log(
    *,
    actual_steps: int,
    sampler_core: str,
    actual_model_calls: int | None = None,
    cache_candidates: int = 0,
    cache_accepts: int = 0,
    cache_rejects: int = 0,
    pc3_used_total: int = 0,
    mean_gamma3: float | None = None,
) -> AnimaSamplerLog:
    return AnimaSamplerLog(
        requested_steps=actual_steps,
        actual_steps=actual_steps,
        latent_in_shape="[1, 16, 1, 8, 8]",
        latent_sample_shape="[1, 16, 1, 8, 8]",
        added_temporal_dim=False,
        channel_adapter="none",
        x_embedder_features="unknown",
        sampler_core=sampler_core,
        flow_schedule="flow_cosmos",
        flow_shift=5.0,
        cfg_schedule_mode="beta_bump",
        cfg_schedule_domain="lambda",
        denoise_legacy_progress=False,
        model_sampling_shift="none",
        denoise=1.0,
        flow_er_order=2,
        flow_pc3_gamma=1.0,
        flow_pc3_tolerance=0.005,
        cosmos_sigma_max=80.0,
        cosmos_sigma_min=0.002,
        cfg_start=6.5,
        cfg_mid=6.0,
        cfg_end=6.0,
        rf_endpoint_noise_refresh_enabled=False,
        rf_endpoint_noise_refresh_strength=0.15,
        rf_endpoint_noise_refresh_until=0.20,
        actual_model_calls=actual_model_calls,
        cache_candidates=cache_candidates,
        cache_accepts=cache_accepts,
        cache_rejects=cache_rejects,
        pc3_used_total=pc3_used_total,
        mean_gamma3=mean_gamma3,
    )


if __name__ == "__main__":
    unittest.main()
