import unittest

from anima_sampler.scheduler import (
    PhaseSteps,
    allocate_phase_steps,
    build_anchored_sigmas,
    build_early_dense_sigmas,
    build_flow_cosmos_beta_sigmas,
    build_flow_cosmos_lambda_biased_sigmas,
    build_flow_cosmos_rho_rf_tail_sigmas,
    build_flow_cosmos_rho_sigmas,
    build_flow_cosmos_sigmas,
    build_phase_positions,
    build_simple_sigmas,
)


class SchedulerTests(unittest.TestCase):
    def test_simple_matches_comfyui_index_pattern(self):
        sigmas = [index / 100 for index in range(101)]

        self.assertEqual(
            build_simple_sigmas(sigmas, 4),
            [1.0, 0.75, 0.5, 0.25, 0.0],
        )

    def test_simple_interpolates_when_steps_exceed_table_resolution(self):
        sigmas = [0.0, 0.25, 0.5, 0.75, 1.0]

        out = build_simple_sigmas(sigmas, 8)

        self.assertEqual(len(out), 9)
        self.assertEqual(out[0], 1.0)
        self.assertEqual(out[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(out, out[1:])))
        self.assertEqual(len(set(out)), len(out))

    def test_flow_cosmos_maps_external_sigmas_to_normalized_flow_time(self):
        sigmas = build_flow_cosmos_sigmas(4, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(len(sigmas), 5)
        self.assertAlmostEqual(sigmas[0], 80.0 / 81.0, places=6)
        self.assertAlmostEqual(sigmas[-2], 0.002 / 1.002, places=6)
        self.assertGreater(sigmas[1], sigmas[2])
        self.assertEqual(sigmas[-1], 0.0)

    def test_flow_cosmos_uses_exact_endpoints_at_normal_step_counts(self):
        sigmas = build_flow_cosmos_sigmas(35, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(len(sigmas), 36)
        self.assertAlmostEqual(sigmas[0], 80.0 / 81.0, places=6)
        self.assertAlmostEqual(sigmas[-2], 0.002 / 1.002, places=6)
        self.assertEqual(sigmas[-1], 0.0)

    def test_flow_cosmos_beta5_applies_report_shift_in_sigma_ratio(self):
        base = build_flow_cosmos_sigmas(4, sigma_max=80.0, sigma_min=0.002)
        shifted = build_flow_cosmos_beta_sigmas(4, beta=5.0, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(len(shifted), 5)
        self.assertAlmostEqual(shifted[0], 400.0 / 401.0, places=6)
        self.assertAlmostEqual(shifted[-2], 0.01 / 1.01, places=6)
        self.assertGreater(shifted[0], base[0])
        self.assertGreater(shifted[1], shifted[2])
        self.assertEqual(shifted[-1], 0.0)

    def test_flow_cosmos_beta5_uses_exact_shifted_endpoints_at_normal_step_counts(self):
        shifted = build_flow_cosmos_beta_sigmas(35, beta=5.0, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(len(shifted), 36)
        self.assertAlmostEqual(shifted[0], 400.0 / 401.0, places=6)
        self.assertAlmostEqual(shifted[-2], 0.01 / 1.01, places=6)
        self.assertEqual(shifted[-1], 0.0)

    def test_flow_cosmos_beta_default_zero_is_unshifted(self):
        base = build_flow_cosmos_sigmas(4, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(
            build_flow_cosmos_beta_sigmas(4, sigma_max=80.0, sigma_min=0.002),
            base,
        )
        self.assertEqual(
            build_flow_cosmos_beta_sigmas(4, beta=0.0, sigma_max=80.0, sigma_min=0.002),
            base,
        )

    def test_flow_cosmos_lambda_biased_keeps_cosmos_endpoints(self):
        sigmas = build_flow_cosmos_lambda_biased_sigmas(
            8,
            strength="default",
            sigma_max=80.0,
            sigma_min=0.002,
        )

        self.assertEqual(len(sigmas), 9)
        self.assertAlmostEqual(sigmas[0], 80.0 / 81.0, places=6)
        self.assertAlmostEqual(sigmas[-2], 0.002 / 1.002, places=6)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:-1])))

    def test_flow_cosmos_lambda_biased_strength_changes_density(self):
        light = build_flow_cosmos_lambda_biased_sigmas(36, strength="light")
        strong = build_flow_cosmos_lambda_biased_sigmas(36, strength="strong")

        light_lambdas = [_flow_lambda(value) for value in light[:-1]]
        strong_lambdas = [_flow_lambda(value) for value in strong[:-1]]

        light_mid_gap = _gap_containing(light_lambdas, 0.8)
        strong_mid_gap = _gap_containing(strong_lambdas, 0.8)
        self.assertLess(strong_mid_gap, light_mid_gap)

    def test_flow_cosmos_rho7_uses_predict2_style_order_grid(self):
        sigmas = build_flow_cosmos_rho_sigmas(4, order=7.0, sigma_max=80.0, sigma_min=0.002)

        self.assertEqual(len(sigmas), 5)
        self.assertAlmostEqual(sigmas[0], 80.0 / 81.0, places=6)
        self.assertGreater(sigmas[1], sigmas[2])
        self.assertAlmostEqual(sigmas[-2], 0.002 / 1.002, places=6)
        self.assertEqual(sigmas[-1], 0.0)

    def test_flow_cosmos_rho7_rf_tail_keeps_endpoints_and_monotonicity(self):
        sigmas = build_flow_cosmos_rho_rf_tail_sigmas(
            35,
            tail_lambda_start=0.5,
            order=7.0,
            sigma_max=80.0,
            sigma_min=0.002,
        )

        self.assertEqual(len(sigmas), 36)
        self.assertAlmostEqual(sigmas[0], 80.0 / 81.0, places=6)
        self.assertAlmostEqual(sigmas[-2], 0.002 / 1.002, places=6)
        self.assertEqual(sigmas[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(sigmas, sigmas[1:-1])))

    def test_flow_cosmos_rho7_rf_tail_reduces_late_rf_log_time_gap(self):
        rho7 = build_flow_cosmos_rho_sigmas(35, order=7.0, sigma_max=80.0, sigma_min=0.002)
        hybrid = build_flow_cosmos_rho_rf_tail_sigmas(
            35,
            tail_lambda_start=0.5,
            order=7.0,
            sigma_max=80.0,
            sigma_min=0.002,
        )

        rho7_last_gap = _flow_ell(rho7[-2]) - _flow_ell(rho7[-3])
        hybrid_last_gap = _flow_ell(hybrid[-2]) - _flow_ell(hybrid[-3])

        self.assertLess(hybrid_last_gap, rho7_last_gap)

    def test_flow_cosmos_rho7_rf_tail_preserves_rho_prefix_before_switch(self):
        rho7 = build_flow_cosmos_rho_sigmas(35, order=7.0, sigma_max=80.0, sigma_min=0.002)
        hybrid = build_flow_cosmos_rho_rf_tail_sigmas(
            35,
            tail_lambda_start=0.5,
            order=7.0,
            sigma_max=80.0,
            sigma_min=0.002,
        )
        sigma_switch = 0.6065306597126334
        rho7_external = [_external_sigma_from_flow_t(value) for value in rho7[:-1]]
        hybrid_external = [_external_sigma_from_flow_t(value) for value in hybrid[:-1]]
        prefix_count = sum(sigma > sigma_switch for sigma in rho7_external)

        self.assertGreater(prefix_count, 1)
        for actual, expected in zip(hybrid_external[:prefix_count], rho7_external[:prefix_count]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertAlmostEqual(hybrid_external[prefix_count], sigma_switch, places=6)

    def test_flow_cosmos_rho7_rf_tail_auto_switches_on_late_ell_gap(self):
        rho7 = build_flow_cosmos_rho_sigmas(35, order=7.0, sigma_max=80.0, sigma_min=0.002)
        auto = build_flow_cosmos_rho_rf_tail_sigmas(
            35,
            tail_lambda_start=None,
            tail_delta_ell_max=0.5,
            order=7.0,
            sigma_max=80.0,
            sigma_min=0.002,
        )

        self.assertEqual(len(auto), 36)
        self.assertAlmostEqual(auto[0], 80.0 / 81.0, places=6)
        self.assertAlmostEqual(auto[-2], 0.002 / 1.002, places=6)
        self.assertEqual(auto[-1], 0.0)
        self.assertTrue(all(left > right for left, right in zip(auto, auto[1:-1])))
        self.assertLess(_flow_ell(auto[-2]) - _flow_ell(auto[-3]), _flow_ell(rho7[-2]) - _flow_ell(rho7[-3]))

        first_tail_index = next(
            index
            for index, (rho_value, auto_value) in enumerate(zip(rho7[:-1], auto[:-1]))
            if abs(rho_value - auto_value) > 1e-10
        )
        tail_gaps = [
            _flow_ell(auto[index + 1]) - _flow_ell(auto[index])
            for index in range(first_tail_index - 1, len(auto) - 2)
        ]
        self.assertLessEqual(max(tail_gaps), 0.5 + 1e-9)

    def test_phase_steps_sum_to_requested_steps(self):
        phase_steps = allocate_phase_steps(
            36,
            early_step_ratio=0.50,
            mid_step_ratio=0.32,
        )

        self.assertEqual(phase_steps.total, 36)
        self.assertEqual(phase_steps, PhaseSteps(early=18, mid=12, late=6))

    def test_phase_positions_include_boundaries_and_end_at_zero(self):
        positions = build_phase_positions(
            PhaseSteps(early=2, mid=1, late=1),
            early_end=0.7,
            mid_end=0.2,
        )

        self.assertEqual(positions, [1.0, 0.85, 0.7, 0.2, 0.0])

    def test_early_dense_spends_more_entries_above_boundary(self):
        sigmas = [index / 1000 for index in range(1001)]
        dense = build_early_dense_sigmas(
            sigmas,
            36,
            early_step_ratio=0.50,
            mid_step_ratio=0.32,
            early_end=0.7,
            mid_end=0.22,
        )

        self.assertEqual(len(dense), 37)
        self.assertEqual(dense[0], 1.0)
        self.assertEqual(dense[-1], 0.0)
        self.assertEqual(sum(sigma >= 0.7 for sigma in dense), 19)

    def test_rejects_invalid_boundaries(self):
        with self.assertRaises(ValueError):
            build_early_dense_sigmas([0.0, 0.5, 1.0], 10, early_end=0.2, mid_end=0.7)

    def test_acceleration_lora_14_step_anchor_shape_can_be_recreated(self):
        sigmas = [index / 1000 for index in range(1001)]
        anchored = build_anchored_sigmas(
            sigmas,
            anchor_positions=[1.0, 0.9375, 0.8333333, 0.625, 0.0],
            interval_steps=[5, 5, 2, 2],
        )

        expected = [
            1.0,
            0.9875,
            0.975,
            0.9625,
            0.95,
            0.9375,
            0.9166667,
            0.8958333,
            0.875,
            0.8541667,
            0.8333333,
            0.7291667,
            0.625,
            0.3125,
            0.0,
        ]
        for actual, target in zip(anchored, expected):
            self.assertAlmostEqual(actual, target, places=5)

    def test_wan_20_step_reference_anchor_shape_can_be_calculated(self):
        sigmas = [index / 1000 for index in range(1001)]
        anchored = build_anchored_sigmas(
            sigmas,
            anchor_positions=[1.0, 0.9375, 0.8333333, 0.625, 0.0],
            interval_steps=[5, 5, 5, 5],
        )

        expected = [
            1.0,
            0.9875,
            0.975,
            0.9625,
            0.95,
            0.9375,
            0.9166667,
            0.8958333,
            0.875,
            0.8541667,
            0.8333333,
            0.7916666,
            0.75,
            0.7083333,
            0.6666667,
            0.625,
            0.5,
            0.375,
            0.25,
            0.125,
            0.0,
        ]
        self.assertEqual(len(anchored), 21)
        for actual, target in zip(anchored, expected):
            self.assertAlmostEqual(actual, target, places=5)


def _flow_lambda(t: float) -> float:
    import math

    return math.log((1.0 - t) / t)


def _flow_ell(t: float) -> float:
    import math

    return -math.log(t)


def _external_sigma_from_flow_t(t: float) -> float:
    return t / (1.0 - t)


def _gap_containing(values: list[float], target: float) -> float:
    for left, right in zip(values, values[1:]):
        if left <= target <= right:
            return right - left
    raise AssertionError(f"target {target} was not bracketed")


if __name__ == "__main__":
    unittest.main()
