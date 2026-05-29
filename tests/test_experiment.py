import unittest

import torch

from anima_sampler.experiment import (
    NO_SECONDARY_SWEEP,
    PARAMETER_MATRIX_KEYS,
    PARAMETER_SWEEP_KEYS,
    build_labeled_comparison_grid,
    build_parameter_combinations,
    parse_sweep_values,
)
from anima_sampler.flow_sampler import CFG_SCHEDULE_MODES


class ExperimentHelperTests(unittest.TestCase):
    def test_parse_float_sweep_values(self):
        self.assertEqual(
            parse_sweep_values("2.0, 3.0\n4.0", "cfg", max_runs=8),
            [2.0, 3.0, 4.0],
        )

    def test_cosmos_sigma_values_are_sweepable(self):
        self.assertIn("cosmos_sigma_max", PARAMETER_SWEEP_KEYS)
        self.assertIn("cosmos_sigma_min", PARAMETER_SWEEP_KEYS)
        self.assertIn("flow_shift", PARAMETER_SWEEP_KEYS)
        self.assertEqual(parse_sweep_values("40, 80", "cosmos_sigma_max", max_runs=8), [40.0, 80.0])
        self.assertEqual(parse_sweep_values("2, 3, 5", "flow_shift", max_runs=8), [2.0, 3.0, 5.0])

    def test_cfg_legacy_progress_is_sweepable_bool(self):
        self.assertIn("cfg_legacy_progress", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values("true, false, on, off", "cfg_legacy_progress", max_runs=8),
            [True, False, True, False],
        )

    def test_denoise_legacy_progress_is_sweepable_bool(self):
        self.assertIn("denoise_legacy_progress", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values("true, false", "denoise_legacy_progress", max_runs=8),
            [True, False],
        )

    def test_final_clean_pass_is_sweepable_bool(self):
        self.assertIn("final_clean_pass", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values("true, false", "final_clean_pass", max_runs=8),
            [True, False],
        )

    def test_seed_and_steps_are_sweepable_integer_values(self):
        self.assertIn("seed", PARAMETER_SWEEP_KEYS)
        self.assertIn("steps", PARAMETER_SWEEP_KEYS)
        self.assertIn("flow_er_order", PARAMETER_SWEEP_KEYS)
        self.assertIn("flow_pc3_gamma", PARAMETER_SWEEP_KEYS)
        self.assertIn("flow_pc3_tolerance", PARAMETER_SWEEP_KEYS)
        self.assertEqual(parse_sweep_values("67, 68, 69", "seed", max_runs=8), [67, 68, 69])
        self.assertEqual(parse_sweep_values("25, 30, 35, 40", "steps", max_runs=8), [25, 30, 35, 40])
        self.assertEqual(parse_sweep_values("1, 2, 3", "flow_er_order", max_runs=8), [1, 2, 3])
        self.assertEqual(parse_sweep_values("0.25, 0.5, 1.0", "flow_pc3_gamma", max_runs=8), [0.25, 0.5, 1.0])
        self.assertEqual(
            parse_sweep_values("0.002, 0.005, 0.01", "flow_pc3_tolerance", max_runs=8),
            [0.002, 0.005, 0.01],
        )

    def test_integer_sweep_rejects_fractional_values(self):
        with self.assertRaises(ValueError):
            parse_sweep_values("1.5", "steps", max_runs=8)

    def test_parse_sweep_values_respects_max_runs(self):
        self.assertEqual(
            parse_sweep_values("1 2 3 4", "cfg", max_runs=2),
            [1.0, 2.0],
        )

    def test_late_cfg_start_is_sweepable(self):
        self.assertIn("late_cfg_start", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values("0.65, 0.75, 0.85", "late_cfg_start", max_runs=8),
            [0.65, 0.75, 0.85],
        )

    def test_cfg_schedule_mode_is_sweepable_enum(self):
        self.assertIn("cfg_schedule_mode", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values(
                "beta_bump, low_to_high, limited_interval, legacy_boost, constant",
                "cfg_schedule_mode",
                max_runs=8,
            ),
            CFG_SCHEDULE_MODES,
        )

    def test_dynamic_cfg_curve_parameters_are_sweepable(self):
        for parameter in (
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
        ):
            self.assertIn(parameter, PARAMETER_SWEEP_KEYS)
        self.assertEqual(parse_sweep_values("0.4, 0.6", "cfg_peak_boost", max_runs=8), [0.4, 0.6])

    def test_rf_endpoint_noise_refresh_enabled_is_sweepable_bool(self):
        self.assertIn("rf_endpoint_noise_refresh_enabled", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values("on, off", "rf_endpoint_noise_refresh_enabled", max_runs=8),
            [True, False],
        )

    def test_rf_endpoint_noise_refresh_strength_and_until_are_sweepable(self):
        self.assertIn("rf_endpoint_noise_refresh_strength", PARAMETER_SWEEP_KEYS)
        self.assertIn("rf_endpoint_noise_refresh_until", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values(
                "0.05, 0.15, 0.25",
                "rf_endpoint_noise_refresh_strength",
                max_runs=8,
            ),
            [0.05, 0.15, 0.25],
        )

    def test_flow_solver_is_sweepable_enum(self):
        self.assertIn("flow_solver", PARAMETER_SWEEP_KEYS)
        self.assertEqual(
            parse_sweep_values(
                (
                    "flow_euler, flow_ab2, flow_heun, flow_pc3_damped, "
                    "flow_3m_damped, flow_unipc2_x0, flow_er"
                ),
                "flow_solver",
                max_runs=12,
            ),
            [
                "flow_euler",
                "flow_ab2",
                "flow_heun",
                "flow_pc3_damped",
                "flow_3m_damped",
                "flow_unipc2_x0",
                "flow_er",
            ],
        )

    def test_flow_solver_rejects_removed_values(self):
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_rho7_euler", "flow_solver", max_runs=8)

    def test_flow_schedule_accepts_current_schedule_enum(self):
        self.assertEqual(
            parse_sweep_values(
                (
                    "flow_cosmos, flow_cosmos_rf_tail, flow_cosmos_lambda_biased_strong, "
                    "flow_cosmos_rho7, flow_rf_linear_shift, "
                    "flow_rf_linear_s_tail_shift5, simple"
                ),
                "flow_schedule",
                max_runs=12,
            ),
            [
                "flow_cosmos",
                "flow_cosmos_rf_tail",
                "flow_cosmos_lambda_biased_strong",
                "flow_cosmos_rho7",
                "flow_rf_linear_shift",
                "flow_rf_linear_s_tail_shift5",
                "simple",
            ],
        )

    def test_flow_schedule_rejects_removed_values(self):
        with self.assertRaises(ValueError):
            parse_sweep_values("flowmatch_euler", "flow_schedule", max_runs=8)
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_cosmos_rho7_rf_tail_auto", "flow_schedule", max_runs=8)
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_cosmos_rho7_rf_tail_balanced", "flow_schedule", max_runs=8)
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_cosmos_beta5", "flow_schedule", max_runs=8)
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_cosmos_shift5_rf_tail_auto", "flow_schedule", max_runs=8)

    def test_enum_sweep_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            parse_sweep_values("flow_bad", "flow_solver", max_runs=8)

    def test_build_parameter_combinations_builds_primary_only_sweep(self):
        self.assertIn(NO_SECONDARY_SWEEP, PARAMETER_MATRIX_KEYS)
        self.assertEqual(
            build_parameter_combinations(
                "steps",
                "30, 35",
                NO_SECONDARY_SWEEP,
                max_runs=8,
            ),
            [{"steps": 30}, {"steps": 35}],
        )

    def test_build_parameter_combinations_builds_matrix_in_row_major_order(self):
        self.assertEqual(
            build_parameter_combinations(
                "flow_schedule",
                "flow_cosmos, flow_cosmos_lambda_biased_strong",
                "flow_solver",
                "flow_er, flow_heun",
                max_runs=8,
            ),
            [
                {"flow_schedule": "flow_cosmos", "flow_solver": "flow_er"},
                {"flow_schedule": "flow_cosmos_lambda_biased_strong", "flow_solver": "flow_er"},
                {"flow_schedule": "flow_cosmos", "flow_solver": "flow_heun"},
                {
                    "flow_schedule": "flow_cosmos_lambda_biased_strong",
                    "flow_solver": "flow_heun",
                },
            ],
        )

    def test_build_parameter_combinations_allows_rho7_schedule_with_tail_auto_sweep(self):
        self.assertEqual(
            build_parameter_combinations(
                "flow_schedule",
                "flow_cosmos_rho7",
                "flow_rho7_tail_auto",
                "false, true",
                max_runs=8,
            ),
            [
                {"flow_schedule": "flow_cosmos_rho7", "flow_rho7_tail_auto": False},
                {"flow_schedule": "flow_cosmos_rho7", "flow_rho7_tail_auto": True},
            ],
        )

    def test_build_parameter_combinations_respects_max_runs(self):
        self.assertEqual(
            build_parameter_combinations(
                "steps",
                "20, 30, 40",
                "flow_solver",
                "flow_er, flow_heun",
                max_runs=4,
            ),
            [
                {"steps": 20, "flow_solver": "flow_er"},
                {"steps": 30, "flow_solver": "flow_er"},
                {"steps": 40, "flow_solver": "flow_er"},
                {"steps": 20, "flow_solver": "flow_heun"},
            ],
        )

    def test_build_parameter_combinations_rejects_duplicate_parameters(self):
        with self.assertRaises(ValueError):
            build_parameter_combinations(
                "steps",
                "30",
                "steps",
                "35",
                max_runs=8,
            )

    def test_build_labeled_grid_returns_single_image_tensor(self):
        images = [
            torch.zeros(1, 16, 16, 3),
            torch.ones(1, 16, 16, 3),
        ]

        grid = build_labeled_comparison_grid(
            images,
            ["a=1", "a=2"],
            columns=2,
            label_height=48,
            gap=4,
        )

        self.assertEqual(tuple(grid.shape), (1, 64, 36, 3))
        self.assertGreaterEqual(float(grid.min()), 0.0)
        self.assertLessEqual(float(grid.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
