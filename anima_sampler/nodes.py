"""ComfyUI node definitions for the Anima corrective sampler."""

from __future__ import annotations

from .experiment import (
    PARAMETER_MATRIX_KEYS,
    PARAMETER_SWEEP_KEYS,
    build_labeled_comparison_grid,
    build_parameter_combinations,
    parse_sweep_values,
)
from .flow_sampler import (
    CFG_SCHEDULE_MODES,
    FLOW_SCHEDULES,
    FLOW_SOLVERS,
    run_comfy_anima_sampler,
    run_comfy_native_sampler,
)


ANIMA_FLOW_SETTINGS = "ANIMA_FLOW_SETTINGS"

DEFAULT_SCHEDULE_SWEEP_VALUES = (
    "flow_cosmos_lambda_biased_strong, flow_cosmos_rho7_rf_tail_auto"
)
DEFAULT_SOLVER_SWEEP_VALUES = (
    "flow_pc3_damped, flow_pc3_fsal_gated, "
    "flow_3m_sparse_pc3_fsal, flow_3m_damped, flow_heun"
)
NATIVE_ER_SDE_SIMPLE_SAMPLER = "er_sde"
NATIVE_ER_SDE_SIMPLE_SCHEDULER = "simple"

TEST_PROMPT_CASES = {
    "yuruyuri_4girls_dynamic_festival": {
        "positive": (
            "masterpiece, best quality, score_7, safe, very aesthetic, official art, "
            "yuru yuri, yuruyuri, 4girls, akaza akari, toshinou kyouko, funami yui, "
            "yoshikawa chinatsu, nanamori school uniform, school uniform, pleated skirt, "
            "long sleeves, sailor collar, red ribbon, brown cardigan, full body, dynamic angle, "
            "wide shot, dutch angle, foreshortening, complex group pose, synchronized motion, "
            "festival street, rainy night, wet pavement, mirror-like reflections, backlighting, "
            "rim light, high contrast, depth of field, blurry background, bokeh, chromatic aberration, "
            "lens flare, light particles, falling petals, floating ribbons, confetti, motion blur, "
            "huge bouquet, mixed flowers, rose, lily, daisy, baby's-breath, flower petals, leaf, "
            "intricate floral pattern, detailed fabric pattern, embroidered ribbon, lace trim, "
            "Kyouko leaping backward while pulling Akari by the wrist, Akari stumbling forward with "
            "one foot off the ground, Yui catching the oversized bouquet with both hands, Chinatsu "
            "spinning under a long ribbon with her skirt and hair swirling, interlocked arms, crossed legs, "
            "expressive faces, happy, surprised, laughing, wind, ultra-detailed, huge filesize"
        ),
        "negative": (
            "fused fingers, mutated hands, bad hands, extra fingers, missing fingers, fused arms, "
            "extra arms, missing arms, extra legs, missing legs, extra toes, bad feet, bad anatomy, "
            "wrong body count, duplicate character, merged bodies, cropped head, simple background, "
            "nipples, cleavage, nsfw, worst quality, low quality, score_1, score_2, score_3, "
            "lowres, bad, text, error, jpeg artifacts, watermark, unfinished, displeasing, oldest, "
            "early, signature, artist name, username, scan, abstract, english text, shiny hair"
        ),
    },
}

ANIMA_FLOW_BASELINE = {
    "steps": 35,
    "cfg": 6.0,
    "flow_solver": "flow_pc3_damped",
    "flow_er_order": 2,
    "flow_pc3_gamma": 1.0,
    "flow_pc3_tolerance": 0.005,
    "flow_schedule": "flow_cosmos",
    "cosmos_sigma_max": 80.0,
    "cosmos_sigma_min": 0.002,
    "denoise_legacy_progress": False,
    "cfg_legacy_progress": False,
    "cfg_schedule_mode": "beta_bump",
    "early_cfg_boost": 0.5,
    "early_cfg_until": 0.30,
    "late_cfg_scale": 0.92,
    "late_cfg_start": 0.76,
    "cfg_early_scale": 0.98,
    "cfg_early_ramp_end": 0.10,
    "cfg_peak_boost": 0.60,
    "cfg_bump_start": 0.08,
    "cfg_bump_end": 0.68,
    "cfg_beta_alpha": 4.0,
    "cfg_beta_beta": 7.0,
    "cfg_interval_start": 0.12,
    "cfg_interval_rise_end": 0.24,
    "cfg_interval_fall_start": 0.36,
    "cfg_interval_end": 0.58,
    "rf_endpoint_noise_refresh_enabled": False,
    "rf_endpoint_noise_refresh_strength": 0.15,
    "rf_endpoint_noise_refresh_until": 0.20,
}


class AnimaFlowSettings:
    """Reusable Flow sampler settings with the current tested baseline defaults."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": ANIMA_FLOW_BASELINE["steps"], "min": 1, "max": 1000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg"],
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.1,
                    },
                ),
                "flow_solver": (FLOW_SOLVERS, {"default": ANIMA_FLOW_BASELINE["flow_solver"]}),
                "flow_er_order": (
                    "INT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_er_order"],
                        "min": 1,
                        "max": 3,
                    },
                ),
                "flow_pc3_gamma": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_gamma"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                    },
                ),
                "flow_pc3_tolerance": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["flow_pc3_tolerance"],
                        "min": 0.0001,
                        "max": 0.05,
                        "step": 0.0005,
                    },
                ),
                "flow_schedule": (FLOW_SCHEDULES, {"default": ANIMA_FLOW_BASELINE["flow_schedule"]}),
                "cosmos_sigma_max": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_max"],
                        "min": 1.0,
                        "max": 1000.0,
                        "step": 0.5,
                    },
                ),
                "cosmos_sigma_min": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cosmos_sigma_min"],
                        "min": 0.0001,
                        "max": 1.0,
                        "step": 0.0001,
                    },
                ),
                "denoise_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["denoise_legacy_progress"]},
                ),
                "cfg_legacy_progress": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["cfg_legacy_progress"]},
                ),
                "cfg_schedule_mode": (
                    CFG_SCHEDULE_MODES,
                    {"default": ANIMA_FLOW_BASELINE["cfg_schedule_mode"]},
                ),
                "cfg_early_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "cfg_early_ramp_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_early_ramp_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_peak_boost": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_peak_boost"],
                        "min": 0.0,
                        "max": 5.0,
                        "step": 0.05,
                    },
                ),
                "cfg_bump_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_bump_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_bump_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_beta_alpha": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_alpha"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "cfg_beta_beta": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_beta_beta"],
                        "min": 1.0001,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "cfg_interval_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_interval_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_interval_rise_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_interval_rise_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_interval_fall_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_interval_fall_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "cfg_interval_end": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["cfg_interval_end"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "early_cfg_boost": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["early_cfg_boost"],
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.05,
                    },
                ),
                "early_cfg_until": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["early_cfg_until"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "late_cfg_scale": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_scale"],
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                    },
                ),
                "late_cfg_start": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["late_cfg_start"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "rf_endpoint_noise_refresh_enabled": (
                    "BOOLEAN",
                    {"default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_enabled"]},
                ),
                "rf_endpoint_noise_refresh_strength": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_strength"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "rf_endpoint_noise_refresh_until": (
                    "FLOAT",
                    {
                        "default": ANIMA_FLOW_BASELINE["rf_endpoint_noise_refresh_until"],
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = (ANIMA_FLOW_SETTINGS, "STRING")
    RETURN_NAMES = ("settings", "summary")
    FUNCTION = "build"
    CATEGORY = "Anima/Error Corrective Sampling"

    def build(
        self,
        steps,
        cfg,
        flow_solver,
        flow_er_order,
        flow_pc3_gamma,
        flow_pc3_tolerance,
        flow_schedule,
        cosmos_sigma_max,
        cosmos_sigma_min,
        denoise_legacy_progress,
        cfg_legacy_progress,
        cfg_schedule_mode,
        cfg_early_scale,
        cfg_early_ramp_end,
        cfg_peak_boost,
        cfg_bump_start,
        cfg_bump_end,
        cfg_beta_alpha,
        cfg_beta_beta,
        cfg_interval_start,
        cfg_interval_rise_end,
        cfg_interval_fall_start,
        cfg_interval_end,
        early_cfg_boost,
        early_cfg_until,
        late_cfg_scale,
        late_cfg_start,
        rf_endpoint_noise_refresh_enabled,
        rf_endpoint_noise_refresh_strength,
        rf_endpoint_noise_refresh_until,
    ):
        settings = _flow_settings_dict(
            steps=steps,
            cfg=cfg,
            flow_solver=flow_solver,
            flow_er_order=flow_er_order,
            flow_pc3_gamma=flow_pc3_gamma,
            flow_pc3_tolerance=flow_pc3_tolerance,
            flow_schedule=flow_schedule,
            cosmos_sigma_max=cosmos_sigma_max,
            cosmos_sigma_min=cosmos_sigma_min,
            denoise_legacy_progress=denoise_legacy_progress,
            cfg_legacy_progress=cfg_legacy_progress,
            cfg_schedule_mode=cfg_schedule_mode,
            cfg_early_scale=cfg_early_scale,
            cfg_early_ramp_end=cfg_early_ramp_end,
            cfg_peak_boost=cfg_peak_boost,
            cfg_bump_start=cfg_bump_start,
            cfg_bump_end=cfg_bump_end,
            cfg_beta_alpha=cfg_beta_alpha,
            cfg_beta_beta=cfg_beta_beta,
            cfg_interval_start=cfg_interval_start,
            cfg_interval_rise_end=cfg_interval_rise_end,
            cfg_interval_fall_start=cfg_interval_fall_start,
            cfg_interval_end=cfg_interval_end,
            early_cfg_boost=early_cfg_boost,
            early_cfg_until=early_cfg_until,
            late_cfg_scale=late_cfg_scale,
            late_cfg_start=late_cfg_start,
            rf_endpoint_noise_refresh_enabled=rf_endpoint_noise_refresh_enabled,
            rf_endpoint_noise_refresh_strength=rf_endpoint_noise_refresh_strength,
            rf_endpoint_noise_refresh_until=rf_endpoint_noise_refresh_until,
        )
        return settings, _format_settings_summary(settings)


class AnimaFlowCorrectiveSampler:
    """Paper-style Flow Matching sampler with selectable Flow schedules."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "flow_settings": (ANIMA_FLOW_SETTINGS,),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
                "add_noise": ("BOOLEAN", {"default": True}),
                "disable_pbar": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "log")
    FUNCTION = "sample"
    CATEGORY = "Anima/Error Corrective Sampling"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        flow_settings,
        seed,
        denoise,
        add_noise,
        disable_pbar,
    ):
        params = _normalize_settings_object(flow_settings)
        return _run_sampler_with_params(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            params={**params, "seed": seed},
            denoise=denoise,
            add_noise=add_noise,
            disable_pbar=disable_pbar,
        )


def _run_sampler_with_params(
    *,
    model,
    positive,
    negative,
    latent_image,
    params: dict,
    denoise,
    add_noise,
    disable_pbar,
):
    return run_comfy_anima_sampler(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent_image,
        seed=int(params["seed"]),
        steps=int(params["steps"]),
        cfg=float(params["cfg"]),
        denoise=denoise,
        flow_solver=str(params["flow_solver"]),
        flow_er_order=int(params["flow_er_order"]),
        flow_pc3_gamma=float(params["flow_pc3_gamma"]),
        flow_pc3_tolerance=float(params["flow_pc3_tolerance"]),
        flow_schedule=str(params["flow_schedule"]),
        cosmos_sigma_max=float(params["cosmos_sigma_max"]),
        cosmos_sigma_min=float(params["cosmos_sigma_min"]),
        denoise_legacy_progress=bool(params["denoise_legacy_progress"]),
        cfg_schedule_domain=_cfg_domain_from_settings(params),
        cfg_schedule_mode=str(params["cfg_schedule_mode"]),
        early_cfg_boost=float(params["early_cfg_boost"]),
        early_cfg_until=float(params["early_cfg_until"]),
        late_cfg_scale=float(params["late_cfg_scale"]),
        late_cfg_start=float(params["late_cfg_start"]),
        cfg_early_scale=float(params["cfg_early_scale"]),
        cfg_early_ramp_end=float(params["cfg_early_ramp_end"]),
        cfg_peak_boost=float(params["cfg_peak_boost"]),
        cfg_bump_start=float(params["cfg_bump_start"]),
        cfg_bump_end=float(params["cfg_bump_end"]),
        cfg_beta_alpha=float(params["cfg_beta_alpha"]),
        cfg_beta_beta=float(params["cfg_beta_beta"]),
        cfg_interval_start=float(params["cfg_interval_start"]),
        cfg_interval_rise_end=float(params["cfg_interval_rise_end"]),
        cfg_interval_fall_start=float(params["cfg_interval_fall_start"]),
        cfg_interval_end=float(params["cfg_interval_end"]),
        rf_endpoint_noise_refresh_enabled=bool(params["rf_endpoint_noise_refresh_enabled"]),
        rf_endpoint_noise_refresh_strength=float(params["rf_endpoint_noise_refresh_strength"]),
        rf_endpoint_noise_refresh_until=float(params["rf_endpoint_noise_refresh_until"]),
        add_noise=add_noise,
        disable_pbar=disable_pbar,
    )


class AnimaFlowParameterSweep:
    """Run several Flow sampler settings and return one labeled comparison grid."""

    @classmethod
    def INPUT_TYPES(cls):
        flow_inputs = AnimaFlowCorrectiveSampler.INPUT_TYPES()["required"].copy()
        flow_inputs["vae"] = ("VAE",)
        flow_inputs["sweep_parameter"] = (
            PARAMETER_SWEEP_KEYS,
            {"default": "flow_schedule"},
        )
        flow_inputs["sweep_values"] = (
            "STRING",
            {
                "default": DEFAULT_SCHEDULE_SWEEP_VALUES,
                "multiline": True,
            },
        )
        flow_inputs["columns"] = ("INT", {"default": 3, "min": 1, "max": 4})
        flow_inputs["max_runs"] = ("INT", {"default": 2, "min": 1, "max": 12})
        flow_inputs["label_height"] = ("INT", {"default": 96, "min": 48, "max": 192})
        flow_inputs["grid_gap"] = ("INT", {"default": 12, "min": 0, "max": 64})
        flow_inputs["filename_note"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
            },
        )
        return {"required": flow_inputs}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "log")
    FUNCTION = "sweep"
    CATEGORY = "Anima/Error Corrective Sampling"

    def sweep(
        self,
        model,
        positive,
        negative,
        latent_image,
        flow_settings,
        seed,
        denoise,
        add_noise,
        disable_pbar,
        vae,
        sweep_parameter,
        sweep_values,
        columns,
        max_runs,
        label_height,
        grid_gap,
        filename_note,
    ):
        values = parse_sweep_values(sweep_values, sweep_parameter, max_runs=max_runs)
        if not values:
            raise ValueError("sweep_values must contain at least one value")

        base_params = _normalize_settings_object(flow_settings)
        base_params["seed"] = seed

        images = []
        labels = []
        logs = [
            "AnimaFlowParameterSweep",
            f"sweep_parameter: {sweep_parameter}",
            f"runs: {len(values)}",
            f"seed: {base_params['seed']}",
            f"steps: {base_params['steps']}",
            "settings: connected",
            f"denoise: {denoise:.4f}",
            f"note: {filename_note}" if filename_note else "note: <empty>",
        ]

        for run_index, value in enumerate(values):
            params = dict(base_params)
            params[sweep_parameter] = value
            latent, run_log = _run_sampler_with_params(
                model=model,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                params=params,
                denoise=denoise,
                add_noise=add_noise,
                disable_pbar=disable_pbar,
            )
            image = _decode_first_image(vae, latent)
            images.append(image)
            labels.append(_format_sweep_label(run_index + 1, sweep_parameter, value, params))
            logs.append("")
            logs.append(f"run {run_index + 1}: {sweep_parameter}={value}")
            logs.append(run_log)

        grid = build_labeled_comparison_grid(
            images,
            labels,
            columns=columns,
            label_height=label_height,
            gap=grid_gap,
        )
        return grid, "\n".join(logs)


class AnimaFlowMatrixSweep:
    """Run a two-parameter Flow sampler matrix and return one comparison grid."""

    @classmethod
    def INPUT_TYPES(cls):
        flow_inputs = AnimaFlowCorrectiveSampler.INPUT_TYPES()["required"].copy()
        flow_inputs["vae"] = ("VAE",)
        flow_inputs["primary_sweep_parameter"] = (
            PARAMETER_SWEEP_KEYS,
            {"default": "flow_schedule"},
        )
        flow_inputs["primary_sweep_values"] = (
            "STRING",
            {
                "default": DEFAULT_SCHEDULE_SWEEP_VALUES,
                "multiline": True,
            },
        )
        flow_inputs["secondary_sweep_parameter"] = (
            PARAMETER_MATRIX_KEYS,
            {"default": "flow_solver"},
        )
        flow_inputs["secondary_sweep_values"] = (
            "STRING",
            {
                "default": DEFAULT_SOLVER_SWEEP_VALUES,
                "multiline": True,
            },
        )
        flow_inputs["include_comfy_er_sde_simple"] = ("BOOLEAN", {"default": False})
        flow_inputs["columns"] = ("INT", {"default": 5, "min": 1, "max": 8})
        flow_inputs["max_runs"] = ("INT", {"default": 10, "min": 1, "max": 32})
        flow_inputs["label_height"] = ("INT", {"default": 112, "min": 48, "max": 256})
        flow_inputs["grid_gap"] = ("INT", {"default": 12, "min": 0, "max": 64})
        flow_inputs["filename_note"] = (
            "STRING",
            {
                "default": "",
                "multiline": False,
            },
        )
        return {"required": flow_inputs}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison", "log")
    FUNCTION = "sweep"
    CATEGORY = "Anima/Error Corrective Sampling"

    def sweep(
        self,
        model,
        positive,
        negative,
        latent_image,
        flow_settings,
        seed,
        denoise,
        add_noise,
        disable_pbar,
        vae,
        primary_sweep_parameter,
        primary_sweep_values,
        secondary_sweep_parameter,
        secondary_sweep_values,
        include_comfy_er_sde_simple,
        columns,
        max_runs,
        label_height,
        grid_gap,
        filename_note,
    ):
        combinations = build_parameter_combinations(
            primary_sweep_parameter,
            primary_sweep_values,
            secondary_sweep_parameter,
            secondary_sweep_values,
            max_runs=max_runs,
        )
        if not combinations:
            raise ValueError("matrix sweep must contain at least one run")

        base_params = _normalize_settings_object(flow_settings)
        base_params["seed"] = seed

        images = []
        labels = []
        logs = [
            "AnimaFlowMatrixSweep",
            f"primary_sweep_parameter: {primary_sweep_parameter}",
            f"secondary_sweep_parameter: {secondary_sweep_parameter}",
            f"custom_runs: {len(combinations)}",
            f"total_runs: {len(combinations) + (1 if include_comfy_er_sde_simple else 0)}",
            f"include_comfy_er_sde_simple: {bool(include_comfy_er_sde_simple)}",
            f"seed: {base_params['seed']}",
            f"steps: {base_params['steps']}",
            "settings: connected",
            f"denoise: {denoise:.4f}",
            f"note: {filename_note}" if filename_note else "note: <empty>",
        ]

        for run_index, overrides in enumerate(combinations):
            params = dict(base_params)
            params.update(overrides)
            latent, run_log = _run_sampler_with_params(
                model=model,
                positive=positive,
                negative=negative,
                latent_image=latent_image,
                params=params,
                denoise=denoise,
                add_noise=add_noise,
                disable_pbar=disable_pbar,
            )
            image = _decode_first_image(vae, latent)
            images.append(image)
            labels.append(_format_matrix_label(run_index + 1, overrides, params))
            logs.append("")
            logs.append(f"run {run_index + 1}: {_format_override_summary(overrides)}")
            logs.append(run_log)

        if include_comfy_er_sde_simple:
            native_index = len(images) + 1
            latent, run_log = run_comfy_native_sampler(
                model=model,
                positive=positive,
                negative=negative,
                latent=latent_image,
                seed=int(base_params["seed"]),
                steps=int(base_params["steps"]),
                cfg=float(base_params["cfg"]),
                denoise=denoise,
                sampler_name=NATIVE_ER_SDE_SIMPLE_SAMPLER,
                scheduler=NATIVE_ER_SDE_SIMPLE_SCHEDULER,
                add_noise=add_noise,
                disable_pbar=disable_pbar,
            )
            image = _decode_first_image(vae, latent)
            images.append(image)
            labels.append(_format_native_comparison_label(native_index, base_params))
            logs.append("")
            logs.append(
                "run "
                f"{native_index}: comfy_native_sampler={NATIVE_ER_SDE_SIMPLE_SAMPLER}, "
                f"scheduler={NATIVE_ER_SDE_SIMPLE_SCHEDULER}"
            )
            logs.append(run_log)

        grid = build_labeled_comparison_grid(
            images,
            labels,
            columns=columns,
            label_height=label_height,
            gap=grid_gap,
        )
        return grid, "\n".join(logs)


class AnimaFlowTestPrompt:
    """Reusable text prompts for matched sampler comparison tests."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_case": (
                    list(TEST_PROMPT_CASES),
                    {"default": "yuruyuri_4girls_dynamic_festival"},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_text", "negative_text")
    FUNCTION = "build"
    CATEGORY = "Anima/Error Corrective Sampling"

    def build(self, prompt_case):
        prompt = TEST_PROMPT_CASES[str(prompt_case)]
        return prompt["positive"], prompt["negative"]


NODE_CLASS_MAPPINGS = {
    "AnimaFlowSettings": AnimaFlowSettings,
    "AnimaFlowCorrectiveSampler": AnimaFlowCorrectiveSampler,
    "AnimaFlowParameterSweep": AnimaFlowParameterSweep,
    "AnimaFlowMatrixSweep": AnimaFlowMatrixSweep,
    "AnimaFlowTestPrompt": AnimaFlowTestPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaFlowSettings": "Anima Flow Settings",
    "AnimaFlowCorrectiveSampler": "Anima Flow Corrective Sampler",
    "AnimaFlowParameterSweep": "Anima Flow Parameter Sweep",
    "AnimaFlowMatrixSweep": "Anima Flow Matrix Sweep",
    "AnimaFlowTestPrompt": "Anima Flow Test Prompt",
}


def _decode_first_image(vae, latent):
    images = vae.decode(latent["samples"])
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images[:1].detach().cpu()


def _format_sweep_label(run_index: int, sweep_parameter: str, value, params: dict) -> str:
    return (
        f"{run_index}. {sweep_parameter}={value}\n"
        f"seed={int(params['seed'])}, steps={int(params['steps'])}, "
        f"cfg={float(params['cfg']):.2f}\n"
        f"{params['flow_solver']}, order={int(params['flow_er_order'])}, "
        f"pc3={float(params['flow_pc3_gamma']):.2f}/{float(params['flow_pc3_tolerance']):.4f}, "
        f"schedule={params['flow_schedule']}, "
        f"cfg_mode={params['cfg_schedule_mode']}, "
        f"legacy_cfg_progress={'on' if params['cfg_legacy_progress'] else 'off'}, "
        f"legacy_denoise_progress={'on' if params['denoise_legacy_progress'] else 'off'}\n"
        f"cosmos={float(params['cosmos_sigma_max']):.1f}/{float(params['cosmos_sigma_min']):.4f}, "
        f"peak_cfg={float(params['cfg_peak_boost']):.2f}, "
        f"refresh={'on' if params['rf_endpoint_noise_refresh_enabled'] else 'off'}:"
        f"{float(params['rf_endpoint_noise_refresh_strength']):.2f}/"
        f"{float(params['rf_endpoint_noise_refresh_until']):.2f}"
    )


def _format_matrix_label(run_index: int, overrides: dict, params: dict) -> str:
    return (
        f"{run_index}. {_format_override_summary(overrides)}\n"
        f"seed={int(params['seed'])}, steps={int(params['steps'])}, "
        f"calls={_estimated_model_calls(params)}, cfg={float(params['cfg']):.2f}\n"
        f"{params['flow_solver']}, order={int(params['flow_er_order'])}, "
        f"pc3={float(params['flow_pc3_gamma']):.2f}/{float(params['flow_pc3_tolerance']):.4f}\n"
        f"schedule={params['flow_schedule']}, "
        f"cfg_mode={params['cfg_schedule_mode']}, "
        f"cfg_domain={_cfg_domain_from_settings(params)}, "
        f"denoise_domain={_denoise_domain_from_settings(params)}\n"
        f"cosmos={float(params['cosmos_sigma_max']):.1f}/{float(params['cosmos_sigma_min']):.4f}, "
        f"refresh={'on' if params['rf_endpoint_noise_refresh_enabled'] else 'off'}:"
        f"{float(params['rf_endpoint_noise_refresh_strength']):.2f}/"
        f"{float(params['rf_endpoint_noise_refresh_until']):.2f}"
    )


def _format_native_comparison_label(run_index: int, params: dict) -> str:
    return (
        f"{run_index}. ComfyUI native {NATIVE_ER_SDE_SIMPLE_SAMPLER} + "
        f"{NATIVE_ER_SDE_SIMPLE_SCHEDULER}\n"
        f"seed={int(params['seed'])}, steps={int(params['steps'])}, "
        f"calls=native, cfg={float(params['cfg']):.2f}\n"
        "stock ComfyUI KSampler path\n"
        "scheduler=simple, sampler=er_sde\n"
        "RF custom controls bypassed"
    )


def _format_override_summary(overrides: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in overrides.items())


def _flow_settings_dict(
    *,
    steps,
    cfg,
    flow_solver,
    flow_er_order,
    flow_pc3_gamma,
    flow_pc3_tolerance,
    flow_schedule,
    cosmos_sigma_max,
    cosmos_sigma_min,
    denoise_legacy_progress,
    cfg_legacy_progress,
    cfg_schedule_mode,
    cfg_early_scale,
    cfg_early_ramp_end,
    cfg_peak_boost,
    cfg_bump_start,
    cfg_bump_end,
    cfg_beta_alpha,
    cfg_beta_beta,
    cfg_interval_start,
    cfg_interval_rise_end,
    cfg_interval_fall_start,
    cfg_interval_end,
    early_cfg_boost,
    early_cfg_until,
    late_cfg_scale,
    late_cfg_start,
    rf_endpoint_noise_refresh_enabled,
    rf_endpoint_noise_refresh_strength,
    rf_endpoint_noise_refresh_until,
) -> dict:
    return _normalize_flow_params(
        {
            "steps": steps,
            "cfg": cfg,
            "flow_solver": flow_solver,
            "flow_er_order": flow_er_order,
            "flow_pc3_gamma": flow_pc3_gamma,
            "flow_pc3_tolerance": flow_pc3_tolerance,
            "flow_schedule": flow_schedule,
            "cosmos_sigma_max": cosmos_sigma_max,
            "cosmos_sigma_min": cosmos_sigma_min,
            "denoise_legacy_progress": denoise_legacy_progress,
            "cfg_legacy_progress": cfg_legacy_progress,
            "cfg_schedule_mode": cfg_schedule_mode,
            "cfg_early_scale": cfg_early_scale,
            "cfg_early_ramp_end": cfg_early_ramp_end,
            "cfg_peak_boost": cfg_peak_boost,
            "cfg_bump_start": cfg_bump_start,
            "cfg_bump_end": cfg_bump_end,
            "cfg_beta_alpha": cfg_beta_alpha,
            "cfg_beta_beta": cfg_beta_beta,
            "cfg_interval_start": cfg_interval_start,
            "cfg_interval_rise_end": cfg_interval_rise_end,
            "cfg_interval_fall_start": cfg_interval_fall_start,
            "cfg_interval_end": cfg_interval_end,
            "early_cfg_boost": early_cfg_boost,
            "early_cfg_until": early_cfg_until,
            "late_cfg_scale": late_cfg_scale,
            "late_cfg_start": late_cfg_start,
            "rf_endpoint_noise_refresh_enabled": rf_endpoint_noise_refresh_enabled,
            "rf_endpoint_noise_refresh_strength": rf_endpoint_noise_refresh_strength,
            "rf_endpoint_noise_refresh_until": rf_endpoint_noise_refresh_until,
        }
    )


def _normalize_settings_object(flow_settings) -> dict:
    if not isinstance(flow_settings, dict):
        raise ValueError("flow_settings must be an Anima Flow Settings object")

    out = dict(ANIMA_FLOW_BASELINE)
    for key, value in flow_settings.items():
        if key in ANIMA_FLOW_BASELINE:
            out[key] = value
    return _normalize_flow_params(out)


def _normalize_flow_params(params: dict) -> dict:
    out = dict(params)
    out["steps"] = int(out["steps"])
    out["cfg"] = float(out["cfg"])
    out["flow_solver"] = str(out["flow_solver"])
    out["flow_er_order"] = int(out["flow_er_order"])
    out["flow_pc3_gamma"] = float(out["flow_pc3_gamma"])
    out["flow_pc3_tolerance"] = float(out["flow_pc3_tolerance"])
    out["flow_schedule"] = str(out["flow_schedule"])
    out["cosmos_sigma_max"] = float(out["cosmos_sigma_max"])
    out["cosmos_sigma_min"] = float(out["cosmos_sigma_min"])
    out["denoise_legacy_progress"] = _as_bool(out["denoise_legacy_progress"])
    out["cfg_legacy_progress"] = _as_bool(out["cfg_legacy_progress"])
    out["cfg_schedule_mode"] = str(out["cfg_schedule_mode"])
    out["cfg_early_scale"] = float(out["cfg_early_scale"])
    out["cfg_early_ramp_end"] = float(out["cfg_early_ramp_end"])
    out["cfg_peak_boost"] = float(out["cfg_peak_boost"])
    out["cfg_bump_start"] = float(out["cfg_bump_start"])
    out["cfg_bump_end"] = float(out["cfg_bump_end"])
    out["cfg_beta_alpha"] = float(out["cfg_beta_alpha"])
    out["cfg_beta_beta"] = float(out["cfg_beta_beta"])
    out["cfg_interval_start"] = float(out["cfg_interval_start"])
    out["cfg_interval_rise_end"] = float(out["cfg_interval_rise_end"])
    out["cfg_interval_fall_start"] = float(out["cfg_interval_fall_start"])
    out["cfg_interval_end"] = float(out["cfg_interval_end"])
    out["early_cfg_boost"] = float(out["early_cfg_boost"])
    out["early_cfg_until"] = float(out["early_cfg_until"])
    out["late_cfg_scale"] = float(out["late_cfg_scale"])
    out["late_cfg_start"] = float(out["late_cfg_start"])
    out["rf_endpoint_noise_refresh_enabled"] = _as_bool(
        out["rf_endpoint_noise_refresh_enabled"]
    )
    out["rf_endpoint_noise_refresh_strength"] = float(
        out["rf_endpoint_noise_refresh_strength"]
    )
    out["rf_endpoint_noise_refresh_until"] = float(out["rf_endpoint_noise_refresh_until"])

    if out["flow_solver"] not in FLOW_SOLVERS:
        raise ValueError(f"unsupported flow_solver: {out['flow_solver']}")
    if not (1 <= out["flow_er_order"] <= 3):
        raise ValueError("flow_er_order must be in the range [1, 3]")
    if not (0.0 <= out["flow_pc3_gamma"] <= 1.0):
        raise ValueError("flow_pc3_gamma must be in the range [0, 1]")
    if not (0.0 < out["flow_pc3_tolerance"] <= 1.0):
        raise ValueError("flow_pc3_tolerance must be in the range (0, 1]")
    if out["flow_schedule"] not in FLOW_SCHEDULES:
        raise ValueError(f"unsupported flow_schedule: {out['flow_schedule']}")
    if not (0.0 < out["cosmos_sigma_min"] < out["cosmos_sigma_max"]):
        raise ValueError("expected 0 < cosmos_sigma_min < cosmos_sigma_max")
    if out["cfg_schedule_mode"] not in CFG_SCHEDULE_MODES:
        allowed = ", ".join(CFG_SCHEDULE_MODES)
        raise ValueError(f"cfg_schedule_mode must be one of: {allowed}")
    if not (0.0 <= out["cfg_early_scale"] <= 2.0):
        raise ValueError("cfg_early_scale must be in the range [0, 2]")
    if not (0.0 <= out["cfg_early_ramp_end"] <= 1.0):
        raise ValueError("cfg_early_ramp_end must be in the range [0, 1]")
    if out["cfg_peak_boost"] < 0.0:
        raise ValueError("cfg_peak_boost must be non-negative")
    if not (0.0 <= out["cfg_bump_start"] < out["cfg_bump_end"] <= 1.0):
        raise ValueError("expected 0 <= cfg_bump_start < cfg_bump_end <= 1")
    if out["cfg_beta_alpha"] <= 1.0 or out["cfg_beta_beta"] <= 1.0:
        raise ValueError("cfg_beta_alpha and cfg_beta_beta must be > 1")
    if not (
        0.0
        <= out["cfg_interval_start"]
        <= out["cfg_interval_rise_end"]
        <= out["cfg_interval_fall_start"]
        <= out["cfg_interval_end"]
        <= 1.0
    ):
        raise ValueError(
            "expected cfg_interval_start <= cfg_interval_rise_end <= "
            "cfg_interval_fall_start <= cfg_interval_end within [0, 1]"
        )
    if not (0.0 <= out["rf_endpoint_noise_refresh_strength"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_strength must be in the range [0, 1]")
    if not (0.0 <= out["rf_endpoint_noise_refresh_until"] <= 1.0):
        raise ValueError("rf_endpoint_noise_refresh_until must be in the range [0, 1]")
    return out


def _format_settings_summary(settings: dict) -> str:
    return "\n".join(
        [
            "AnimaFlowSettings",
            f"steps: {settings['steps']}",
            "steps_semantics: RF integration intervals, not diffusion denoise steps",
            f"estimated_model_calls: {_estimated_model_calls(settings)}",
            f"cfg: {settings['cfg']:.2f}",
            f"flow_solver: {settings['flow_solver']}",
            f"flow_er_order: {settings['flow_er_order']}",
            f"flow_pc3_gamma: {settings['flow_pc3_gamma']:.4f}",
            f"flow_pc3_tolerance: {settings['flow_pc3_tolerance']:.6f}",
            f"flow_schedule: {settings['flow_schedule']}",
            f"cosmos_sigma_max: {settings['cosmos_sigma_max']:.4f}",
            f"cosmos_sigma_min: {settings['cosmos_sigma_min']:.6f}",
            f"denoise_legacy_progress: {settings['denoise_legacy_progress']}",
            f"denoise_schedule_domain: {_denoise_domain_from_settings(settings)}",
            f"cfg_legacy_progress: {settings['cfg_legacy_progress']}",
            f"cfg_schedule_mode: {settings['cfg_schedule_mode']}",
            f"cfg_schedule_domain: {_cfg_domain_from_settings(settings)}",
            f"cfg_early_scale: {settings['cfg_early_scale']:.4f}",
            f"cfg_early_ramp_end: {settings['cfg_early_ramp_end']:.4f}",
            f"cfg_peak_boost: {settings['cfg_peak_boost']:.4f}",
            f"cfg_bump_start: {settings['cfg_bump_start']:.4f}",
            f"cfg_bump_end: {settings['cfg_bump_end']:.4f}",
            f"cfg_beta_alpha: {settings['cfg_beta_alpha']:.4f}",
            f"cfg_beta_beta: {settings['cfg_beta_beta']:.4f}",
            f"cfg_interval_start: {settings['cfg_interval_start']:.4f}",
            f"cfg_interval_rise_end: {settings['cfg_interval_rise_end']:.4f}",
            f"cfg_interval_fall_start: {settings['cfg_interval_fall_start']:.4f}",
            f"cfg_interval_end: {settings['cfg_interval_end']:.4f}",
            f"early_cfg_boost: {settings['early_cfg_boost']:.4f}",
            f"early_cfg_until: {settings['early_cfg_until']:.4f}",
            f"late_cfg_scale: {settings['late_cfg_scale']:.4f}",
            f"late_cfg_start: {settings['late_cfg_start']:.4f}",
            (
                "rf_endpoint_noise_refresh_enabled: "
                f"{settings['rf_endpoint_noise_refresh_enabled']}"
            ),
            (
                "rf_endpoint_noise_refresh_strength: "
                f"{settings['rf_endpoint_noise_refresh_strength']:.4f}"
            ),
            (
                "rf_endpoint_noise_refresh_until: "
                f"{settings['rf_endpoint_noise_refresh_until']:.4f}"
            ),
        ]
    )


def _cfg_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["cfg_legacy_progress"]) else "lambda"


def _denoise_domain_from_settings(settings: dict) -> str:
    return "progress" if bool(settings["denoise_legacy_progress"]) else "lambda"


def _estimated_model_calls(settings: dict) -> int:
    steps = max(1, int(settings["steps"]))
    if settings["flow_solver"] in {
        "flow_heun",
        "flow_pc3_damped",
        "flow_pc3_fsal_gated",
    }:
        return max(1, steps * 2 - 1)
    if settings["flow_solver"] == "flow_3m_sparse_pc3_fsal":
        sparse_budget = min(10, max(5, round(0.23 * steps)))
        return max(1, steps + sparse_budget)
    return steps


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected boolean value, got: {value!r}")
