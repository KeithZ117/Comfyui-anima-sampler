# Experiment Feedback

This folder stores user-visible generation feedback and sampler experiment
notes. Treat these records as project data, not casual chat history.

Use dated Markdown files for each investigation or feedback batch. Prefer this
shape:

- Observation: what changed in generated images.
- Settings: solver, schedule, steps, CFG, seed, prompt, stochastic settings,
  kick settings, and model context when available.
- Interpretation: the current hypothesis, clearly separated from observation.
- Action: code/config changes already made or proposed.
- Follow-up: comparisons that should be rerun on matched seeds/prompts.

If prompt, seed, or exact settings were not provided, write `not provided`.
