# Notebooks

Use this directory for notebook workflows (Colab and local).

- Main Colab workflow: `notebooks/main.ipynb`
- Keep training/experimentation notebooks here.
- Prefer importing reusable logic from `src/` and `scripts/`.
- Notebook training is now aligned with `src/training/trainer.py`.
- Notebook sync steps write to `outputs/` (canonical) instead of the legacy `output/` folder.
- Notebook training cell builds `args` via `trainer.parse_args([...])` using `config/baseline.yaml` plus a small override list.
