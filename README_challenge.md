# Competition track of the GRaM workshop

[![](https://img.shields.io/badge/Website-GRaM_workshop-white)](https://gram-workshop.github.io)
[![](https://img.shields.io/badge/Website-GRaM_competition-teal)](https://gram-competition.github.io)
[![](https://img.shields.io/badge/Hugging_Face-Dataset-yellow)](https://huggingface.co/datasets/gram-competition/warped-ifw)

<img src=".logos/beyondmath.svg" height="60vw">&nbsp;&nbsp;&nbsp;<img src=".logos/mcml.svg" height="60vw">

This repository functions as submission portal for the competition hosted in conjunction with the Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM) at ICLR 2026.
For description of the challenge refer to the competition website (link above).

Deadline is on **April 22, 2026 (AoE)**.

## Submission guidelines

In order to participate in the competition, your team has to create a valid submission in the form of a **pull request** to this repository. The requirements for a valid submission are listed in the following.
You can mimic our implementation of a basic MLP for reference.

Create a **class implementation** of your model (fully contained) in the directory `models/<model name>/` that can be instantiated without arguments, i.e.,
```python
model = ModelName()
```
and is callable (e.g., via `model.forward`) by the signature
```python
def __call__(
    t: torch.Tensor,
    pos: torch.Tensor,
    idcs_airfoil: list[torch.Tensor],
    velocity_in: torch.Tensor
) -> torch.Tensor:
    ...
    return velocity_out
```
with tensor dimensions
```
t: (batch size, 10)
pos: (batch size, 100k, 3)
velocity_in: (batch size, 5, 100k, 3)
velocity_out: (batch size, 5, 100k, 3)
```
where elements of the list `idcs_airfoil` are variable-length tensors indexing `pos`, i.e., take values in `[0, 100k)`.

Feel free to use a different backend than PyTorch (JAX, etc.) but please match the typing in the signature above.
It is fine if your model depends on external libraries (xFormers, your own, etc.) as long as they are easy to install.

Provide **model weights** along with your pull request, either by uploading to the directory `models/<model name>/` or via download link if their file size would be too large.
Your model must load the weights during construction.

Create an **import entry** in `models/__init__.py` that imports you model's constructor from `models/<model name>/`.

*Optional:* provide a Markdown file under `models/<model name>/` detailing your training process and other important information to reproduce your approach.

## Submission policy
Each team may create one submission.
We will monitor submissions on a rolling basis and notify participants once their submission is valid.

## Local Development Structure

To keep competition constraints intact while improving day-to-day research workflow, this repository now uses an incremental modular layout:

```
config/
    baseline.yaml
src/
    data/
    training/
    evaluation/
    utils/
scripts/
    train.py
    evaluate.py
notebooks/
    README.md
    main.ipynb
outputs/
    runs/
```

`models/mlp/` remains the competition-safe baseline model package.
The refactor currently externalizes data and training-loop helper logic into `src/` while preserving the existing training behavior.
The notebook workflow was moved from the repository root to `notebooks/main.ipynb`.
An additional research model now lives in `models/spatiotemporal_mno/`; the submission-facing constructor pattern remains the same.

## Training Commands

Existing command (still supported):

```bash
python main.py --epochs 100
```

`main.py` is now a compatibility wrapper that forwards to the modular training entrypoint in `src/training/trainer.py`.

New script entrypoint:

```bash
python scripts/train.py --epochs 100
```

YAML config + CLI override workflow:

```bash
python scripts/train.py --config config/baseline.yaml
python scripts/train.py --config config/baseline.yaml --epochs 200 --lr 5e-4
python scripts/train.py --config config/spatiotemporal_mno.yaml
python scripts/train.py --model-name spatiotemporal_mno --config config/spatiotemporal_mno.yaml
```

By default, each run now also writes artifacts into `outputs/runs/<run_name>/`, including:

- `config.json`
- `checkpoints/best.pt`
- history CSV/JSON and loss plot

Legacy root-level `output/` is retired; use `outputs/` as the single artifact root.

Standalone evaluation command:

```bash
python scripts/evaluate.py --config config/baseline.yaml --checkpoint-path models/mlp/state_dict.pt --split test
python scripts/evaluate.py --config config/baseline.yaml --checkpoint-path outputs/runs/<run_name>/checkpoints/best.pt --split test
python scripts/evaluate.py --config config/spatiotemporal_mno.yaml --model-name spatiotemporal_mno --checkpoint-path outputs/runs/<run_name>/checkpoints/best.pt --split test
```

If YAML loading is used, install dependency:

```bash
pip install pyyaml
```
