# Spatiotemporal MNO

This model is an alternative to the baseline `models/mlp/` architecture.

## Idea

The baseline flattens the full 5-frame velocity history into one per-point feature vector and lets the MNO stack infer temporal structure implicitly.

`SpatiotemporalMNO` makes that factorisation explicit:

1. Encode each observed frame separately with a shared spatial encoder.
2. Run shared MNO blocks on each frame to build geometry-aware spatial latents.
3. Forecast future latent trajectories with a temporal sequence model.
4. Refine each future latent field with MNO blocks and decode to velocity residuals.

## Why this matches the challenge

- The airfoil geometry is fixed across the 10 timestamps, so spatial encoding can be shared across all observed frames.
- The next 5 frames depend on temporal evolution, not just the current spatial field, so an explicit latent temporal model is useful.
- The output is still a per-point velocity field, so the decoder keeps point alignment and the no-slip boundary constraint from the baseline.

## Implementation details

File: `models/spatiotemporal_mno/model.py`

- Inputs per observed frame:
  - `pos`
  - the current velocity frame only
  - Fourier embedding of the current timestamp
  - airfoil mask
  - wall distance
  - local surface frame
- Shared spatial backbone:
  - frame encoder MLP
  - MNO blocks over the point cloud
- Temporal forecaster:
  - GRU over the sequence of per-point latent states
  - direct seq2seq latent prediction for all future steps at once
- Future decoder:
  - output-time conditioning
  - MNO refinement blocks on each forecast latent field
  - pointwise decoder MLP to residual velocity
- Physics prior:
  - persistence baseline from the last observed frame
  - hard no-slip enforcement on airfoil points in scaled space

## Training workflow

Train with the shared CLI:

```bash
python scripts/train.py --config config/spatiotemporal_mno.yaml
```

Or override from the command line:

```bash
python scripts/train.py \
  --model-name spatiotemporal_mno \
  --config config/spatiotemporal_mno.yaml \
  --dataset-dir dataset_huggingface/warped-ifw
```

Evaluate:

```bash
python scripts/evaluate.py \
  --config config/spatiotemporal_mno.yaml \
  --model-name spatiotemporal_mno \
  --checkpoint-path outputs/runs/<run_name>/checkpoints/best.pt \
  --split test
```

## Practical notes

- This model keeps the same training loop, loss functions, dataset pipeline, and checkpoint format as the baseline.
- It is heavier than the baseline because it applies MNO blocks before and after temporal forecasting.
- The temporal module is direct multi-step forecasting, not autoregressive rollout, to limit compounding error over the 5 predicted steps.
