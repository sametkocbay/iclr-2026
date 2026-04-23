# W&B Logging On The Cluster

Training jobs write W&B local files to scratch by default:

```bash
/dss/mcmlscratch/08/di97hih/outputs/wandb
```

The SLURM scripts use:

```bash
ENABLE_WANDB=1
WANDB_PROJECT=iclr_develop
WANDB_RUN_NAME="${RUN_NAME}"
WANDB_DIR=/dss/mcmlscratch/08/di97hih/outputs/wandb
WANDB_MODE=online
```

## 1. Install W&B If Needed

From the project root:

```bash
cd /dss/dsshome1/08/di97hih/iclr_develop
source .venv/bin/activate
python -m pip install wandb
```

## 2. Log In Once

In an SSH session:

```bash
cd /dss/dsshome1/08/di97hih/iclr_develop
source .venv/bin/activate
wandb login
```

Paste your W&B API key when prompted. This stores credentials in your home directory, so later SLURM jobs can log automatically.

## 3. Launch A Job With Online W&B Logging

```bash
sbatch scripts/train_baseline.slurm
sbatch scripts/train_spatiotemporal_mno.slurm
```

Override project or run name if needed:

```bash
WANDB_PROJECT=iclr_develop WANDB_RUN_NAME=baseline_28k_b4 sbatch scripts/train_baseline.slurm
```

Open the run in your browser from the W&B URL printed in the job log.

## 4. If Compute Nodes Have No Internet

Use offline mode:

```bash
WANDB_MODE=offline sbatch scripts/train_baseline.slurm
```

After the job finishes, sync from a login node:

```bash
cd /dss/mcmlscratch/08/di97hih/outputs/wandb
wandb sync offline-run-*
```

If there are nested run folders:

```bash
find /dss/mcmlscratch/08/di97hih/outputs/wandb -name 'offline-run-*' -type d -maxdepth 3
wandb sync /path/to/offline-run-directory
```

## Useful Overrides

Disable W&B:

```bash
ENABLE_WANDB=0 sbatch scripts/train_baseline.slurm
```

Force offline logging:

```bash
WANDB_MODE=offline sbatch scripts/train_spatiotemporal_mno.slurm
```

Use a different local W&B directory:

```bash
WANDB_DIR=/dss/mcmlscratch/08/di97hih/outputs/wandb_test sbatch scripts/train_baseline.slurm
```
