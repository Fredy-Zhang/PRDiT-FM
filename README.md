# PRDiT-FM: Flow Matching for Pixel-Level 3D CT Generation

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=bWtRZQ1rm2)
[![Poster](https://img.shields.io/badge/Poster-ICLR%202026-8A2BE2)](https://iclr.cc/media/PosterPDFs/ICLR%202026/10008602.png?t=1774447885.2973316)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository extends **PRDiT** (*Pixel-Level Residual Diffusion Transformer*)
with a straight-line **Flow Matching** objective and ODE-based sampling for
voxel-level 3D CT generation. The original two-stage coarse-to-fine PRDiT
architecture is retained: a local patch denoiser learns coarse structure, then
a global residual DiT refines the full volume.

The original PRDiT paper was accepted at **ICLR 2026**:
[paper](https://openreview.net/forum?id=bWtRZQ1rm2) ·
[poster](https://iclr.cc/media/PosterPDFs/ICLR%202026/10008602.png?t=1774447885.2973316)

<p align="center">
  <img src="assets/overview.png" width="95%" alt="PRDiT architecture overview">
</p>

## What changed from PRDiT?

| Component | Previous PRDiT path | PRDiT-FM path |
|---|---|---|
| Training process | Discrete diffusion / dual image-and-noise prediction | Continuous straight-line Flow Matching |
| Model output | Two channels (`out_channels: 2`) | One velocity channel (`out_channels: 1`) |
| Time | Discrete diffusion timesteps | Continuous `t ~ U(0, 1)` |
| Sampling | Reverse diffusion | Forward ODE integration from noise to data |
| Solver | Diffusion sampler | Euler (default) or second-order Heun |
| Sampling budget | Diffusion timesteps | ODE steps and number of function evaluations (NFE) |

For a data volume `x_data`, Gaussian noise `x_noise`, and continuous time `t`,
the new process uses

```text
x_t = (1 - t) x_noise + t x_data
v_target = x_data - x_noise
L_FM = ||v_theta(x_t, t) - v_target||^2
```

Here, `t = 0` is noise and `t = 1` is data. During sampling, the model solves
`dx/dt = v_theta(x_t, t)` in that direction. The continuous time is multiplied
by 1000 only when passed to PRDiT's existing timestep embedder; this does not
change the interpolation or velocity target.

Other implementation changes include:

- a unified implementation in [`diffusion/flow_matching.py`](diffusion/flow_matching.py);
- step-based training through `training.total_steps`;
- depth-0 validation by reconstruction at `t = 0.5`, with best-checkpoint saving;
- automatic loading and freezing of the coarse path for stage 2;
- an end-to-end script that creates splits, trains both stages, wires the
  coarse checkpoint into the global config, samples, and optionally evaluates.

The legacy `IaNDiffusion` and Gaussian diffusion code remains under
[`diffusion/`](diffusion/) for comparison and checkpoint compatibility.

## Repository layout

```text
configs/local/              Stage-1 (depth 0) configurations
configs/global/             Stage-2 (depth > 0) configurations
datasets/                   LIDC-IDRI and RAD-ChestCT loaders
diffusion/flow_matching.py  Flow Matching loss and ODE solvers
models/                     Local denoiser and global residual DiT
evaluations/                3D FID, MMD, and MS-SSIM
scripts/run_flow_pipeline.sh
train.py                    Distributed two-stage training
sample.py                   Flow Matching inference
```

## Installation

The training code requires a CUDA-capable PyTorch installation for distributed
training. Python 3.10 and PyTorch 2.x are recommended.

```bash
conda create -n prdit-fm python=3.10 -y
conda activate prdit-fm

# Select the PyTorch command appropriate for your CUDA version from pytorch.org.
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install numpy scipy pyyaml nibabel monai timm wandb tqdm \
  matplotlib pillow opencv-python tensorboard dicom2nifti
```

> This snapshot does not include a root `requirements.txt`. The command above
> lists the packages imported by training, sampling, visualization, and dataset
> preprocessing. Evaluation can instead use the pinned environment described
> in [`evaluations/README.md`](evaluations/README.md).

The `flash_attn` config flag uses PyTorch scaled-dot-product attention; it does
not require the separate `flash-attn` package.

## Data preparation

PRDiT supports **LIDC-IDRI** and **RAD-ChestCT**. Follow
[`datasets/README.md`](datasets/README.md) to download and preprocess a dataset,
then set these fields in both its local and global YAML files:

```yaml
data:
  path: /absolute/path/to/preprocessed/data
  train_list: lidc_data/train.txt  # or rad_data/train.txt
  val_list: lidc_data/val.txt      # or rad_data/val.txt
```

The end-to-end pipeline creates missing train/validation lists using
`data.val_frac` and `training.seed`.

## Configuration and process selection

`train.py` selects the config directory from the training stage:

- `--from_scratch` loads `configs/local/<name>` for the depth-0 coarse model;
- without `--from_scratch`, it loads `configs/global/<name>` for the global
  residual model.

The generative process is selected by `model.out_channels`:

- `out_channels: 1` → Flow Matching with a single velocity head;
- `out_channels: 2` → legacy `IaNDiffusion` with image and noise heads.

The supplied LIDC configs are configured for Flow Matching. The supplied RAD
configs currently retain the legacy two-head setting. To run PRDiT-FM on RAD,
change `model.out_channels` to `1` in both `configs/local/rad.yaml` and
`configs/global/rad.yaml`, and add `model.num_sampling_steps` (for example,
`100`) to both configs before training new checkpoints. Do not mix one-head and
two-head checkpoints.

## Training

### Recommended: complete pipeline

After editing the YAML paths, run both training stages and sampling with:

```bash
CONFIG=lidc.yaml NPROC=4 bash scripts/run_flow_pipeline.sh
```

The script automatically:

1. generates missing train/validation splits;
2. trains the local depth-0 model;
3. finds its latest best checkpoint and writes that path to
   `configs/global/<CONFIG>`;
4. trains the global residual model with the coarse path frozen;
5. finds the global checkpoint and generates evaluation samples;
6. optionally computes 3D FID and MMD when evaluation weights and real data
   are provided.

Useful overrides:

```bash
# Skip completed training stages and only sample/evaluate.
RUN_COARSE=0 RUN_FINE=0 RUN_EVAL=1 \
CONFIG=lidc.yaml bash scripts/run_flow_pipeline.sh

# Generate more samples with more Euler steps.
EVAL_TOTAL_SAMPLES=1000 EVAL_NUM_SAMPLES=4 EVAL_SAMPLING_STEPS=200 \
CONFIG=lidc.yaml NPROC=4 bash scripts/run_flow_pipeline.sh

# Enable FID and MMD after sampling.
FID_PRETRAIN_PATH=evaluations/pretrained/resnet_50.pth \
DATA_ROOT_REAL=/path/to/preprocessed/lidc \
EVAL_DATASET=lidc-idri \
CONFIG=lidc.yaml NPROC=4 bash scripts/run_flow_pipeline.sh
```

`NPROC` defaults to the detected GPU count and is reduced when necessary so it
divides the configured global batch size. Set `RUN_COARSE`, `RUN_FINE`, or
`RUN_EVAL` to `0` to skip a stage.

> The pipeline updates `model.pretrained_path` in the selected global YAML file.

### Manual two-stage training

Stage 1 trains the local model from scratch:

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  train.py --config lidc.yaml --from_scratch
```

Find the stage-1 checkpoint under
`results_lidc/<run>-PRDiT-B-12-0/checkpoints/`, set it in
`configs/global/lidc.yaml`, then train stage 2:

```yaml
model:
  pretrained_path: results_lidc/001-PRDiT-B-12-0/checkpoints/best_<loss>.pt
```

```bash
torchrun --nnodes=1 --nproc_per_node=4 \
  train.py --config lidc.yaml
```

The configured `training.batch_size` is global and must be divisible by the
number of distributed processes. Checkpoints, logs, and validation samples are
written beneath `output.results_dir`.

## Sampling

`sample.py` always reads a filename from `configs/global/` and uses Flow
Matching, so pass `lidc.yaml` rather than a full config path:

```bash
python sample.py \
  --config lidc.yaml \
  --ckpt results_lidc/002-PRDiT-B-12-4/checkpoints/130000.pt \
  --num-samples 4 \
  --total-samples 100 \
  --num-sampling-steps 100 \
  --output-dir samples/lidc_euler
```

Euler is the default solver and uses one model evaluation per step, so
`NFE = num_sampling_steps`. Add `--new` to use Heun:

```bash
python sample.py \
  --config lidc.yaml \
  --ckpt /path/to/global_checkpoint.pt \
  --num-sampling-steps 100 \
  --output-dir samples/lidc_heun \
  --new
```

Heun uses two model evaluations per step (`NFE = 2 × num_sampling_steps`). Each
output directory contains:

- `xs/`: final ODE samples at `t = 1`;
- `x0/`: the sampler's returned final data estimate (identical to the final
  state in the current Flow Matching implementation).

Volumes are saved as NIfTI files with orthogonal-view PNG previews.

## Evaluation

See [`evaluations/README.md`](evaluations/README.md) for complete setup and
commands. The available in-repository metrics are:

- **3D FID** using MedicalNet 3D ResNet-50 features;
- **3D MMD** using the same features and an RBF kernel;
- **MS-SSIM** for sample diversity.

For example, after generating samples into `samples/lidc_euler/xs`:

```bash
python evaluations/fid.py \
  --dataset lidc-idri \
  --img_size 128 \
  --data_root_real /path/to/preprocessed/lidc \
  --data_root_fake samples/lidc_euler/xs \
  --pretrain_path evaluations/pretrained/resnet_50.pth \
  --path_to_activations samples/lidc_euler/activations
```

The W-Critic used for the paper's Wasserstein estimate is not included in this
repository.

## Reproducibility notes

- Report the model config, checkpoint, solver, ODE step count, and NFE.
- Flow Matching checkpoints require `out_channels: 1` and are not compatible
  with legacy two-head checkpoints.
- `num_sampling_steps` controls inference only; it is unrelated to
  `training.total_steps`.
- The stage-2 run depends on the exact stage-1 checkpoint recorded in
  `model.pretrained_path`.
- The pipeline mutates the global config when it records that checkpoint, so
  preserve the resulting YAML with experimental artifacts.

## Citation

If this code is useful in your research, please cite the original PRDiT paper:

```bibtex
@inproceedings{zhang2026pixellevel,
  title     = {Pixel-Level Residual Diffusion Transformer: Scalable 3D {CT} Volume Generation},
  author    = {Zhenkai Zhang and Markus Hiller and Krista A. Ehinger and Tom Drummond},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=bWtRZQ1rm2}
}
```

## License

Released under the [Apache License 2.0](LICENSE).
