# Direct 512³ Flow Matching from scratch

`HR512HourglassFlow` trains and samples one dense `512 × 512 × 512` Flow
Matching state from noise to data. The `512 → 256 → 128 → 256 → 512`
sequence is only the network's internal feature hierarchy; it is not a
128/256/512 cascade and it never replaces the full-resolution flow state.

No pretrained 128³ or 256³ checkpoint is needed. The supplied configuration
sets `model.from_scratch: True`, leaves both checkpoint paths empty, and
initializes every parameter randomly.

## Velocity decomposition

The lightweight local branch operates directly on `x_t` and predicts
`v_local` at 512³. The hourglass branch predicts the part that remains after
the local prediction:

```text
v_target = x_data - x_noise
global_target_512 = v_target - stop_gradient(v_local)
v_total = v_local + v_global_512
```

The global velocity is represented as a residual pyramid:

```text
v_global_256 = upsample(v_global_128) + delta_v_256
v_global_512 = upsample(v_global_256) + delta_v_512
```

The loss independently supervises `v_local`, `v_global_128`, `delta_v_256`,
`delta_v_512`, and the reconstructed full velocity. The 256³ and 512³ loss
weights can start at 0.1 and increase linearly after 10k steps, reaching 1.0
at 30k steps. This is an end-to-end loss curriculum, not pretraining.

## Shapes in the formal configuration

For batch size `B=1` and one voxel channel:

```text
x_t / v_local / v_total             [1, 1, 512, 512, 512]
encoder feature at 256³              [1, 8, 256, 256, 256]
encoder feature at 128³              [1, 32, 128, 128, 128]
patch tokens (patch size 8)           [1, 4096, 768]
v_global_128                          [1, 1, 128, 128, 128]
delta_v_256 / v_global_256            [1, 1, 256, 256, 256]
delta_v_512 / v_global_512            [1, 1, 512, 512, 512]
```

The 512³ skip is a single-channel high-frequency residual. It is never stored
as a wide learned feature. Both skip levels use timestep-conditioned sigmoid
gates initialized with bias `-2`.

## Verification and training

Run the CPU-scale architecture, loss, backward-compatibility, and Euler tests:

```bash
python tests/test_highres_hourglass_flow.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_highres_hourglass_flow.py
```

The environment variable keeps unrelated globally installed pytest plugins
from changing the repository test environment.

The 64³ model/data configuration is
`configs/global/lidc_64_hourglass_smoke.yaml`. After filling its dataset path,
it can be exercised with:

```bash
torchrun --standalone --nproc_per_node=1 train.py \
  --config lidc_64_hourglass_smoke.yaml
```

For the intended four-GPU run, first fill in the LIDC dataset path in
`configs/global/lidc_512_hourglass.yaml`, then run:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 train.py \
  --config lidc_512_hourglass.yaml
```

The formal configuration enables BF16 autocast, Transformer and spatial
feature checkpointing, gradient accumulation, gradient clipping, and EMA.

## Memory status

The 512³ output and loss targets are intrinsically large. The 256³ encoder
activation with 8 channels and the 128³ activation with 32 channels are also
substantial even in BF16. SDPA avoids materializing a conventional attention
matrix, and checkpointing reduces saved activations, but the configuration has
**not** been validated with a real 512³ forward/backward pass. Peak memory and
throughput still need measurement on the target `4 × A100 80GB` system; do not
interpret the CPU smoke tests as evidence that the full job fits.
