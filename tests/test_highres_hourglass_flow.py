"""CPU-scale tests for the direct high-resolution hourglass branch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from diffusion.flow_matching import FlowMatching
from models.highres_hourglass_flow import HR512HourglassFlow, resize_volume


def make_model() -> HR512HourglassFlow:
    return HR512HourglassFlow(
        input_size=32,
        in_channels=1,
        out_channels=1,
        channels_256=4,
        channels_128=8,
        transformer_hidden_size=32,
        transformer_depth=1,
        transformer_heads=4,
        bottleneck_patch_size=2,
        skip_channels_256=2,
        skip_channels_512=1,
        local_hidden_channels=2,
        gradient_checkpointing=True,
        feature_checkpointing=True,
    )


def test_forward_shapes_and_pyramid_reconstruction():
    torch.manual_seed(0)
    model = make_model().eval()
    x = torch.randn(1, 1, 32, 32, 32)
    output = model(x, torch.tensor([500.0]))

    assert output["velocity"].shape == x.shape
    assert output["local_velocity"].shape == x.shape
    assert output["global_velocity_128"].shape == (1, 1, 8, 8, 8)
    assert output["global_residual_256"].shape == (1, 1, 16, 16, 16)
    assert output["global_residual_512"].shape == x.shape
    assert output["global_velocity_512"].shape == x.shape

    reconstructed_256 = (
        resize_volume(output["global_velocity_128"], 16)
        + output["global_residual_256"]
    )
    reconstructed_512 = (
        resize_volume(reconstructed_256, 32)
        + output["global_residual_512"]
    )
    torch.testing.assert_close(reconstructed_512, output["global_velocity_512"])


def test_multiscale_loss_is_finite_and_backward_reaches_all_branches():
    torch.manual_seed(1)
    model = make_model().train()
    flow = FlowMatching(
        num_sampling_steps=2,
        residual_warmup_start_steps=1,
        residual_warmup_steps=3,
    )
    x_data = torch.randn(1, 1, 32, 32, 32)
    losses = flow.training_losses(model, x_data, t=torch.tensor([0.4]), step=2)
    loss = losses["loss"].mean()

    assert torch.isfinite(loss)
    for key in (
        "loss/local", "loss/global_128", "loss/residual_256",
        "loss/residual_512", "loss/full", "loss/total",
    ):
        assert key in losses
        assert torch.isfinite(losses[key]).all()

    loss.backward()
    parameters = dict(model.named_parameters())
    for name in (
        "local_denoiser.output.weight",
        "transformer_blocks.0.attn.qkv.weight",
        "residual_256_head.weight",
        "residual_512_head.2.weight",
    ):
        assert parameters[name].grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameters[name].grad).all()


class DummyTensorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t):
        del t
        return self.scale * x


def test_tensor_output_loss_backward_compatibility():
    model = DummyTensorModel()
    flow = FlowMatching(num_sampling_steps=2)
    losses = flow.training_losses(model, torch.randn(2, 1, 8, 8, 8))
    assert set(losses) == {"loss", "velocity_loss"}
    losses["loss"].mean().backward()
    assert model.scale.grad is not None


def test_euler_sampling_accepts_dictionary_output():
    torch.manual_seed(2)
    model = make_model().eval()
    flow = FlowMatching(num_sampling_steps=2)
    noise = torch.randn(1, 1, 32, 32, 32)
    sample, trajectory, nfe = flow.sample(
        model, x_noise=noise, num_steps=2, solver="euler"
    )
    assert sample.shape == noise.shape
    assert len(trajectory) == 1
    assert nfe == 2
    assert torch.isfinite(sample).all()

    heun_sample, _, heun_nfe = flow.sample(
        model, x_noise=noise, num_steps=2, solver="heun"
    )
    assert heun_sample.shape == noise.shape
    assert heun_nfe == 4
    assert torch.isfinite(heun_sample).all()


if __name__ == "__main__":
    test_forward_shapes_and_pyramid_reconstruction()
    test_multiscale_loss_is_finite_and_backward_reaches_all_branches()
    test_tensor_output_loss_backward_compatibility()
    test_euler_sampling_accepts_dictionary_output()
    print("All high-resolution hourglass Flow Matching tests passed.")
