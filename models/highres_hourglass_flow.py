"""Direct high-resolution hourglass Flow Matching model.

The model always receives and returns a dense full-resolution flow state.  Its
smaller spatial grids are an internal feature hierarchy, not separate
generative stages.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def resize_volume(x: torch.Tensor, size: int) -> torch.Tensor:
    """Trilinearly resize a 5-D volume with one interpolation convention."""
    return F.interpolate(x, size=(size, size, size), mode="trilinear", align_corners=False)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class SDPAAttention(nn.Module):
    """Self-attention that uses PyTorch SDPA and preserves autocast dtype."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(
            batch, tokens, 3, self.num_heads, self.head_size
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.projection(attended)


class HighresDiTBlock(nn.Module):
    """AdaLN-conditioned Transformer block backed by memory-efficient SDPA."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.norm_mlp = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.attn = SDPAAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(conditioning).chunk(6, dim=-1)
        )
        x = x + gate_attn[:, None, :] * self.attn(
            _modulate(self.norm_attention(x), shift_attn, scale_attn)
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            _modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )
        return x


class SinusoidalTimeEmbedding(nn.Module):
    """Numerically stable sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half, 1)
        )
        angles = t.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.frequency_size % 2:
            embedding = F.pad(embedding, (0, 1))
        return self.mlp(embedding)


class LightweightLocalDenoiser(nn.Module):
    """Shallow high-resolution velocity head with at most a few channels."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, time_size: int):
        super().__init__()
        if not 1 <= hidden_channels <= 8:
            raise ValueError("local_hidden_channels must be between 1 and 8")
        self.input = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = nn.Conv3d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1,
            groups=hidden_channels,
        )
        self.time_scale_shift = nn.Linear(time_size, 2 * hidden_channels)
        self.output = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        scale, shift = self.time_scale_shift(time_embedding).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        h = h + self.depthwise(F.silu(h))
        return self.output(F.silu(h))


class HR512HourglassFlow(nn.Module):
    """Direct dense Flow Matching model with a residual velocity pyramid.

    ``input_size`` is configurable solely so that the same architecture can be
    tested cheaply.  The public output names describe their role in the real
    512 -> 256 -> 128 configuration.
    """

    def __init__(
        self,
        input_size: int = 512,
        in_channels: int = 1,
        out_channels: int = 1,
        channels_256: int = 8,
        channels_128: int = 32,
        transformer_hidden_size: int = 768,
        transformer_depth: int = 4,
        transformer_heads: int = 12,
        bottleneck_patch_size: int = 8,
        skip_channels_256: int = 4,
        skip_channels_512: int = 1,
        local_hidden_channels: int = 2,
        mlp_ratio: float = 4.0,
        gradient_checkpointing: bool = True,
        feature_checkpointing: bool = True,
        **_unused,
    ):
        super().__init__()
        if input_size % 4:
            raise ValueError("input_size must be divisible by four")
        if (input_size // 4) % bottleneck_patch_size:
            raise ValueError("bottleneck_patch_size must divide input_size / 4")
        if transformer_hidden_size % transformer_heads:
            raise ValueError("transformer_hidden_size must be divisible by transformer_heads")
        if out_channels != in_channels:
            raise ValueError("Flow velocity channels must match the input state channels")
        if not 1 <= skip_channels_512 <= 4:
            raise ValueError("skip_channels_512 must be between 1 and 4")

        self.input_size = int(input_size)
        self.size_256 = self.input_size // 2
        self.size_128 = self.input_size // 4
        self.depth = int(transformer_depth)  # keeps existing evaluation routing generative
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.feature_checkpointing = bool(feature_checkpointing)

        time_size = transformer_hidden_size
        self.t_embedder = SinusoidalTimeEmbedding(time_size)
        self.local_denoiser = LightweightLocalDenoiser(
            in_channels, out_channels, local_hidden_channels, time_size
        )

        self.stem_512_to_256 = nn.Conv3d(
            in_channels, channels_256, kernel_size=4, stride=2, padding=1
        )
        self.encoder_256_to_128 = nn.Conv3d(
            channels_256, channels_128, kernel_size=4, stride=2, padding=1
        )
        self.skip_256_projection = nn.Conv3d(channels_256, skip_channels_256, 1)

        patch = bottleneck_patch_size
        self.patch_embed = nn.Conv3d(
            channels_128, transformer_hidden_size, kernel_size=patch, stride=patch
        )
        token_grid = self.size_128 // patch
        self.pos_d = nn.Parameter(torch.zeros(1, token_grid, transformer_hidden_size))
        self.pos_h = nn.Parameter(torch.zeros(1, token_grid, transformer_hidden_size))
        self.pos_w = nn.Parameter(torch.zeros(1, token_grid, transformer_hidden_size))
        self.transformer_blocks = nn.ModuleList([
            HighresDiTBlock(
                transformer_hidden_size,
                transformer_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(transformer_depth)
        ])
        self.token_norm = nn.LayerNorm(transformer_hidden_size, eps=1e-6)
        self.patch_decode = nn.ConvTranspose3d(
            transformer_hidden_size, channels_128, kernel_size=patch, stride=patch
        )
        self.global_128_head = nn.Conv3d(channels_128, out_channels, 3, padding=1)

        self.decoder_128_to_256 = nn.ConvTranspose3d(
            channels_128, channels_256, kernel_size=4, stride=2, padding=1
        )
        self.skip_256_to_decoder = nn.Conv3d(skip_channels_256, channels_256, 1)
        self.gate_256 = nn.Linear(time_size, channels_256)
        self.residual_256_head = nn.Conv3d(channels_256, out_channels, 3, padding=1)

        self.skip_512_projection = nn.Conv3d(in_channels, skip_channels_512, 1)
        self.skip_512_to_output = nn.Conv3d(skip_channels_512, out_channels, 1)
        self.gate_512 = nn.Linear(time_size, out_channels)
        self.residual_512_head = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 1),
        )

        nn.init.normal_(self.pos_d, std=0.02)
        nn.init.normal_(self.pos_h, std=0.02)
        nn.init.normal_(self.pos_w, std=0.02)
        nn.init.constant_(self.gate_256.bias, -2.0)
        nn.init.constant_(self.gate_512.bias, -2.0)

        self._config_to_log = {
            "input hierarchy": f"{self.input_size}->{self.size_256}->{self.size_128}",
            "channels": f"{in_channels}->{channels_256}->{channels_128}",
            "tokens": token_grid ** 3,
            "transformer hidden/depth/heads": (
                transformer_hidden_size, transformer_depth, transformer_heads
            ),
        }

    def log_config(self, rank: int = 0) -> None:
        if rank == 0:
            for key, value in self._config_to_log.items():
                print(f"HR512HourglassFlow {key}: {value}")

    def _maybe_checkpoint(self, module, *args, enabled: bool):
        if enabled and self.training and torch.is_grad_enabled():
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)

    def _transformer(self, feature_128: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        tokens_3d = self.patch_embed(feature_128)
        batch, hidden, depth, height, width = tokens_3d.shape
        position = (
            self.pos_d[:, :depth, None, None, :]
            + self.pos_h[:, None, :height, None, :]
            + self.pos_w[:, None, None, :width, :]
        ).reshape(1, depth * height * width, hidden)
        tokens = tokens_3d.flatten(2).transpose(1, 2)
        tokens = tokens + position.to(dtype=tokens.dtype)
        for block in self.transformer_blocks:
            tokens = self._maybe_checkpoint(
                block, tokens, time_embedding, enabled=self.gradient_checkpointing
            )
        tokens = self.token_norm(tokens)
        tokens_3d = tokens.transpose(1, 2).reshape(batch, hidden, depth, height, width)
        return self.patch_decode(tokens_3d)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **_unused) -> Dict[str, torch.Tensor]:
        if x_t.ndim != 5 or tuple(x_t.shape[2:]) != (self.input_size,) * 3:
            raise ValueError(
                f"Expected [B, C, {self.input_size}, {self.input_size}, {self.input_size}], "
                f"got {tuple(x_t.shape)}"
            )
        time_embedding = self.t_embedder(t)
        local_velocity = self.local_denoiser(x_t, time_embedding)

        feature_256 = self._maybe_checkpoint(
            self.stem_512_to_256, x_t, enabled=self.feature_checkpointing
        )
        feature_256 = F.silu(feature_256)
        skip_256 = self.skip_256_projection(feature_256)
        feature_128 = self._maybe_checkpoint(
            self.encoder_256_to_128, feature_256, enabled=self.feature_checkpointing
        )
        feature_128 = F.silu(feature_128)

        transformed_128 = self._transformer(feature_128, time_embedding)
        transformed_128 = transformed_128 + feature_128
        global_velocity_128 = self.global_128_head(F.silu(transformed_128))

        decoder_256 = self._maybe_checkpoint(
            self.decoder_128_to_256, transformed_128, enabled=self.feature_checkpointing
        )
        gate_256 = torch.sigmoid(self.gate_256(time_embedding))[:, :, None, None, None]
        decoder_256 = decoder_256 + gate_256 * self.skip_256_to_decoder(skip_256)
        delta_velocity_256 = self.residual_256_head(F.silu(decoder_256))
        global_velocity_256 = resize_volume(global_velocity_128, self.size_256) + delta_velocity_256

        x_256 = F.avg_pool3d(x_t, kernel_size=2, stride=2)
        skip_512 = x_t - resize_volume(x_256, self.input_size)
        skip_512 = self.skip_512_to_output(self.skip_512_projection(skip_512))
        gate_512 = torch.sigmoid(self.gate_512(time_embedding))[:, :, None, None, None]
        global_512_base = resize_volume(global_velocity_256, self.input_size)
        decoder_512 = global_512_base + gate_512 * skip_512
        delta_velocity_512 = self.residual_512_head(decoder_512)
        global_velocity_512 = global_512_base + delta_velocity_512
        velocity = local_velocity + global_velocity_512

        return {
            "velocity": velocity,
            "local_velocity": local_velocity,
            "global_velocity_128": global_velocity_128,
            "global_residual_256": delta_velocity_256,
            "global_residual_512": delta_velocity_512,
            "global_velocity_256": global_velocity_256,
            "global_velocity_512": global_velocity_512,
        }
