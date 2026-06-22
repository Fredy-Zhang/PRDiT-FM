"""Flow Matching (rectified-flow / straight-line) process for PRDiT.

Unified definition used across this codebase
-------------------------------------------
Let ``x_noise ~ N(0, I)`` and ``x_data`` be a real sample, with time
``t ~ U(0, 1)``::

    x_t              = (1 - t) * x_noise + t * x_data
    target_velocity  = x_data - x_noise

Convention (deliberately avoids ``x0`` / ``x1`` naming):

- ``t = 0``  -> pure noise        (``x_t = x_noise``)
- ``t = 1``  -> real data         (``x_t = x_data``)
- the model predicts the **velocity** ``v_theta(x_t, t)`` directly,
  trained with ``L = || v_theta(x_t, t) - (x_data - x_noise) ||^2``.

Notes
-----
* Straight-line path with ``sigma_min = 0`` (the simplest baseline).
* ``t`` is continuous in ``[0, 1]``. Only when feeding the model's timestep
  embedder is ``t`` rescaled to the embedder's expected range
  (``model_t = t * embed_t_scale``, default ``1000``); the interpolation and
  velocity targets always use the raw ``t in [0, 1]``.
* There is **no** DDPM ``beta`` / ``alpha`` / ``alphas_cumprod`` / posterior
  logic here. ``num_timesteps`` of DDPM is unrelated to either the number of
  training iterations or the number of sampling steps.
* Sampling solves the ODE ``dx/dt = v_theta(x_t, t)`` from ``t = 0`` (noise)
  to ``t = 1`` (data). Euler is the default; Heun is an optional 2nd-order
  solver. Report both ``num_steps`` and NFE (number of function evaluations).
"""

from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class FlowMatching:
    """Straight-line Flow Matching process with a single velocity head.

    Parameters
    ----------
    num_sampling_steps : int, optional
        Default number of ODE integration steps used by :meth:`sample`
        (default ``100``).
    loss_type : str, optional
        Velocity reconstruction loss; only ``"l2"`` is supported (default
        ``"l2"``).
    embed_t_scale : float, optional
        Scale applied to ``t in [0, 1]`` before it is passed to the model's
        timestep embedder (default ``1000.0``). This does **not** affect the
        interpolation or the velocity target.
    """

    def __init__(
        self,
        num_sampling_steps: int = 100,
        loss_type: str = "l2",
        embed_t_scale: float = 1000.0,
        lambda_local: float = 1.0,
        lambda_128: float = 1.0,
        lambda_256: float = 1.0,
        lambda_512: float = 1.0,
        lambda_full: float = 1.0,
        residual_warmup_steps: int = 0,
        residual_warmup_start_steps: int = 10_000,
        residual_warmup_initial_weight: float = 0.1,
        use_bf16: bool = False,
    ):
        if loss_type != "l2":
            raise NotImplementedError(f"Loss type '{loss_type}' is not implemented.")
        self.num_sampling_steps = int(num_sampling_steps)
        self.loss_type = loss_type
        self.embed_t_scale = float(embed_t_scale)
        self.lambda_local = float(lambda_local)
        self.lambda_128 = float(lambda_128)
        self.lambda_256 = float(lambda_256)
        self.lambda_512 = float(lambda_512)
        self.lambda_full = float(lambda_full)
        self.residual_warmup_steps = int(residual_warmup_steps)
        self.residual_warmup_start_steps = int(residual_warmup_start_steps)
        self.residual_warmup_initial_weight = float(residual_warmup_initial_weight)
        self.use_bf16 = bool(use_bf16)

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
        """Reshape a ``[B]`` time tensor to broadcast over a volume tensor."""
        return t.reshape((-1,) + (1,) * (ndim - 1))

    def _model_t(self, t: torch.Tensor) -> torch.Tensor:
        """Map ``t in [0, 1]`` to the model's timestep-embedder input range."""
        return t * self.embed_t_scale

    @staticmethod
    def _velocity(output: Union[torch.Tensor, Mapping[str, torch.Tensor]]) -> torch.Tensor:
        """Select the final velocity while preserving tensor-model support."""
        if isinstance(output, Mapping):
            if "velocity" not in output:
                raise KeyError("Dictionary model output must contain a 'velocity' tensor")
            return output["velocity"]
        return output

    @staticmethod
    def _resize(x: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
        return F.interpolate(x, size=spatial_shape, mode="trilinear", align_corners=False)

    def _call_model(self, model, x: torch.Tensor, t: torch.Tensor, model_kwargs: Dict):
        enabled = self.use_bf16 and x.device.type in {"cuda", "cpu"}
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=enabled):
            return model(x, self._model_t(t), **model_kwargs)

    def _residual_weight_scale(self, step: Optional[int]) -> float:
        """Return the configurable 256/512 curriculum multiplier."""
        if self.residual_warmup_steps <= 0 or step is None:
            return 1.0
        start = min(self.residual_warmup_start_steps, self.residual_warmup_steps)
        if step <= start:
            return self.residual_warmup_initial_weight
        if step >= self.residual_warmup_steps:
            return 1.0
        progress = (step - start) / max(self.residual_warmup_steps - start, 1)
        return self.residual_warmup_initial_weight + progress * (
            1.0 - self.residual_warmup_initial_weight
        )

    def interpolate(
        self,
        x_data: torch.Tensor,
        x_noise: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the straight-line interpolant and its velocity target.

        Returns ``(x_t, target_velocity)`` with
        ``x_t = (1 - t) * x_noise + t * x_data`` and
        ``target_velocity = x_data - x_noise``.
        """
        t_b = self._expand_t(t, x_data.ndim).to(x_data.dtype)
        x_t = (1.0 - t_b) * x_noise + t_b * x_data
        target_velocity = x_data - x_noise
        return x_t, target_velocity

    # -- Training -------------------------------------------------------------

    def training_losses(
        self,
        model: torch.nn.Module,
        x_data: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        step: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the per-sample velocity-matching loss.

        Parameters
        ----------
        model : torch.nn.Module
            Velocity model returning a tensor with the same channel count as
            ``x_data``.
        x_data : torch.Tensor
            Clean input volumes of shape ``[B, C, D, H, W]``.
        t : torch.Tensor or None, optional
            Continuous times in ``[0, 1]`` of shape ``[B]``. Sampled uniformly
            when ``None`` (the standard Flow Matching choice).
        model_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the model.

        Returns
        -------
        dict
            ``{"loss": Tensor[B], "velocity_loss": Tensor[B]}``
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = x_data.shape[0]
        if t is None:
            t = torch.rand(batch_size, device=x_data.device, dtype=x_data.dtype)
        else:
            t = t.to(device=x_data.device, dtype=x_data.dtype)

        x_noise = torch.randn_like(x_data)
        x_t, target_velocity = self.interpolate(x_data, x_noise, t)

        output = self._call_model(model, x_t, t, model_kwargs)

        spatial_dims = list(range(1, x_data.ndim))
        if not isinstance(output, Mapping):
            velocity_loss = (output - target_velocity).pow(2).mean(dim=spatial_dims)
            return {"loss": velocity_loss, "velocity_loss": velocity_loss}

        required = {
            "velocity", "local_velocity", "global_velocity_128",
            "global_residual_256", "global_residual_512",
        }
        missing = required.difference(output)
        if missing:
            raise KeyError(f"Dictionary model output is missing: {sorted(missing)}")

        v_total = output["velocity"]
        v_local = output["local_velocity"]
        v_global_128 = output["global_velocity_128"]
        delta_v_256 = output["global_residual_256"]
        delta_v_512 = output["global_residual_512"]

        # Detaching here is essential: pyramid losses must not update the local
        # branch through the definition of their residual target.
        global_target_512 = target_velocity - v_local.detach()
        global_target_256 = self._resize(global_target_512, delta_v_256.shape[2:])
        global_target_128 = self._resize(global_target_512, v_global_128.shape[2:])
        delta_target_256 = global_target_256 - self._resize(
            global_target_128, delta_v_256.shape[2:]
        )
        delta_target_512 = global_target_512 - self._resize(
            global_target_256, delta_v_512.shape[2:]
        )

        loss_local = (v_local - target_velocity).pow(2).mean(dim=spatial_dims)
        loss_global_128 = (v_global_128 - global_target_128).pow(2).mean(dim=spatial_dims)
        loss_residual_256 = (delta_v_256 - delta_target_256).pow(2).mean(dim=spatial_dims)
        loss_residual_512 = (delta_v_512 - delta_target_512).pow(2).mean(dim=spatial_dims)
        loss_full = (v_total - target_velocity).pow(2).mean(dim=spatial_dims)

        residual_scale = self._residual_weight_scale(step)
        total = (
            self.lambda_local * loss_local
            + self.lambda_128 * loss_global_128
            + residual_scale * self.lambda_256 * loss_residual_256
            + residual_scale * self.lambda_512 * loss_residual_512
            + self.lambda_full * loss_full
        )
        return {
            "loss": total,
            "velocity_loss": loss_full,
            "loss/local": loss_local,
            "loss/global_128": loss_global_128,
            "loss/residual_256": loss_residual_256,
            "loss/residual_512": loss_residual_512,
            "loss/full": loss_full,
            "loss/total": total,
            "loss/residual_weight_scale": torch.full_like(total, residual_scale),
        }

    # -- Reconstruction (used by depth-0 / coarse evaluation) -----------------

    @torch.no_grad()
    def predict_x_data(
        self,
        model: torch.nn.Module,
        x_data: torch.Tensor,
        t_value: float = 0.5,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate ``x_data`` from a noised sample via the predicted velocity.

        Given ``x_t`` at a single time ``t``, the straight-line model implies
        ``x_data = x_t + (1 - t) * v_theta(x_t, t)``. Returns
        ``(x_data_hat, x_t)``. Used as a cheap reconstruction metric for the
        coarse (depth-0) stage.
        """
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = x_data.shape[0]
        t = torch.full((batch_size,), float(t_value), device=x_data.device, dtype=x_data.dtype)
        x_noise = torch.randn_like(x_data)
        x_t, _ = self.interpolate(x_data, x_noise, t)

        v_pred = self._velocity(self._call_model(model, x_t, t, model_kwargs))
        x_data_hat = x_t + (1.0 - t_value) * v_pred
        return x_data_hat, x_t

    # -- Sampling -------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: Optional[Tuple[int, ...]] = None,
        x_noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        solver: str = "euler",
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """Integrate ``dx/dt = v_theta(x_t, t)`` from ``t=0`` (noise) to ``t=1``.

        Parameters
        ----------
        model : torch.nn.Module
            Velocity model.
        shape : tuple of int, optional
            Output shape; required when ``x_noise`` is not given.
        x_noise : torch.Tensor, optional
            Initial ``N(0, I)`` sample at ``t = 0``. Drawn from ``shape`` when
            omitted.
        num_steps : int, optional
            Number of integration steps (default :attr:`num_sampling_steps`).
        solver : {"euler", "heun"}, optional
            ODE integrator (default ``"euler"``).
        device : torch.device, optional
            Device for a freshly drawn ``x_noise``.
        model_kwargs : dict or None, optional
            Extra keyword arguments forwarded to the model.
        return_trajectory : bool, optional
            Also return the per-step trajectory tensors (default ``False``).

        Returns
        -------
        tuple
            ``(x_data_sample, trajectory, nfe)`` — the final ``t = 1`` sample,
            an optional list of intermediate states (``[x_noise]`` when
            ``return_trajectory`` is ``False``), and the number of function
            evaluations.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if num_steps is None:
            num_steps = self.num_sampling_steps
        solver = solver.lower()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unknown solver '{solver}'. Use 'euler' or 'heun'.")

        if x_noise is None:
            if shape is None:
                raise ValueError("Either `shape` or `x_noise` must be provided.")
            x_noise = torch.randn(shape, device=device)

        x = x_noise
        device = x.device
        batch_size = x.shape[0]
        dt = 1.0 / num_steps

        trajectory: List[torch.Tensor] = [x.detach().cpu()] if return_trajectory else [x_noise]
        nfe = 0

        for step in range(num_steps):
            t_scalar = step * dt
            t = torch.full((batch_size,), t_scalar, device=device, dtype=x.dtype)
            v = self._velocity(self._call_model(model, x, t, model_kwargs))
            nfe += 1

            if solver == "euler":
                x = x + dt * v
            else:  # heun: predictor + endpoint-velocity corrector
                x_pred = x + dt * v
                t_next = torch.full((batch_size,), min((step + 1) * dt, 1.0), device=device, dtype=x.dtype)
                v_next = self._velocity(self._call_model(model, x_pred, t_next, model_kwargs))
                nfe += 1
                x = x + dt * 0.5 * (v + v_next)

            if return_trajectory:
                trajectory.append(x.detach().cpu())

        return x, trajectory, nfe

    # -- Backward-compatible wrapper for the training / sampling scripts ------

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        z: torch.Tensor,
        clip_denoised: bool = False,
        progress: bool = False,
        new_sampling: bool = False,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Compatibility shim mirroring the old diffusion sampler interface.

        ``z`` is the initial ``N(0, I)`` noise at ``t = 0``; ``new_sampling``
        selects the Heun solver. Returns ``(trajectory, final_sample)`` so that
        existing callers (``xs[-1]`` / final sample) keep working.
        """
        solver = "heun" if new_sampling else "euler"
        print(
            f"Flow-matching sampling: solver={solver}, "
            f"steps={self.num_sampling_steps}, "
            f"NFE={'~' + str(2 * self.num_sampling_steps) if solver == 'heun' else self.num_sampling_steps}"
        )
        x_sample, trajectory, nfe = self.sample(
            model,
            x_noise=z,
            num_steps=self.num_sampling_steps,
            solver=solver,
            model_kwargs=model_kwargs,
            return_trajectory=True,
        )
        print(f"Flow-matching sampling done: NFE={nfe}")
        return trajectory, x_sample
