# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asymmetric actor-critic network architecture with 5-frame history encoder.

Implements the architecture specified in ``README.md``:
- Single-frame proprioceptive observation: ``o_t in R^25``
- 5-frame flattened history observation: ``s_t = [o_{t-4}; ...; o_t] in R^125``
- History encoder ``f_phi``: ``125 -> 128 -> 64 -> 3`` (24,579 params; deployed)
- Policy actor ``pi_theta``: ``28 -> 128 -> 64 -> 32 -> 6`` (14,246 params; deployed)
- Value critic ``V_psi``: ``144 -> 256 -> 128 -> 64 -> 1`` (78,337 params; training only)
- Total deployed parameters (encoder + actor): 38,825 params, 38,400 MACs.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from tensordict import TensorDict

# Network dimension constants matching README.md
OBS_DIM: int = 25
HISTORY_LENGTH: int = 5
HISTORY_OBS_DIM: int = OBS_DIM * HISTORY_LENGTH  # 125
LATENT_DIM: int = 3
ACTOR_INPUT_DIM: int = OBS_DIM + LATENT_DIM  # 28
ACTION_DIM: int = 6
CRITIC_OBS_DIM: int = 144

ENCODER_HIDDEN_DIMS: tuple[int, ...] = (128, 64)
ACTOR_HIDDEN_DIMS: tuple[int, ...] = (128, 64, 32)
CRITIC_HIDDEN_DIMS: tuple[int, ...] = (256, 128, 64)

ENCODER_NUM_PARAMS: int = 24_579
ACTOR_NUM_PARAMS: int = 14_246
DEPLOYED_NUM_PARAMS: int = ENCODER_NUM_PARAMS + ACTOR_NUM_PARAMS  # 38,825
DEPLOYED_MACS: int = 38_400
CRITIC_NUM_PARAMS: int = 78_337

DEFAULT_ONNX_OPSET: int = 18
FALLBACK_ONNX_OPSET: int = 17


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count the number of parameters in a PyTorch module.

    Args:
        module: PyTorch module whose parameters will be counted.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in module.parameters() if not trainable_only or p.requires_grad)


def count_linear_macs(module: nn.Module) -> int:
    """Count multiply-accumulate operations (MACs) across all nn.Linear layers.

    Args:
        module: PyTorch module to inspect.

    Returns:
        Sum of in_features * out_features across all linear layers.
    """
    return sum(m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear))


class EncoderActorModel(MLPModel):
    """Actor model with a 5-frame history MLP encoder f_phi and policy head pi_theta.

    Given the flattened 5-frame observation history ``s_t = [o_{t-4}; ...; o_t]`` in R^125,
    computes ``z_t = f_phi(s_t)`` in R^3 and ``a_t = pi_theta([o_t; z_t])`` in R^6,
    where ``o_t = s_t[..., -25:]`` in R^25 is the current frame observation.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = ACTION_DIM,
        hidden_dims: tuple[int, ...] | list[int] = ACTOR_HIDDEN_DIMS,
        encoder_hidden_dims: tuple[int, ...] | list[int] = ENCODER_HIDDEN_DIMS,
        latent_dim: int = LATENT_DIM,
        frame_obs_dim: int = OBS_DIM,
        history_length: int = HISTORY_LENGTH,
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize the encoder actor model.

        Args:
            obs: Sample observation TensorDict used to infer input shapes.
            obs_groups: Mapping from observation set name to active observation group keys.
            obs_set: Observation set key (typically "actor").
            output_dim: Action dimension output by pi_theta.
            hidden_dims: Hidden layer dimensions for pi_theta.
            encoder_hidden_dims: Hidden layer dimensions for history encoder f_phi.
            latent_dim: Latent dimension output by f_phi.
            frame_obs_dim: Single-frame proprioceptive observation dimension.
            history_length: Number of stacked frames in the actor observation history.
            activation: Activation function name for both MLPs.
            obs_normalization: Whether to normalize flattened actor observations.
            distribution_cfg: Optional action distribution configuration dictionary.
            **_kwargs: Unused extra keyword arguments forwarded by runner factories.

        Raises:
            ValueError: If concatenated actor observation dimension is incompatible with
                frame_obs_dim and history_length.
        """
        self.frame_obs_dim = int(frame_obs_dim)
        self.history_length = int(history_length)
        self.history_obs_dim = self.frame_obs_dim * self.history_length
        self.latent_dim = int(latent_dim)
        self.encoder_hidden_dims = tuple(encoder_hidden_dims)
        self._last_latent: torch.Tensor | None = None

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        expected_dims = {self.history_obs_dim, self.frame_obs_dim + self.history_obs_dim}
        if self.obs_dim not in expected_dims:
            raise ValueError(
                f"EncoderActorModel expects concatenated actor observation dimension in {sorted(expected_dims)} "
                f"(for frame_obs_dim={self.frame_obs_dim}, history_length={self.history_length}), "
                f"but got {self.obs_dim} from groups {self.obs_groups}."
            )

        self.encoder = MLP(
            input_dim=self.history_obs_dim,
            output_dim=self.latent_dim,
            hidden_dims=list(self.encoder_hidden_dims),
            activation=activation,
        )

    @property
    def actor(self) -> MLP:
        """Policy actor MLP pi_theta."""
        return self.mlp

    @property
    def last_latent(self) -> torch.Tensor | None:
        """Most recently computed 3D latent vector z_t."""
        return self._last_latent

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute flattened 1D observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            shape = obs[obs_group].shape
            if len(shape) == 2:
                obs_dim += int(shape[-1])
            elif len(shape) == 3:
                obs_dim += int(shape[-2] * shape[-1])
            else:
                raise ValueError(
                    f"EncoderActorModel supports 2D (B, D) or 3D (B, H, D) observations, "
                    f"got shape {shape} for '{obs_group}'."
                )
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return input dimensionality of policy MLP pi_theta (25 + 3 = 28)."""
        return self.frame_obs_dim + self.latent_dim

    def _extract_flat_obs(self, obs: TensorDict) -> torch.Tensor:
        """Concatenate and flatten active observation groups along the feature dimension."""
        tensors = [obs[group].reshape(obs[group].shape[0], -1) for group in self.obs_groups]
        return torch.cat(tensors, dim=-1)

    def encode(self, obs_history: torch.Tensor) -> torch.Tensor:
        """Encode 5-frame observation history s_t (125D) into latent z_t (3D).

        Args:
            obs_history: Normalized observation history tensor of shape (B, 125).

        Returns:
            Latent tensor z_t of shape (B, 3).
        """
        return self.encoder(obs_history)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Normalize observations, run history encoder f_phi, and return [o_t; z_t].

        Args:
            obs: Observation TensorDict containing actor observation groups.
            masks: Unused sequence mask placeholder for MLPModel API compatibility.
            hidden_state: Unused recurrent hidden state placeholder.

        Returns:
            Concatenated actor input tensor [o_t; z_t] of shape (B, 28).
        """
        del masks, hidden_state
        flat_obs = self._extract_flat_obs(obs)
        norm_obs = self.obs_normalizer(flat_obs)
        if self.obs_dim == self.history_obs_dim:
            s_t = norm_obs
            o_t = s_t[..., -self.frame_obs_dim :]
        else:
            o_t = norm_obs[..., : self.frame_obs_dim]
            s_t = norm_obs[..., self.frame_obs_dim :]
        z_t = self.encoder(s_t)
        self._last_latent = z_t
        return torch.cat([o_t, z_t], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation normalization statistics from a batch of observations.

        Args:
            obs: Observation TensorDict sampled from environment rollouts.
        """
        if self.obs_normalization:
            flat_obs = self._extract_flat_obs(obs)
            self.obs_normalizer.update(flat_obs)  # type: ignore[operator]

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-exportable deterministic single-input single-output module.

        Returns:
            Wrapped nn.Module ready for torch.jit.script.
        """
        return _TorchEncoderActorModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-exportable deterministic single-input single-output module.

        Args:
            verbose: Whether to enable verbose export diagnostics.

        Returns:
            Wrapped nn.Module with get_dummy_inputs, input_names, and output_names.
        """
        return _OnnxEncoderActorModel(self, verbose=verbose)


class CriticModel(MLPModel):
    """Privileged value critic V_psi (144 -> 256 -> 128 -> 64 -> 1, 78,337 parameters)."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = CRITIC_HIDDEN_DIMS,
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        **_kwargs: object,
    ) -> None:
        """Initialize the privileged critic network.

        Args:
            obs: Sample observation TensorDict used to infer critic observation shape.
            obs_groups: Mapping from observation set name to active observation group keys.
            obs_set: Observation set key (typically "critic").
            output_dim: Scalar value output dimension (1).
            hidden_dims: Hidden layer dimensions for V_psi.
            activation: Activation function name for the critic MLP.
            obs_normalization: Whether to normalize privileged critic observations.
            distribution_cfg: Optional distribution configuration (None for deterministic scalar output).
            **_kwargs: Unused extra keyword arguments forwarded by runner factories.
        """
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )


class EncoderPPO(PPO):
    """PPO with explicit base linear velocity estimation supervision ||z_t - v_b||_2^2."""

    def __init__(self, *args: object, vel_loss_coef: float = 1.0, **kwargs: object) -> None:
        """Initialize EncoderPPO with velocity supervision loss coefficient.

        Args:
            *args: Positional arguments forwarded to rsl_rl.algorithms.PPO.
            vel_loss_coef: Weight on auxiliary encoder velocity estimation MSE loss.
            **kwargs: Keyword arguments forwarded to rsl_rl.algorithms.PPO.
        """
        super().__init__(*args, **kwargs)
        self.vel_loss_coef = float(vel_loss_coef)

    def update(self) -> dict[str, float]:
        """Run PPO optimization epochs with auxiliary encoder velocity estimation MSE loss.

        Returns:
            Dictionary of mean losses logged over the update epochs, including "vel_est".
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_vel_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            vel_loss = torch.zeros((), device=self.device)
            raw_actor = getattr(self, "_raw_actor", self.actor)
            last_latent = getattr(raw_actor, "last_latent", None)
            if self.vel_loss_coef > 0.0 and last_latent is not None and "critic" in batch.observations:
                z_t = last_latent[:original_batch_size]
                target_vel = batch.observations["critic"][:original_batch_size, :LATENT_DIM]
                vel_loss = nn.functional.mse_loss(z_t, target_vel)
                loss = loss + self.vel_loss_coef * vel_loss

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_vel_loss += vel_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_vel_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "vel_est": mean_vel_loss,
        }
        if self.rnd and mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry and mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict


class _TorchEncoderActorModel(nn.Module):
    """Single-input single-output TorchScript export wrapper for EncoderActorModel."""

    def __init__(self, model: EncoderActorModel) -> None:
        """Copy trained weights and normalizer buffers for TorchScript export.

        Args:
            model: Trained EncoderActorModel instance to export.
        """
        super().__init__()
        self.frame_obs_dim = model.frame_obs_dim
        self.history_obs_dim = model.history_obs_dim
        self.obs_dim = model.obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.actor = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from history observation ``obs`` of shape (B, 125).

        Args:
            obs: Flattened observation history tensor of shape (B, 125).

        Returns:
            Deterministic action tensor of shape (B, 6).
        """
        norm_obs = self.obs_normalizer(obs)
        if self.obs_dim == self.history_obs_dim:
            s_t = norm_obs
            o_t = s_t[:, -self.frame_obs_dim :]
        else:
            o_t = norm_obs[:, : self.frame_obs_dim]
            s_t = norm_obs[:, self.frame_obs_dim :]
        z_t = self.encoder(s_t)
        raw_actions = self.actor(torch.cat([o_t, z_t], dim=-1))
        return self.deterministic_output(raw_actions)

    @torch.jit.export
    def reset(self) -> None:
        """No-op reset hook for compatibility with recurrent export runners."""


class _OnnxEncoderActorModel(nn.Module):
    """Single-input single-output ONNX export wrapper for EncoderActorModel.

    Satisfies the deployment constraints in README.md: single input ``obs`` of
    shape (1, 125), single deterministic output ``actions`` of shape (1, 6), and
    in-graph empirical observation normalization.
    """

    is_recurrent: bool = False

    def __init__(self, model: EncoderActorModel, verbose: bool = False) -> None:
        """Copy trained weights and normalizer buffers for ONNX export.

        Args:
            model: Trained EncoderActorModel instance to export.
            verbose: Whether to enable verbose export diagnostics.
        """
        super().__init__()
        self.verbose = verbose
        self.input_size = model.obs_dim
        self.frame_obs_dim = model.frame_obs_dim
        self.history_obs_dim = model.history_obs_dim
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.actor = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX graph tracing.

        Args:
            obs: Flattened observation history tensor of shape (B, 125).

        Returns:
            Deterministic action tensor of shape (B, 6).
        """
        norm_obs = self.obs_normalizer(obs)
        if self.input_size == self.history_obs_dim:
            s_t = norm_obs
            o_t = s_t[:, -self.frame_obs_dim :]
        else:
            o_t = norm_obs[:, : self.frame_obs_dim]
            s_t = norm_obs[:, self.frame_obs_dim :]
        z_t = self.encoder(s_t)
        raw_actions = self.actor(torch.cat([o_t, z_t], dim=-1))
        return self.deterministic_output(raw_actions)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative single-input dummy tensor (1, 125) for ONNX tracing.

        Returns:
            Single-element tuple containing a zero tensor of shape (1, input_size).
        """
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        """Single ONNX input tensor name."""
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        """Single ONNX output tensor name."""
        return ["actions"]


class AsymmetricActorCritic(nn.Module):
    """Complete asymmetric actor-critic network container matching README.md.

    Bundles encoder f_phi (125 -> 128 -> 64 -> 3, 24,579 params), actor pi_theta
    (28 -> 128 -> 64 -> 32 -> 6, 14,246 params), and critic V_psi
    (144 -> 256 -> 128 -> 64 -> 1, 78,337 params).
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        history_length: int = HISTORY_LENGTH,
        latent_dim: int = LATENT_DIM,
        action_dim: int = ACTION_DIM,
        critic_obs_dim: int = CRITIC_OBS_DIM,
        encoder_hidden_dims: tuple[int, ...] | list[int] = ENCODER_HIDDEN_DIMS,
        actor_hidden_dims: tuple[int, ...] | list[int] = ACTOR_HIDDEN_DIMS,
        critic_hidden_dims: tuple[int, ...] | list[int] = CRITIC_HIDDEN_DIMS,
        activation: str = "elu",
        obs_normalization: bool = True,
        init_noise_std: float = 1.0,
    ) -> None:
        """Initialize the combined asymmetric actor-critic container.

        Args:
            obs_dim: Single-frame proprioceptive observation dimension.
            history_length: Number of stacked history frames.
            latent_dim: Output dimension of history encoder f_phi.
            action_dim: Action vector dimension output by actor pi_theta.
            critic_obs_dim: Privileged critic observation dimension.
            encoder_hidden_dims: Hidden layer dimensions for f_phi.
            actor_hidden_dims: Hidden layer dimensions for pi_theta.
            critic_hidden_dims: Hidden layer dimensions for V_psi.
            activation: Activation function name shared across all MLPs.
            obs_normalization: Whether to attach empirical observation normalization.
            init_noise_std: Initial exploration standard deviation for Gaussian actions.
        """
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.history_length = int(history_length)
        self.history_obs_dim = self.obs_dim * self.history_length
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.critic_obs_dim = int(critic_obs_dim)

        dummy_obs = TensorDict(
            {
                "policy": torch.zeros(1, self.history_obs_dim),
                "critic": torch.zeros(1, self.critic_obs_dim),
            },
            batch_size=[1],
        )
        obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        dist_cfg = {
            "class_name": "GaussianDistribution",
            "init_std": init_noise_std,
            "std_type": "scalar",
        }

        self.actor_model = EncoderActorModel(
            obs=dummy_obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            encoder_hidden_dims=encoder_hidden_dims,
            latent_dim=self.latent_dim,
            frame_obs_dim=self.obs_dim,
            history_length=self.history_length,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=dist_cfg,
        )
        self.critic_model = CriticModel(
            obs=dummy_obs,
            obs_groups=obs_groups,
            obs_set="critic",
            output_dim=1,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )

    @property
    def encoder(self) -> MLP:
        """History encoder MLP f_phi (125 -> 128 -> 64 -> 3)."""
        return self.actor_model.encoder

    @property
    def actor(self) -> MLP:
        """Policy actor MLP pi_theta (28 -> 128 -> 64 -> 32 -> 6)."""
        return self.actor_model.mlp

    @property
    def critic(self) -> MLP:
        """Value critic MLP V_psi (144 -> 256 -> 128 -> 64 -> 1)."""
        return self.critic_model.mlp

    @property
    def obs_normalizer(self) -> EmpiricalNormalization | nn.Identity:
        """Observation normalizer for the deployed actor/encoder input."""
        return self.actor_model.obs_normalizer

    def act_inference(
        self,
        obs_history: torch.Tensor,
        obs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute deterministic action mean a_t and latent z_t.

        Args:
            obs_history: Observation history tensor of shape (B, 125).
            obs: Optional pre-sliced current-frame observation tensor of shape (B, 25).

        Returns:
            Tuple (a_t, z_t) containing deterministic actions and latent estimates.
        """
        norm_s = self.actor_model.obs_normalizer(obs_history)
        o_t = norm_s[..., -self.obs_dim :] if obs is None else obs
        z_t = self.encoder(norm_s)
        a_t = self.actor(torch.cat([o_t, z_t], dim=-1))
        if self.actor_model.distribution is not None:
            a_t = self.actor_model.distribution.deterministic_output(a_t)
        return a_t, z_t

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Evaluate critic value V_psi(c_t).

        Args:
            critic_obs: Privileged critic observation tensor of shape (B, 144).

        Returns:
            Scalar value tensor of shape (B, 1).
        """
        norm_c = self.critic_model.obs_normalizer(critic_obs)
        return self.critic(norm_c)


def export_policy_to_onnx(
    policy: EncoderActorModel | AsymmetricActorCritic,
    path: str | os.PathLike[str],
    filename: str = "policy.onnx",
    opset_version: int = DEFAULT_ONNX_OPSET,
    fallback_opset_version: int = FALLBACK_ONNX_OPSET,
    verbose: bool = False,
) -> Path:
    """Export the deployed policy (obs_normalizer + encoder + actor) to ONNX.

    Args:
        policy: An EncoderActorModel or AsymmetricActorCritic instance.
        path: Directory path or full file path where the ONNX model will be written.
        filename: Output filename when ``path`` is a directory. Defaults to "policy.onnx".
        opset_version: Target ONNX opset version. Defaults to 18 per README.md.
        fallback_opset_version: Fallback opset version (17 for STM32 toolchains).
        verbose: Whether to enable verbose ONNX export logging.

    Returns:
        Resolved pathlib.Path to the exported ONNX file.

    Raises:
        TypeError: If ``policy`` is neither EncoderActorModel nor AsymmetricActorCritic.
    """
    if isinstance(policy, AsymmetricActorCritic):
        actor_model = policy.actor_model
    elif isinstance(policy, EncoderActorModel):
        actor_model = policy
    else:
        raise TypeError(f"Expected EncoderActorModel or AsymmetricActorCritic, got {type(policy).__name__}.")

    target_path = Path(path)
    if target_path.suffix.lower() != ".onnx":
        target_path = target_path / filename
    target_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_model = actor_model.as_onnx(verbose=verbose)
    onnx_model.to("cpu")
    onnx_model.eval()

    dummy_inputs = onnx_model.get_dummy_inputs()
    input_names = onnx_model.input_names
    output_names = onnx_model.output_names

    try:
        torch.onnx.export(
            onnx_model,
            dummy_inputs,
            str(target_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
        )
    except Exception:
        if opset_version == fallback_opset_version:
            raise
        torch.onnx.export(
            onnx_model,
            dummy_inputs,
            str(target_path),
            export_params=True,
            opset_version=fallback_opset_version,
            do_constant_folding=True,
            verbose=verbose,
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
        )

    return target_path
