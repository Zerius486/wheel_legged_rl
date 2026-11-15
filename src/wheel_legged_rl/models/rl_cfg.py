# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for the wheel-legged RL actor-critic models."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoAlgorithmCfg

from .actor_critic import (
    ACTOR_HIDDEN_DIMS,
    CRITIC_HIDDEN_DIMS,
    ENCODER_HIDDEN_DIMS,
    HISTORY_LENGTH,
    LATENT_DIM,
    OBS_DIM,
)


@configclass
class RslRlEncoderActorModelCfg(RslRlMLPModelCfg):
    """Configuration for wheel_legged_rl.models.EncoderActorModel."""

    class_name: str = "wheel_legged_rl.models:EncoderActorModel"
    """Qualified model class path resolved by rsl_rl.utils.resolve_callable."""

    hidden_dims: list[int] = list(ACTOR_HIDDEN_DIMS)
    """Actor MLP hidden dimensions [128, 64, 32] (28 -> 128 -> 64 -> 32 -> 6, 14,246 params)."""

    encoder_hidden_dims: list[int] = list(ENCODER_HIDDEN_DIMS)
    """History encoder MLP hidden dimensions [128, 64] (125 -> 128 -> 64 -> 3, 24,579 params)."""

    latent_dim: int = LATENT_DIM
    """Latent dimension z_t in R^3 produced by the history encoder."""

    frame_obs_dim: int = OBS_DIM
    """Single-frame proprioceptive observation dimension o_t in R^25."""

    history_length: int = HISTORY_LENGTH
    """Number of stacked observation frames (5 frames at 100 Hz = 50 ms)."""

    activation: str = "elu"
    """Activation function for both encoder and actor MLPs."""

    obs_normalization: bool = True
    """Whether to include empirical observation normalization inside the model and exported ONNX graph."""


@configclass
class RslRlCriticModelCfg(RslRlMLPModelCfg):
    """Configuration for wheel_legged_rl.models.CriticModel."""

    class_name: str = "wheel_legged_rl.models:CriticModel"
    """Qualified model class path resolved by rsl_rl.utils.resolve_callable."""

    hidden_dims: list[int] = list(CRITIC_HIDDEN_DIMS)
    """Critic MLP hidden dimensions [256, 128, 64] (144 -> 256 -> 128 -> 64 -> 1, 78,337 params)."""

    activation: str = "elu"
    """Activation function for the critic MLP."""

    obs_normalization: bool = True
    """Whether to normalize privileged critic observations."""


@configclass
class RslRlEncoderPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for wheel_legged_rl.models.EncoderPPO with velocity supervision."""

    class_name: str = "wheel_legged_rl.models:EncoderPPO"
    """Qualified algorithm class path resolved by rsl_rl.utils.resolve_callable."""

    vel_loss_coef: float = 1.0
    """Coefficient for the encoder base linear velocity estimation MSE loss."""
