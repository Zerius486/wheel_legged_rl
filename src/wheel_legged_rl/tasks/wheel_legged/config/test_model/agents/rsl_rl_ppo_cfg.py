# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO runner configuration for the test-model v1.0 environment."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from wheel_legged_rl.models import (
    ACTOR_HIDDEN_DIMS,
    CRITIC_HIDDEN_DIMS,
    ENCODER_HIDDEN_DIMS,
    HISTORY_LENGTH,
    LATENT_DIM,
    OBS_DIM,
    RslRlCriticModelCfg,
    RslRlEncoderActorModelCfg,
    RslRlEncoderPpoAlgorithmCfg,
)


@configclass
class WheelLeggedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Asymmetric actor-critic PPO runner configuration for ``test-model v1.0``."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "test_model_v1_0"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlEncoderActorModelCfg(
        hidden_dims=list(ACTOR_HIDDEN_DIMS),
        encoder_hidden_dims=list(ENCODER_HIDDEN_DIMS),
        latent_dim=LATENT_DIM,
        frame_obs_dim=OBS_DIM,
        history_length=HISTORY_LENGTH,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlEncoderActorModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlCriticModelCfg(
        hidden_dims=list(CRITIC_HIDDEN_DIMS),
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlEncoderPpoAlgorithmCfg(
        value_loss_coef=1.0,
        vel_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class WheelLeggedJumpPPORunnerCfg(WheelLeggedPPORunnerCfg):
    """Asymmetric actor-critic PPO runner configuration for ``test-model v1.0`` jump training."""

    num_steps_per_env = 48
    max_iterations = 2000
    experiment_name = "test_model_v1_0_jump"
    algorithm = RslRlEncoderPpoAlgorithmCfg(
        value_loss_coef=1.0,
        vel_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
