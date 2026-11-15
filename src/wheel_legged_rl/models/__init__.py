# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural network models and RSL-RL configurations for wheel-legged locomotion."""

from .actor_critic import (
    ACTION_DIM,
    ACTOR_HIDDEN_DIMS,
    ACTOR_INPUT_DIM,
    ACTOR_NUM_PARAMS,
    CRITIC_HIDDEN_DIMS,
    CRITIC_NUM_PARAMS,
    CRITIC_OBS_DIM,
    DEFAULT_ONNX_OPSET,
    DEPLOYED_MACS,
    DEPLOYED_NUM_PARAMS,
    ENCODER_HIDDEN_DIMS,
    ENCODER_NUM_PARAMS,
    FALLBACK_ONNX_OPSET,
    HISTORY_LENGTH,
    HISTORY_OBS_DIM,
    LATENT_DIM,
    OBS_DIM,
    AsymmetricActorCritic,
    CriticModel,
    EncoderActorModel,
    EncoderPPO,
    count_linear_macs,
    count_parameters,
    export_policy_to_onnx,
)
from .rl_cfg import RslRlCriticModelCfg, RslRlEncoderActorModelCfg, RslRlEncoderPpoAlgorithmCfg

__all__ = [
    "ACTION_DIM",
    "ACTOR_HIDDEN_DIMS",
    "ACTOR_INPUT_DIM",
    "ACTOR_NUM_PARAMS",
    "CRITIC_HIDDEN_DIMS",
    "CRITIC_NUM_PARAMS",
    "CRITIC_OBS_DIM",
    "DEFAULT_ONNX_OPSET",
    "DEPLOYED_MACS",
    "DEPLOYED_NUM_PARAMS",
    "ENCODER_HIDDEN_DIMS",
    "ENCODER_NUM_PARAMS",
    "FALLBACK_ONNX_OPSET",
    "HISTORY_LENGTH",
    "HISTORY_OBS_DIM",
    "LATENT_DIM",
    "OBS_DIM",
    "AsymmetricActorCritic",
    "CriticModel",
    "EncoderActorModel",
    "EncoderPPO",
    "RslRlCriticModelCfg",
    "RslRlEncoderActorModelCfg",
    "RslRlEncoderPpoAlgorithmCfg",
    "count_linear_macs",
    "count_parameters",
    "export_policy_to_onnx",
]
