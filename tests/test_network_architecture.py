# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the asymmetric actor-critic network architecture defined in README.md."""

from __future__ import annotations

from pathlib import Path

import onnx
import pytest
import torch
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from wheel_legged_rl.models import (
    ACTION_DIM,
    ACTOR_INPUT_DIM,
    ACTOR_NUM_PARAMS,
    CRITIC_NUM_PARAMS,
    CRITIC_OBS_DIM,
    DEFAULT_ONNX_OPSET,
    DEPLOYED_MACS,
    DEPLOYED_NUM_PARAMS,
    ENCODER_NUM_PARAMS,
    FALLBACK_ONNX_OPSET,
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
from wheel_legged_rl.tasks.wheel_legged.config.test_model.agents.rsl_rl_ppo_cfg import (
    WheelLeggedPPORunnerCfg,
)


def test_readme_module_dimensions_and_parameter_counts() -> None:
    """Verify exact module structures, parameter counts, and MACs against README.md."""
    net = AsymmetricActorCritic()

    # encoder f_phi: 125 -> 128 -> 64 -> 3 (24,579 params)
    assert count_parameters(net.encoder) == ENCODER_NUM_PARAMS == 24_579
    # actor pi_theta: 28 -> 128 -> 64 -> 32 -> 6 (14,246 params)
    assert count_parameters(net.actor) == ACTOR_NUM_PARAMS == 14_246
    # critic V_psi: 144 -> 256 -> 128 -> 64 -> 1 (78,337 params)
    assert count_parameters(net.critic) == CRITIC_NUM_PARAMS == 78_337
    assert count_parameters(net.critic_model) == CRITIC_NUM_PARAMS == 78_337

    # Deployed total (encoder + actor): 38,825 params, 38,400 MACs
    deployed_params = count_parameters(net.encoder) + count_parameters(net.actor)
    deployed_macs = count_linear_macs(net.encoder) + count_linear_macs(net.actor)
    assert deployed_params == DEPLOYED_NUM_PARAMS == 38_825
    assert deployed_macs == DEPLOYED_MACS == 38_400

    # Memory footprint check from README: fp32 ~= 151.7 KB, int8 ~= 37.9 KB
    assert round(deployed_params * 4 / 1024, 1) == 151.7
    assert round(deployed_params * 1 / 1024, 1) == 37.9


def test_forward_shapes_and_history_encoding() -> None:
    """Verify s_t -> z_t -> a_t and critic value forward passes."""
    batch_size = 8
    net = AsymmetricActorCritic(obs_normalization=True)

    frames = [torch.randn(batch_size, OBS_DIM) for _ in range(5)]
    s_t = torch.cat(frames, dim=-1)  # [o_{t-4}; ...; o_t] in R^125
    c_t = torch.randn(batch_size, CRITIC_OBS_DIM)

    assert s_t.shape == (batch_size, HISTORY_OBS_DIM)
    assert torch.equal(s_t[:, -OBS_DIM:], frames[-1])

    obs_td = TensorDict({"policy": s_t, "critic": c_t}, batch_size=[batch_size])
    net.actor_model.update_normalization(obs_td)
    net.critic_model.update_normalization(obs_td)

    actor_in = net.actor_model.get_latent(obs_td)
    assert actor_in.shape == (batch_size, ACTOR_INPUT_DIM)
    assert net.actor_model.last_latent is not None
    assert net.actor_model.last_latent.shape == (batch_size, LATENT_DIM)

    actions_det = net.actor_model(obs_td, stochastic_output=False)
    actions_stoch = net.actor_model(obs_td, stochastic_output=True)
    values = net.critic_model(obs_td)

    assert actions_det.shape == (batch_size, ACTION_DIM)
    assert actions_stoch.shape == (batch_size, ACTION_DIM)
    assert values.shape == (batch_size, 1)

    a_inf, z_inf = net.act_inference(s_t)
    torch.testing.assert_close(a_inf, actions_det)
    torch.testing.assert_close(z_inf, net.actor_model.last_latent)


@pytest.mark.parametrize("opset_version", [DEFAULT_ONNX_OPSET, FALLBACK_ONNX_OPSET])
def test_jit_and_onnx_export_single_input_single_output(tmp_path: Path, opset_version: int) -> None:
    """Verify single-input single-output deterministic export with normalization included in graph.

    Args:
        tmp_path: Temporary directory fixture provided by pytest.
        opset_version: Target ONNX opset version under test.
    """
    net = AsymmetricActorCritic(obs_normalization=True)
    net.eval()

    sample_obs = TensorDict(
        {
            "policy": torch.randn(16, HISTORY_OBS_DIM) * 3.0 + 1.5,
            "critic": torch.randn(16, CRITIC_OBS_DIM),
        },
        batch_size=[16],
    )
    net.train()
    net.actor_model.update_normalization(sample_obs)
    net.eval()

    test_input = torch.randn(1, HISTORY_OBS_DIM)
    with torch.no_grad():
        expected_out = net.actor_model(
            TensorDict({"policy": test_input}, batch_size=[1]),
            stochastic_output=False,
        )

    # TorchScript export check
    jit_module = torch.jit.script(net.actor_model.as_jit().eval())
    with torch.no_grad():
        jit_out = jit_module(test_input)
    torch.testing.assert_close(jit_out, expected_out)

    # ONNX export check
    onnx_path = export_policy_to_onnx(
        net,
        tmp_path,
        filename=f"policy_opset{opset_version}.onnx",
        opset_version=opset_version,
    )
    assert onnx_path.is_file()

    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    # Single input ('obs': [1, 125]) and single output ('actions': [1, 6])
    assert [inp.name for inp in model_proto.graph.input] == ["obs"]
    assert [out.name for out in model_proto.graph.output] == ["actions"]

    onnx_wrapper = net.actor_model.as_onnx().eval()
    with torch.no_grad():
        wrapper_out = onnx_wrapper(test_input)
    torch.testing.assert_close(wrapper_out, expected_out)
    assert count_parameters(onnx_wrapper, trainable_only=False) == DEPLOYED_NUM_PARAMS


def test_rsl_rl_ppo_integration() -> None:
    """Verify RSL-RL PPO constructs and updates EncoderActorModel and CriticModel."""
    num_envs = 8
    obs = TensorDict(
        {
            "policy": torch.randn(num_envs, HISTORY_OBS_DIM),
            "critic": torch.randn(num_envs, CRITIC_OBS_DIM),
        },
        batch_size=[num_envs],
    )

    class _DummyVecEnv:
        """Minimal vectorized environment stub required by PPO.construct_algorithm."""

        def __init__(self) -> None:
            """Initialize dummy dimensions matching WheelLeggedPPORunnerCfg."""
            self.num_envs = num_envs
            self.num_actions = ACTION_DIM
            self.device = "cpu"

    runner_cfg = WheelLeggedPPORunnerCfg()
    cfg_dict = runner_cfg.to_dict()
    cfg_dict["num_steps_per_env"] = 4
    cfg_dict["multi_gpu"] = None

    alg = PPO.construct_algorithm(obs, _DummyVecEnv(), cfg_dict, device="cpu")
    assert isinstance(alg, EncoderPPO)
    assert alg.vel_loss_coef == 1.0
    assert isinstance(alg.actor, EncoderActorModel)
    assert isinstance(alg.critic, CriticModel)
    assert count_parameters(alg.actor.encoder) == ENCODER_NUM_PARAMS
    assert count_parameters(alg.actor.actor) == ACTOR_NUM_PARAMS
    assert count_parameters(alg.critic.mlp) == CRITIC_NUM_PARAMS

    for _ in range(4):
        actions = alg.act(obs)
        assert actions.shape == (num_envs, ACTION_DIM)
        rewards = torch.zeros(num_envs)
        dones = torch.zeros(num_envs, dtype=torch.long)
        alg.process_env_step(obs, rewards, dones, {})

    alg.compute_returns(obs)
    losses = alg.update()
    assert "value" in losses and "surrogate" in losses and "vel_est" in losses
    assert losses["vel_est"] >= 0.0
