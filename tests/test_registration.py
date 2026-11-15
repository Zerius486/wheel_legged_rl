# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the registered tasks and asset paths."""

import math

import gymnasium as gym
import torch

import wheel_legged_rl.tasks  # noqa: F401
from wheel_legged_rl.assets import (
    GAIN_RANDOMIZATION_RANGE,
    HIP_DAMPING,
    HIP_EFFORT_LIMIT,
    HIP_JOINT_NAMES,
    HIP_STIFFNESS,
    HIP_VELOCITY_LIMIT,
    MODEL_NAME,
    TEST_MODEL_CFG,
    TEST_MODEL_USD_PATH,
    WHEEL_DAMPING,
    WHEEL_EFFORT_LIMIT,
    WHEEL_GEAR_RATIO,
    WHEEL_JOINT_NAMES,
    WHEEL_MOTOR_CONT_TORQUE,
    WHEEL_MOTOR_NO_LOAD_RPM,
    WHEEL_VELOCITY_LIMIT,
)
from wheel_legged_rl.tasks.wheel_legged import mdp
from wheel_legged_rl.tasks.wheel_legged.config.test_model.agents.rsl_rl_ppo_cfg import (
    WheelLeggedPPORunnerCfg,
)
from wheel_legged_rl.tasks.wheel_legged.config.test_model.env_cfg import WheelLeggedEnvCfg


def test_task_registrations() -> None:
    """Verify that registered tasks expose valid environment and agent entry points."""
    assert MODEL_NAME == "test-model v1.0"
    assert TEST_MODEL_USD_PATH.is_file()

    expected = {
        "WheelLeggedRl-Wheel-Legged-TestModel-v1": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "wheel_legged_rl.tasks.wheel_legged.config.test_model.env_cfg:WheelLeggedEnvCfg",
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "wheel_legged_rl.tasks.wheel_legged.config.test_model.agents.rsl_rl_ppo_cfg:WheelLeggedPPORunnerCfg"
            ),
        },
    }

    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        assert spec.kwargs["env_cfg_entry_point"] == expected_values["env_cfg_entry_point"]
        assert spec.kwargs["default_agent"] == expected_values["default_agent"]
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == expected_values["rsl_rl_cfg_entry_point"]


def test_actuator_specs_and_gain_randomization_cfg() -> None:
    """Verify actuator torque/speed limits, baseline gains, and kp/kd domain randomization terms."""
    assert math.isclose(WHEEL_GEAR_RATIO, 268.0 / 17.0, rel_tol=1e-9)
    assert math.isclose(WHEEL_EFFORT_LIMIT, 0.16 * (268.0 / 17.0), rel_tol=1e-9)
    assert math.isclose(WHEEL_VELOCITY_LIMIT, (9400.0 * 2.0 * math.pi / 60.0) / (268.0 / 17.0), rel_tol=1e-9)
    assert WHEEL_MOTOR_CONT_TORQUE == 0.16
    assert WHEEL_MOTOR_NO_LOAD_RPM == 9400.0

    hip_cfg = TEST_MODEL_CFG.actuators["hips"]
    assert hip_cfg.effort_limit_sim == HIP_EFFORT_LIMIT == 20.0
    assert math.isclose(hip_cfg.velocity_limit_sim, HIP_VELOCITY_LIMIT, rel_tol=1e-9)
    assert hip_cfg.stiffness == HIP_STIFFNESS == 30.0
    assert hip_cfg.damping == HIP_DAMPING == 1.0

    wheel_cfg = TEST_MODEL_CFG.actuators["wheels"]
    assert math.isclose(wheel_cfg.effort_limit_sim, WHEEL_EFFORT_LIMIT, rel_tol=1e-9)
    assert math.isclose(wheel_cfg.velocity_limit_sim, WHEEL_VELOCITY_LIMIT, rel_tol=1e-9)
    assert wheel_cfg.stiffness == 0.0
    assert wheel_cfg.damping == WHEEL_DAMPING == 0.5

    env_cfg = WheelLeggedEnvCfg()
    hip_event = env_cfg.events.randomize_hip_gains
    assert hip_event.mode == "reset"
    assert hip_event.params["asset_cfg"].joint_names == list(HIP_JOINT_NAMES)
    assert hip_event.params["stiffness_distribution_params"] == GAIN_RANDOMIZATION_RANGE
    assert hip_event.params["damping_distribution_params"] == GAIN_RANDOMIZATION_RANGE
    assert hip_event.params["operation"] == "scale"
    assert hip_event.params["distribution"] == "uniform"

    wheel_event = env_cfg.events.randomize_wheel_gains
    assert wheel_event.mode == "reset"
    assert wheel_event.params["asset_cfg"].joint_names == list(WHEEL_JOINT_NAMES)
    assert "stiffness_distribution_params" not in wheel_event.params
    assert wheel_event.params["damping_distribution_params"] == GAIN_RANDOMIZATION_RANGE
    assert wheel_event.params["operation"] == "scale"
    assert wheel_event.params["distribution"] == "uniform"


def test_walking_training_cfg_regularization_and_supervision() -> None:
    """Verify regularization rewards, sensor noise, push perturbations, and training scale settings."""
    env_cfg = WheelLeggedEnvCfg()
    assert env_cfg.scene.num_envs == 4096
    assert env_cfg.episode_length_s == 20.0

    # Proprioceptive sensor corruption enabled with component-wise uniform noise
    assert env_cfg.observations.policy.enable_corruption is True
    assert isinstance(env_cfg.observations.policy.proprio.noise, mdp.ProprioUniformNoiseCfg)

    # Push perturbation interval event
    push_event = env_cfg.events.push_robot
    assert push_event.mode == "interval"
    assert push_event.interval_range_s == (5.0, 10.0)
    assert push_event.params["velocity_range"] == {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}

    # Tracking + regularization rewards
    rewards = env_cfg.rewards
    assert rewards.flat_orientation.weight == -5.0
    assert rewards.lin_vel_z.weight == -0.5
    assert rewards.ang_vel_xy.weight == -0.05
    assert rewards.leg_symmetry.weight == -1.0
    assert rewards.hip_coordination.weight == -0.5
    assert rewards.joint_torques.weight == -1.0e-4
    assert rewards.joint_acc.weight == -2.5e-7
    assert rewards.dof_pos_limits.weight == -2.0
    assert rewards.action_rate.weight == -0.01

    # PPO runner & encoder velocity supervision settings
    runner_cfg = WheelLeggedPPORunnerCfg()
    assert runner_cfg.max_iterations == 3000
    assert runner_cfg.save_interval == 100
    assert runner_cfg.algorithm.class_name == "wheel_legged_rl.models:EncoderPPO"
    assert runner_cfg.algorithm.vel_loss_coef == 1.0


def test_parallel_5bar_leg_kinematics() -> None:
    """Verify parallel 5-bar forward kinematics matches robot.usda zero pose, limits, and mirror signs."""
    l0_zero, theta0_zero = mdp.leg_kinematics(torch.zeros(1, 4))
    assert math.isclose(l0_zero[0, 0].item(), mdp.L0_DEFAULT, abs_tol=1e-3)
    assert math.isclose(l0_zero[0, 1].item(), mdp.L0_DEFAULT, abs_tol=1e-3)
    assert abs(theta0_zero[0, 0].item()) < 0.01
    assert abs(theta0_zero[0, 1].item()) < 0.01

    # Symmetric crouch with left/right mirror signs (+0.15, -0.15 vs -0.15, +0.15)
    q_crouch = torch.tensor([[0.15, -0.15, -0.15, 0.15]])
    l0_crouch = mdp.leg_length(q_crouch)
    assert math.isclose(l0_crouch[0, 0].item(), l0_crouch[0, 1].item(), rel_tol=1e-6)
    assert l0_crouch[0, 0].item() < mdp.L0_DEFAULT

    # Check stroke extrema
    delta_min = 0.3469
    delta_max = -0.4756
    q_extremes = torch.tensor(
        [
            [delta_min, -delta_min, -delta_min, delta_min],
            [delta_max, -delta_max, -delta_max, delta_max],
        ]
    )
    l0_ext = mdp.leg_length(q_extremes)
    assert math.isclose(l0_ext[0, 0].item(), mdp.L0_MIN, abs_tol=1e-3)
    assert math.isclose(l0_ext[1, 0].item(), mdp.L0_MAX, abs_tol=1e-3)
