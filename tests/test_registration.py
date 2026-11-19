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
    HIP_PEAK_EFFORT_LIMIT,
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
    WheelLeggedJumpPPORunnerCfg,
    WheelLeggedPPORunnerCfg,
)
from wheel_legged_rl.tasks.wheel_legged.config.test_model.env_cfg import WheelLeggedEnvCfg
from wheel_legged_rl.tasks.wheel_legged.config.test_model.jump_env_cfg import WheelLeggedJumpEnvCfg


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
        "WheelLeggedRl-Wheel-Legged-TestModel-Jump-v1": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": (
                "wheel_legged_rl.tasks.wheel_legged.config.test_model.jump_env_cfg:WheelLeggedJumpEnvCfg"
            ),
            "default_agent": "rsl_rl",
            "rsl_rl_cfg_entry_point": (
                "wheel_legged_rl.tasks.wheel_legged.config.test_model.agents.rsl_rl_ppo_cfg:WheelLeggedJumpPPORunnerCfg"
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


def test_jump_task_config_and_runner() -> None:
    """Verify jump task configuration, peak hip specs, terminations, and 2000-iteration runner."""
    env_cfg = WheelLeggedJumpEnvCfg()
    assert env_cfg.scene.num_envs == 4096
    assert env_cfg.scene.robot.spawn.activate_contact_sensors is True
    assert env_cfg.scene.robot.actuators["hips"].effort_limit_sim == HIP_PEAK_EFFORT_LIMIT == 40.0
    assert env_cfg.scene.robot.actuators["hips"].stiffness == 60.0
    assert env_cfg.scene.robot.actuators["hips"].damping == 1.5
    assert env_cfg.actions.hip_pos.scale == 0.5
    assert env_cfg.scene.contact_forces.prim_path == "{ENV_REGEX_NS}/Robot/.*"
    assert env_cfg.scene.contact_forces.track_air_time is True

    cmd_cfg = env_cfg.commands.base_command
    assert isinstance(cmd_cfg, mdp.WheelLeggedJumpCommandCfg)
    assert cmd_cfg.jump_height_cmd > mdp.JUMP_CMD_THRESHOLD > mdp.L0_MAX

    rewards = env_cfg.rewards
    assert rewards.lin_vel_z.weight == -1.5
    assert rewards.jump_upward_vel.weight == 2.5
    assert rewards.jump_takeoff_extension.weight == 1.0
    assert rewards.jump_flight.weight == 2.0
    assert rewards.uncommanded_flight.weight == -4.0
    assert rewards.jump_air_time.weight == 4.0
    assert rewards.jump_flight_height.weight == 6.0
    assert rewards.jump_leg_tuck.weight == 2.5
    assert rewards.undesired_contacts.weight == -2.0
    assert env_cfg.terminations.low_base_height.params["minimum_height"] == 0.18
    assert env_cfg.terminations.illegal_contact.params["threshold"] == 1.0

    runner_cfg = WheelLeggedJumpPPORunnerCfg()
    assert runner_cfg.num_steps_per_env == 48
    assert runner_cfg.max_iterations == 2000
    assert runner_cfg.experiment_name == "test_model_v1_0_jump"
    assert runner_cfg.algorithm.vel_loss_coef == 1.0
    assert runner_cfg.algorithm.entropy_coef == 0.01


def test_jump_command_and_reward_gating() -> None:
    """Verify jump pulse timing and stance/takeoff/flight reward gating."""

    class _TensorBox:
        def __init__(self, tensor: torch.Tensor) -> None:
            self.torch = tensor

    class _FakeRobot:
        def __init__(self, num_envs: int) -> None:
            self.data = type("Data", (), {})()
            self.data.root_lin_vel_b = _TensorBox(torch.zeros(num_envs, 3))
            self.data.root_ang_vel_b = _TensorBox(torch.zeros(num_envs, 3))
            self.data.root_lin_vel_w = _TensorBox(torch.tensor([[0.0, 0.0, 1.5], [0.0, 0.0, 1.5]]))
            self.data.root_pos_w = _TensorBox(torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.58]]))
            self.data.joint_pos = _TensorBox(
                torch.tensor(
                    [
                        [-0.35, 0.35, 0.35, -0.35],
                        [0.25, -0.25, -0.25, 0.25],
                    ]
                )
            )

    class _FakeContactSensor:
        def __init__(self, forces: torch.Tensor, air_time: torch.Tensor) -> None:
            self.data = type("SensorData", (), {})()
            self.data.net_normal_forces_w_history = _TensorBox(forces)
            self.data.current_air_time = _TensorBox(air_time)

    class _FakeScene(dict):
        def __init__(self, robot: _FakeRobot, contact_sensor: _FakeContactSensor) -> None:
            super().__init__(robot=robot)
            self.sensors = {"contact_forces": contact_sensor}

    class _FakeEnv:
        def __init__(
            self,
            num_envs: int,
            commands: torch.Tensor,
            forces: torch.Tensor,
            air_time: torch.Tensor,
        ) -> None:
            self.num_envs = num_envs
            self.device = "cpu"
            self.robot = _FakeRobot(num_envs)
            self.contact_sensor = _FakeContactSensor(forces, air_time)
            self.scene = _FakeScene(self.robot, self.contact_sensor)
            self.command_manager = type(
                "CmdMgr",
                (),
                {"get_command": staticmethod(lambda _name: commands)},
            )()

    # Env 0: stance (h_cmd=0.30, wheel contact force > 1N); Env 1: jump airborne (h_cmd=0.55, zero force).
    cmds = torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.55]])
    forces = torch.zeros(2, 3, 2, 3)
    forces[0, :, :, 2] = 90.0
    air_time = torch.tensor([[0.0, 0.0], [0.25, 0.30]])
    fake_env = _FakeEnv(num_envs=2, commands=cmds, forces=forces, air_time=air_time)

    from isaaclab.managers import SceneEntityCfg

    hip_cfg = SceneEntityCfg("robot")
    hip_cfg.joint_ids = [0, 1, 2, 3]
    wheel_cfg = SceneEntityCfg("contact_forces")
    wheel_cfg.body_ids = [0, 1]

    assert torch.allclose(mdp.jump_flight(fake_env, "base_command", wheel_cfg), torch.tensor([0.0, 1.0]))
    assert torch.allclose(mdp.uncommanded_flight(fake_env, "base_command", wheel_cfg), torch.tensor([0.0, 0.0]))
    cmds.copy_(torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.30]]))
    assert torch.allclose(mdp.uncommanded_flight(fake_env, "base_command", wheel_cfg), torch.tensor([0.0, 1.0]))
    cmds.copy_(torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.55]]))
    assert torch.allclose(
        mdp.jump_air_time(fake_env, "base_command", max_air_time=0.5, sensor_cfg=wheel_cfg),
        torch.tensor([0.0, 0.25]),
    )
    assert mdp.lin_vel_z_stance_l2(fake_env, "base_command", hip_cfg, wheel_cfg).tolist() == [2.25, 0.0]
    assert mdp.track_leg_length_stance(fake_env, 0.05, "base_command", hip_cfg, wheel_cfg)[1].item() == 0.0
    assert mdp.jump_upward_vel(fake_env, "base_command", asset_cfg=hip_cfg).tolist() == [0.0, 1.5]
    assert math.isclose(
        mdp.jump_flight_height(fake_env, 0.58, 0.15, "base_command", 0.35, hip_cfg, wheel_cfg)[1].item(),
        1.0,
        rel_tol=1e-6,
    )
    assert mdp.jump_leg_tuck(fake_env, 0.20, 0.06, "base_command", hip_cfg, wheel_cfg)[0].item() == 0.0
    assert mdp.jump_leg_tuck(fake_env, 0.20, 0.06, "base_command", hip_cfg, wheel_cfg)[1].item() > 0.5

    # Test WheelLeggedJumpCommand single-jump landing latch and cooldown.
    cmd_cfg = mdp.WheelLeggedJumpCommandCfg(
        asset_name="robot",
        jump_probability=1.0,
        jump_duration_s=0.5,
        cooldown_s=1.0,
        jump_height_cmd=0.55,
    )
    cmd_term = mdp.WheelLeggedJumpCommand(cmd_cfg, fake_env)
    cmd_term.reset()
    assert not torch.any(cmd_term.is_jump_active)
    assert torch.all(cmd_term.command[:, 2] < mdp.JUMP_CMD_THRESHOLD)

    cmd_term._resample([0, 1])
    cmd_term._update_command()
    assert torch.all(cmd_term.is_jump_active)
    assert torch.allclose(cmd_term.command[:, 2], torch.full((2,), 0.55))

    # Env 1 is airborne -> records flight; Env 0 remains grounded -> stays active.
    cmd_term.compute(dt=0.05)
    assert cmd_term.has_entered_flight.tolist() == [False, True]
    assert cmd_term.is_jump_active.tolist() == [True, True]

    # Env 1 touches down -> immediately clears jump pulse and enters cooldown.
    forces[1, :, :, 2] = 90.0
    cmd_term.compute(dt=0.05)
    assert cmd_term.is_jump_active.tolist() == [True, False]
    assert cmd_term.command[1, 2].item() < mdp.JUMP_CMD_THRESHOLD
    assert cmd_term.cooldown_left[1].item() > 0.9

    # Resampling Env 1 during cooldown does not trigger another jump.
    cmd_term._resample([1])
    cmd_term._update_command()
    assert cmd_term.is_jump_active[1].item() is False

    # Timeout expires remaining active Env 0 pulse.
    cmd_term.compute(dt=0.6)
    assert not torch.any(cmd_term.is_jump_active)
    assert torch.all(cmd_term.command[:, 2] < mdp.JUMP_CMD_THRESHOLD)
