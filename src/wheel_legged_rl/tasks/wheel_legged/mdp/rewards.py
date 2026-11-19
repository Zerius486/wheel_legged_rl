# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for wheel-legged locomotion matching ``README.md``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# Linkage constants aligned with docs/robot_params.md (test-model v1.0 parallel 5-bar leg).
C_HIP: float = 0.0945  # half hip spacing along body Y axis [m] (full spacing 0.189 m)
L1: float = 0.210  # thigh length, hip axis to knee axis [m]
L2: float = 0.250  # shank length, knee axis to wheel axle [m]
THETA1_0: float = math.atan2(0.09801029, 0.18572555)  # CAD zero front thigh splay 0.48557355 rad (27.8213 deg)
THETA2_0: float = math.atan2(0.09615449, 0.18669310)  # CAD zero rear thigh splay 0.47560742 rad (27.2503 deg)
L0_MIN: float = 0.1631  # minimum leg length at 19.88 deg hip limit [m]
L0_MAX: float = 0.4410  # maximum leg length at -27.25 deg hip limit [m]
L0_DEFAULT: float = 0.3468  # virtual leg length at CAD zero pose (q = 0) [m]
WHEEL_RADIUS: float = 0.055  # wheel outer radius measured from test-model v1.0 mesh [m]
JUMP_CMD_THRESHOLD: float = 0.45  # h^cmd >= threshold indicates an active jump request [m]
DEFAULT_CONTACT_FORCE_THRESHOLD: float = 1.0  # normal force threshold for wheel-ground contact [N]


def leg_kinematics(joint_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve parallel 5-bar forward kinematics from the 4 active hip angles.

    Accounts for the left-leg ``quat=(0, 0, 1, 0)`` mirror flip in ``robot.usda``,
    the ``2 * C_HIP = 0.189 m`` hip spacing, and the CAD zero-pose offsets.

    Args:
        joint_pos: Hip joint angles of shape ``(num_envs, 4)`` ordered as
            ``[left_thleg1, left_thleg2, right_thleg1, right_thleg2]``, or
            ``(num_envs, 2)`` for a single leg.

    Returns:
        Tuple ``(l0, theta0)`` of virtual leg length [m] and virtual leg fore-aft
        tilt angle [rad], each of shape ``(num_envs, num_legs)``.
    """
    q = joint_pos.reshape(joint_pos.shape[0], -1, 2)
    if q.shape[1] == 2:
        sign1 = q.new_tensor([1.0, -1.0])
        sign2 = q.new_tensor([-1.0, 1.0])
    else:
        sign1 = q.new_ones(q.shape[1])
        sign2 = -q.new_ones(q.shape[1])

    theta1 = THETA1_0 + sign1 * q[..., 0]
    theta2 = THETA2_0 + sign2 * q[..., 1]

    y_k1 = -C_HIP - L1 * torch.sin(theta1)
    z_k1 = -L1 * torch.cos(theta1)
    y_k2 = C_HIP + L1 * torch.sin(theta2)
    z_k2 = -L1 * torch.cos(theta2)

    dy = y_k2 - y_k1
    dz = z_k2 - z_k1
    d = torch.sqrt(torch.clamp(dy**2 + dz**2, min=1.0e-12))
    h = torch.sqrt(torch.clamp(L2**2 - 0.25 * d**2, min=0.0))

    y_w = 0.5 * (y_k1 + y_k2) + h * (dz / d)
    z_w = 0.5 * (z_k1 + z_k2) - h * (dy / d)

    l0 = torch.sqrt(y_w**2 + z_w**2)
    theta0 = torch.atan2(y_w, -z_w)
    return l0, theta0


def leg_length(joint_pos: torch.Tensor) -> torch.Tensor:
    """Compute virtual leg length ``L0`` from the parallel 5-bar hip joint angles.

    Args:
        joint_pos: Hip joint angles of shape ``(num_envs, 4)`` ordered as
            ``[left_thleg1, left_thleg2, right_thleg1, right_thleg2]``.

    Returns:
        Virtual leg lengths [m] of shape ``(num_envs, num_legs)``.
    """
    l0, _ = leg_kinematics(joint_pos)
    return l0


def track_lin_vel_x(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Cauchy kernel on the forward velocity error, ``1 / (1 + (e/std)^2)``.

    In ``robot.usda``, the wheel spin axis is body ``+X`` and the longitudinal rolling
    direction is body ``+Y`` (index 1). The longitudinal body velocity is projected onto
    the horizontal plane (``v_y * cos(pitch)``) so pitching on slopes is not rewarded.

    Args:
        env: Active manager-based RL environment.
        std: Kernel width parameter [m/s].
        command_name: Name of the command term providing velocity targets.
        asset_cfg: SceneEntityCfg selecting the robot articulation.

    Returns:
        Per-environment velocity tracking reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    gravity_y = asset.data.projected_gravity_b.torch[:, 1]
    cos_pitch = torch.sqrt(torch.clamp(1.0 - gravity_y**2, min=0.0))
    error = command[:, 0] - asset.data.root_lin_vel_b.torch[:, 1] * cos_pitch
    return 1.0 / (1.0 + (error / std) ** 2)


def track_ang_vel_z(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Cauchy kernel on the yaw rate error, ``1 / (1 + (e/std)^2)``.

    Args:
        env: Active manager-based RL environment.
        std: Kernel width parameter [rad/s].
        command_name: Name of the command term providing yaw-rate targets.
        asset_cfg: SceneEntityCfg selecting the robot articulation.

    Returns:
        Per-environment yaw-rate tracking reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = command[:, 1] - asset.data.root_ang_vel_b.torch[:, 2]
    return 1.0 / (1.0 + (error / std) ** 2)


def track_leg_length(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Gaussian kernel on leg-length error ``exp(-(e/std)^2)`` averaged across both legs.

    Args:
        env: Active manager-based RL environment.
        std: Kernel width parameter [m].
        command_name: Name of the command term providing target leg length in column 2.
        asset_cfg: SceneEntityCfg selecting the 4 hip joints in order
            ``[left_thleg1, left_thleg2, right_thleg1, right_thleg2]``.

    Returns:
        Per-environment leg-length tracking reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2].unsqueeze(-1)
    error = command - leg_length(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.mean(torch.exp(-torch.square(error / std)), dim=-1)


def leg_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """L2 penalty on left-right virtual leg-length difference ``(L0_left - L0_right)^2``.

    Args:
        env: Active manager-based RL environment.
        asset_cfg: SceneEntityCfg selecting the 4 active hip joints.

    Returns:
        Per-environment squared leg-length asymmetry of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    l0 = leg_length(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.square(l0[:, 0] - l0[:, 1])


def hip_coordination_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """L2 penalty on virtual-leg fore-aft tilt angle ``theta0^2`` summed over both legs.

    Args:
        env: Active manager-based RL environment.
        asset_cfg: SceneEntityCfg selecting the 4 active hip joints.

    Returns:
        Per-environment summed squared virtual-leg tilt of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    _, theta0 = leg_kinematics(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.sum(torch.square(theta0), dim=-1)


def _is_jump_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
) -> torch.Tensor:
    """Return a boolean mask indicating environments with an active jump command."""
    return env.command_manager.get_command(command_name)[:, 2] >= jump_cmd_threshold


def _wheel_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(in_flight, any_wheel_contact)`` boolean masks from wheel contact sensors."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids]
    in_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > contact_force_threshold
    any_contact = torch.any(in_contact, dim=-1)
    return ~any_contact, any_contact


def _stance_mask(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    jump_cmd_threshold: float,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Return True for grounded environments not actively executing a jump."""
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    in_flight, _ = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    return ~(jump_cmd | in_flight)


def track_leg_length_stance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Gaussian leg-length tracking active only during non-jump ground stance.

    Args:
        env: Active manager-based RL environment.
        std: Kernel width parameter [m].
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        asset_cfg: SceneEntityCfg selecting the 4 active hip joints.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Per-environment stance leg-length reward of shape ``(num_envs,)``.
    """
    stance_mask = _stance_mask(env, command_name, sensor_cfg, jump_cmd_threshold, contact_force_threshold)
    return track_leg_length(env, std=std, command_name=command_name, asset_cfg=asset_cfg) * stance_mask


def lin_vel_z_stance_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """L2 penalty on world vertical velocity suppressed during jump commands and flight.

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        asset_cfg: SceneEntityCfg selecting the robot articulation.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Per-environment squared vertical velocity penalty of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    stance_mask = _stance_mask(env, command_name, sensor_cfg, jump_cmd_threshold, contact_force_threshold)
    return torch.square(asset.data.root_lin_vel_w.torch[:, 2]) * stance_mask


def jump_upward_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_vel_z: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
) -> torch.Tensor:
    """Reward positive upward base velocity while a jump command is active.

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        max_vel_z: Upper clamp on rewarded upward velocity [m/s].
        asset_cfg: SceneEntityCfg selecting the robot articulation.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].

    Returns:
        Per-environment upward takeoff velocity reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    vel_z = torch.clamp(asset.data.root_lin_vel_w.torch[:, 2], min=0.0, max=max_vel_z)
    return vel_z * jump_cmd


def jump_takeoff_extension(
    env: ManagerBasedRLEnv,
    target_l0: float,
    std: float,
    command_name: str,
    min_vel_z: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Reward extending both legs during grounded upward push-off under a jump command.

    Args:
        env: Active manager-based RL environment.
        target_l0: Desired extended virtual leg length during push-off [m].
        std: Gaussian kernel width [m].
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        min_vel_z: Minimum upward velocity [m/s] gating takeoff extension reward.
        asset_cfg: SceneEntityCfg selecting the 4 active hip joints.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Per-environment takeoff extension reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    _, any_contact = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    takeoff_mask = jump_cmd & any_contact & (asset.data.root_lin_vel_w.torch[:, 2] > min_vel_z)
    l0 = leg_length(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    reward = torch.mean(torch.exp(-torch.square((l0 - target_l0) / std)), dim=-1)
    return reward * takeoff_mask


def jump_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Reward full airborne state (both wheels off ground) while jump is commanded.

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Binary flight indicator of shape ``(num_envs,)`` cast to float.
    """
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    in_flight, _ = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    return (jump_cmd & in_flight).float()


def uncommanded_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Penalize being airborne when no jump command is active (prevents continuous bouncing).

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Binary indicator of shape ``(num_envs,)`` equal to 1.0 during unsolicited flight.
    """
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    in_flight, _ = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    return ((~jump_cmd) & in_flight).float()


def jump_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_air_time: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
) -> torch.Tensor:
    """Reward sustained dual-wheel airtime during a commanded jump.

    Mirrors open-source ``air_time`` / ``encourage_jump`` shaping by rewarding continuous
    airborne duration across both wheels rather than brief wheel-chatter micro-hops.

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        max_air_time: Upper clamp on rewarded airborne duration [s].
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].

    Returns:
        Per-environment airborne duration reward of shape ``(num_envs,)``.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    air_time = torch.min(contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids], dim=-1)[0]
    return torch.clamp(air_time, min=0.0, max=max_air_time) * jump_cmd


def jump_flight_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    command_name: str,
    min_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Reward base height approaching ``target_height`` during a commanded flight phase.

    Args:
        env: Active manager-based RL environment.
        target_height: Desired apex base world height [m].
        std: Gaussian kernel width [m].
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        min_height: Minimum base world height [m] required for flight-height credit.
        asset_cfg: SceneEntityCfg selecting the robot articulation.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Per-environment flight height reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    in_flight, _ = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    z = asset.data.root_pos_w.torch[:, 2]
    reward = torch.exp(-torch.square((z - target_height) / std))
    return reward * (jump_cmd & in_flight & (z > min_height))


def jump_leg_tuck(
    env: ManagerBasedRLEnv,
    target_l0: float,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    jump_cmd_threshold: float = JUMP_CMD_THRESHOLD,
    contact_force_threshold: float = DEFAULT_CONTACT_FORCE_THRESHOLD,
) -> torch.Tensor:
    """Reward tucking both virtual legs toward ``target_l0`` while airborne during a jump.

    Args:
        env: Active manager-based RL environment.
        target_l0: Desired tucked virtual leg length in flight [m].
        std: Gaussian kernel width [m].
        command_name: Name of the command term providing ``h^cmd`` in column 2.
        asset_cfg: SceneEntityCfg selecting the 4 active hip joints.
        sensor_cfg: SceneEntityCfg selecting the wheel bodies on the contact sensor.
        jump_cmd_threshold: Minimum ``h^cmd`` threshold identifying a jump command [m].
        contact_force_threshold: Normal force threshold for ground contact [N].

    Returns:
        Per-environment airborne leg-tuck reward of shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = _is_jump_commanded(env, command_name, jump_cmd_threshold)
    in_flight, _ = _wheel_contact_state(env, sensor_cfg, contact_force_threshold)
    l0 = leg_length(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    reward = torch.mean(torch.exp(-torch.square((l0 - target_l0) / std)), dim=-1)
    return reward * (jump_cmd & in_flight)
