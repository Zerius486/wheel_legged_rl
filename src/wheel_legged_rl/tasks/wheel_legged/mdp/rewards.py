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

# Linkage constants aligned with docs/robot_params.md (test-model v1.0 parallel 5-bar leg).
C_HIP: float = 0.0945  # half hip spacing along body Y axis [m] (full spacing 0.189 m)
L1: float = 0.210  # thigh length, hip axis to knee axis [m]
L2: float = 0.250  # shank length, knee axis to wheel axle [m]
THETA1_0: float = math.atan2(0.09801029, 0.18572555)  # CAD zero front thigh splay 0.48557355 rad (27.8213 deg)
THETA2_0: float = math.atan2(0.09615449, 0.18669310)  # CAD zero rear thigh splay 0.47560742 rad (27.2503 deg)
L0_MIN: float = 0.1631  # minimum leg length at 19.88 deg hip limit [m]
L0_MAX: float = 0.4410  # maximum leg length at -27.25 deg hip limit [m]
L0_DEFAULT: float = 0.3468  # virtual leg length at CAD zero pose (q = 0) [m]


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
