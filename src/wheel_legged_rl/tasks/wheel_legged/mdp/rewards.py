# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

# 机构常数，与 docs/robot_params.md 对应
L1 = 0.210  # 大腿，髋轴心 ↔ 膝轴心 [m]
L2 = 0.250  # 小腿，膝轴心 ↔ 轮电机轴心 [m]
L0_MIN = 0.1356  # 腿长下限 [m]
L0_MAX = 0.3741  # 腿长上限 [m]


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def leg_length(joint_pos: torch.Tensor) -> torch.Tensor:
    """Leg length from the two hip angles of each leg.

    ``joint_pos`` must be laid out as ``[..., (alpha, gamma)]`` per leg, where ``alpha`` is the
    thigh angle from vertical and ``gamma`` the shank's absolute angle. The kernel below is
    invariant to a 2*pi offset and to the left/right sign convention, so no unwrapping is needed.

    Returns one leg length per leg, shape ``(num_envs, num_legs)``.
    """
    q = joint_pos.reshape(joint_pos.shape[0], -1, 2)
    knee = math.pi - (q[..., 1] - q[..., 0])
    return torch.sqrt(L1**2 + L2**2 - 2.0 * L1 * L2 * torch.cos(knee))


def track_lin_vel_x(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Cauchy kernel on the forward velocity error, ``1 / (1 + (e/std)^2)``.

    The body-x velocity is projected onto the horizontal plane, ``v_x * cos(pitch)``, so a pitched
    chassis on a slope is not credited for vertical motion.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    gravity_x = asset.data.projected_gravity_b.torch[:, 0]
    cos_pitch = torch.sqrt(torch.clamp(1.0 - gravity_x**2, min=0.0))
    error = command[:, 0] - asset.data.root_lin_vel_b.torch[:, 0] * cos_pitch
    return 1.0 / (1.0 + (error / std) ** 2)


def track_ang_vel_z(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Cauchy kernel on the yaw rate error, ``1 / (1 + (e/std)^2)``."""
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
    """Gaussian kernel on the leg length error, ``exp(-(e/std)^2)``, averaged over the two legs.

    ``asset_cfg.joint_names`` must list the four hip joints in the order
    ``[left_alpha, left_gamma, right_alpha, right_gamma]``. The command's third column is the
    target leg length in metres, bounded to ``[L0_MIN, L0_MAX]`` by the command term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, 2].unsqueeze(-1)
    error = command - leg_length(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    return torch.mean(torch.exp(-torch.square(error / std)), dim=-1)
