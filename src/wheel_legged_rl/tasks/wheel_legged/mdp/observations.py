# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the wheel-legged robot MDP matching ``README.md``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg

from wheel_legged_rl.models import CRITIC_OBS_DIM, OBS_DIM

from .rewards import leg_length

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

HEIGHT_SCAN_DIM: int = 77  # 11 x 7 local terrain height samples


def proprio_uniform_noise(data: torch.Tensor, cfg: ProprioUniformNoiseCfg) -> torch.Tensor:
    """Apply component-wise additive uniform noise ``[-u, u]`` to 25D proprioceptive observations.

    Args:
        data: Clean 25D proprioceptive observation tensor of shape ``(..., 25)``.
        cfg: Noise configuration specifying per-sensor uniform bounds.

    Returns:
        Corrupted observation tensor with the same shape and dtype as ``data``.

    Raises:
        ValueError: If the last dimension of ``data`` does not equal 25.
    """
    if data.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected proprio_obs last dimension {OBS_DIM}, got {data.shape[-1]}.")
    noise_scale = data.new_tensor(
        [
            cfg.ang_vel_noise,
            cfg.ang_vel_noise,
            cfg.ang_vel_noise,
            cfg.gravity_noise,
            cfg.gravity_noise,
            cfg.gravity_noise,
            0.0,
            0.0,
            0.0,
            cfg.joint_pos_noise,
            cfg.joint_pos_noise,
            cfg.joint_pos_noise,
            cfg.joint_pos_noise,
            cfg.joint_vel_noise,
            cfg.joint_vel_noise,
            cfg.joint_vel_noise,
            cfg.joint_vel_noise,
            cfg.joint_vel_noise,
            cfg.joint_vel_noise,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    return data + (torch.rand_like(data) * 2.0 - 1.0) * noise_scale


@configclass
class ProprioUniformNoiseCfg(NoiseCfg):
    """Additive uniform sensor noise for single-frame 25D proprioceptive observations."""

    func: Callable[[torch.Tensor, ProprioUniformNoiseCfg], torch.Tensor] = proprio_uniform_noise
    ang_vel_noise: float = 0.2
    gravity_noise: float = 0.05
    joint_pos_noise: float = 0.01
    joint_vel_noise: float = 0.5


def proprio_obs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_command",
    hip_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    joint_vel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Compute single-frame 25D proprioceptive observation ``o_t`` matching README.md.

    Index layout:
        - ``0:3``: Body angular velocity in body frame ``omega_b`` (3D).
        - ``3:6``: Projected gravity unit vector in body frame ``g_b`` (3D).
        - ``6:9``: Commands ``(v_x^cmd, yaw_rate^cmd, h^cmd)`` (3D).
        - ``9:13``: 4 hip joint positions relative to default (4D).
        - ``13:19``: 6 actuated joint velocities, 4 hips + 2 wheels (6D).
        - ``19:25``: Previous action ``a_{t-1}`` (6D).

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing velocity and leg-length targets.
        hip_asset_cfg: SceneEntityCfg selecting the 4 active hip joints in canonical order.
        joint_vel_asset_cfg: SceneEntityCfg selecting all 6 actuated joints (4 hips + 2 wheels).

    Returns:
        Proprioceptive observation tensor of shape ``(num_envs, 25)``.

    Raises:
        RuntimeError: If the concatenated observation dimension differs from 25.
    """
    asset: Articulation = env.scene[hip_asset_cfg.name]

    omega_b = asset.data.root_ang_vel_b.torch[:, :3]
    g_b = asset.data.projected_gravity_b.torch[:, :3]

    if hasattr(env, "command_manager") and command_name in env.command_manager.active_terms:
        cmd = env.command_manager.get_command(command_name)[:, :3]
    else:
        cmd = torch.zeros(env.num_envs, 3, device=env.device, dtype=omega_b.dtype)

    hip_ids = hip_asset_cfg.joint_ids
    q_hip = asset.data.joint_pos.torch[:, hip_ids] - asset.data.default_joint_pos.torch[:, hip_ids]

    vel_ids = joint_vel_asset_cfg.joint_ids
    dq = asset.data.joint_vel.torch[:, vel_ids] - asset.data.default_joint_vel.torch[:, vel_ids]

    last_act = env.action_manager.action[:, :6]

    obs = torch.cat((omega_b, g_b, cmd, q_hip, dq, last_act), dim=-1)
    if obs.shape[-1] != OBS_DIM:
        raise RuntimeError(f"Expected proprio_obs dimension {OBS_DIM}, got {obs.shape[-1]}.")
    return obs


def critic_obs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_command",
    hip_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    passive_knee_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    joint_vel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Compute 144D privileged critic observation ``c_t`` for value network V_psi.

    Index layout:
        - ``0:3`` (3D): Base linear velocity in body frame ``v_b``.
        - ``3:28`` (25D): Current proprioceptive observation ``o_t``.
        - ``28:34`` (6D): 6 actuated joint positions relative to default.
        - ``34:40`` (6D): 6 actuated joint velocities.
        - ``40:46`` (6D): 6 actuated joint accelerations.
        - ``46:52`` (6D): 6 actuated applied joint torques.
        - ``52:56`` (4D): 4 passive knee joint positions relative to default.
        - ``56:60`` (4D): 4 passive knee joint velocities.
        - ``60:62`` (2D): Left and right virtual leg lengths ``L0(q_hip)``.
        - ``62:65`` (3D): Base root position in world frame.
        - ``65:67`` (2D): Left and right leg-length tracking errors ``h^cmd - L0``.
        - ``67:144`` (77D): Terrain height scan features (11 x 7 grid; zeros on flat plane).

    Args:
        env: Active manager-based RL environment.
        command_name: Name of the command term providing velocity and leg-length targets.
        hip_asset_cfg: SceneEntityCfg selecting the 4 active hip joints.
        passive_knee_asset_cfg: SceneEntityCfg selecting the 4 passive knee joints.
        joint_vel_asset_cfg: SceneEntityCfg selecting all 6 actuated joints.

    Returns:
        Privileged critic observation tensor of shape ``(num_envs, 144)``.

    Raises:
        RuntimeError: If the concatenated observation dimension differs from 144.
    """
    asset: Articulation = env.scene[hip_asset_cfg.name]

    base_lin_vel = asset.data.root_lin_vel_b.torch[:, :3]
    o_t = proprio_obs(
        env,
        command_name=command_name,
        hip_asset_cfg=hip_asset_cfg,
        joint_vel_asset_cfg=joint_vel_asset_cfg,
    )

    act_ids = joint_vel_asset_cfg.joint_ids
    knee_ids = passive_knee_asset_cfg.joint_ids

    act_pos = asset.data.joint_pos.torch[:, act_ids] - asset.data.default_joint_pos.torch[:, act_ids]
    act_vel = asset.data.joint_vel.torch[:, act_ids]
    act_acc = asset.data.joint_acc.torch[:, act_ids]
    act_torque = asset.data.applied_torque.torch[:, act_ids]

    knee_pos = asset.data.joint_pos.torch[:, knee_ids] - asset.data.default_joint_pos.torch[:, knee_ids]
    knee_vel = asset.data.joint_vel.torch[:, knee_ids]

    hip_abs_pos = asset.data.joint_pos.torch[:, hip_asset_cfg.joint_ids]
    l0 = leg_length(hip_abs_pos)
    h_cmd = o_t[:, 8:9]
    l0_error = h_cmd - l0

    root_pos = asset.data.root_pos_w.torch[:, :3]
    height_scan = torch.zeros(env.num_envs, HEIGHT_SCAN_DIM, device=env.device, dtype=base_lin_vel.dtype)

    priv_obs = torch.cat(
        (
            base_lin_vel,
            o_t,
            act_pos,
            act_vel,
            act_acc,
            act_torque,
            knee_pos,
            knee_vel,
            l0,
            root_pos,
            l0_error,
            height_scan,
        ),
        dim=-1,
    )
    if priv_obs.shape[-1] != CRITIC_OBS_DIM:
        raise RuntimeError(f"Expected critic_obs dimension {CRITIC_OBS_DIM}, got {priv_obs.shape[-1]}.")
    return priv_obs
