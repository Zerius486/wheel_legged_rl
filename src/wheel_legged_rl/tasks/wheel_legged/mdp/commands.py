# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generators for wheel-legged robot locomotion: (v_x^cmd, yaw_rate^cmd, h^cmd)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .rewards import L0_MAX, L0_MIN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class WheelLeggedCommand(CommandTerm):
    """Uniform command generator for ``(v_x^cmd, yaw_rate^cmd, h^cmd)`` in R^3.

    Columns of ``command``:
        - ``0``: Forward linear velocity ``v_x^cmd`` [m/s].
        - ``1``: Yaw angular velocity ``yaw_rate^cmd`` [rad/s].
        - ``2``: Target leg length / height ``h^cmd`` [m], bounded to ``[L0_MIN, L0_MAX]``.
    """

    cfg: WheelLeggedCommandCfg

    def __init__(self, cfg: WheelLeggedCommandCfg, env: ManagerBasedEnv) -> None:
        """Initialize the wheel-legged command generator.

        Args:
            cfg: Command configuration specifying sampling ranges and standing fraction.
            env: Owning manager-based environment.
        """
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Command tensor of shape ``(num_envs, 3)``: ``[v_x_cmd, ang_vel_z_cmd, height_cmd]``."""
        return self._command

    def _update_metrics(self) -> None:
        self.metrics["error_vel_x"] = torch.abs(self._command[:, 0] - self.robot.data.root_lin_vel_b.torch[:, 1])
        self.metrics["error_vel_yaw"] = torch.abs(self._command[:, 1] - self.robot.data.root_ang_vel_b.torch[:, 2])

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        r = torch.empty(len(env_ids), device=self.device)
        self._command[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self._command[env_ids, 1] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        h_min = max(self.cfg.ranges.leg_length[0], L0_MIN)
        h_max = min(self.cfg.ranges.leg_length[1], L0_MAX)
        self._command[env_ids, 2] = r.uniform_(h_min, h_max)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        standing_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_ids) > 0:
            self._command[standing_ids, :2] = 0.0


@configclass
class WheelLeggedCommandCfg(CommandTermCfg):
    """Configuration for WheelLeggedCommand."""

    class_type: type = WheelLeggedCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    rel_standing_envs: float = 0.05
    """Fraction of environments sampled with zero linear and yaw velocity commands."""

    @configclass
    class Ranges:
        """Sampling ranges for the 3D command."""

        lin_vel_x: tuple[float, float] = (-2.0, 2.0)
        """Forward linear velocity command range [m/s]."""

        ang_vel_z: tuple[float, float] = (-2.0, 2.0)
        """Yaw angular velocity command range [rad/s]."""

        leg_length: tuple[float, float] = (0.22, 0.36)
        """Target leg length / height command range [m], within ``[L0_MIN, L0_MAX]``."""

    ranges: Ranges = Ranges()
