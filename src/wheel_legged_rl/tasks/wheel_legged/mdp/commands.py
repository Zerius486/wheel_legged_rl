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

from .rewards import L0_DEFAULT, L0_MAX, L0_MIN

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


class WheelLeggedJumpCommand(WheelLeggedCommand):
    """Command generator adding periodic jump triggers onto ``h^cmd`` in ``R^3``.

    During stance/driving intervals, ``h^cmd`` holds a sampled target leg length inside
    ``[L0_MIN, L0_MAX]``. On periodic command resamples (excluding episode reset), a
    fixed-duration jump pulse sets ``h^cmd`` to ``jump_height_cmd`` (above ``L0_MAX``),
    signaling the policy to execute a jump while keeping ``OBS_DIM = 25`` unchanged.
    """

    cfg: WheelLeggedJumpCommandCfg

    def __init__(self, cfg: WheelLeggedJumpCommandCfg, env: ManagerBasedEnv) -> None:
        """Initialize the jump-capable wheel-legged command generator.

        Args:
            cfg: Jump command configuration.
            env: Owning manager-based environment.
        """
        super().__init__(cfg, env)
        self.nominal_height_cmd = torch.full((self.num_envs,), L0_DEFAULT, device=self.device)
        self.jump_time_left = torch.zeros(self.num_envs, device=self.device)
        self.cooldown_left = torch.zeros(self.num_envs, device=self.device)
        self.is_jump_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_entered_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._wheel_body_ids: list[int] | None = None
        self.metrics["jump_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset commands, suppress jump at t=0, and schedule an early first jump resample.

        Args:
            env_ids: Environment indices to reset, or None for all environments.

        Returns:
            Logged metric dictionary from the previous episode.
        """
        resolved_ids = list(range(self.num_envs)) if env_ids is None else env_ids
        extras = super().reset(resolved_ids)
        self.jump_time_left[resolved_ids] = 0.0
        self.cooldown_left[resolved_ids] = 0.0
        self.is_jump_active[resolved_ids] = False
        self.has_entered_flight[resolved_ids] = False
        self._command[resolved_ids, 2] = self.nominal_height_cmd[resolved_ids]
        self.time_left[resolved_ids] = self.time_left[resolved_ids].uniform_(*self.cfg.initial_jump_delay_range)
        return extras

    def _get_wheel_in_flight(self) -> torch.Tensor | None:
        """Return bool tensor ``(num_envs,)`` when all wheels lack contact, or None if unavailable."""
        sensors = getattr(self._env.scene, "sensors", None)
        if not sensors or self.cfg.contact_sensor_name not in sensors:
            return None
        sensor = sensors[self.cfg.contact_sensor_name]
        if self._wheel_body_ids is None:
            find_fn = getattr(sensor, "find_sensors", None) or getattr(sensor, "find_bodies", None)
            if callable(find_fn):
                self._wheel_body_ids, _ = find_fn(self.cfg.wheel_body_pattern)
            else:
                self._wheel_body_ids = [0, 1]
        forces = sensor.data.net_normal_forces_w_history.torch[:, :, self._wheel_body_ids]
        in_contact = torch.max(torch.linalg.norm(forces, dim=-1), dim=1)[0] > self.cfg.contact_force_threshold
        return ~torch.any(in_contact, dim=-1)

    def compute(self, dt: float) -> None:
        """Advance jump-pulse timers, clear completed jumps on landing, and update outputs.

        Args:
            dt: Policy step duration [s].
        """
        was_active = self.jump_time_left > 0.0
        self.jump_time_left = torch.clamp(self.jump_time_left - dt, min=0.0)
        self.cooldown_left = torch.clamp(self.cooldown_left - dt, min=0.0)

        in_flight = self._get_wheel_in_flight()
        if in_flight is not None:
            active_now = self.jump_time_left > 0.0
            self.has_entered_flight |= active_now & in_flight
            landed_after_jump = active_now & self.has_entered_flight & (~in_flight)
            if torch.any(landed_after_jump):
                self.jump_time_left[landed_after_jump] = 0.0
                self.has_entered_flight[landed_after_jump] = False
                self.cooldown_left[landed_after_jump] = self.cfg.cooldown_s

        timed_out = was_active & (self.jump_time_left <= 0.0)
        if torch.any(timed_out):
            self.has_entered_flight[timed_out] = False
            self.cooldown_left[timed_out] = torch.maximum(
                self.cooldown_left[timed_out],
                torch.full_like(self.cooldown_left[timed_out], self.cfg.cooldown_s),
            )

        super().compute(dt)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["jump_fraction"] = self.is_jump_active.float()

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        if not isinstance(env_ids, slice) and len(env_ids) == 0:
            return
        self.nominal_height_cmd[env_ids] = self._command[env_ids, 2].clone()
        num_resampled = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
        can_jump = (self.cooldown_left[env_ids] <= 0.0) & (~self.is_jump_active[env_ids])
        trigger = can_jump & (torch.rand(num_resampled, device=self.device) <= self.cfg.jump_probability)
        self.jump_time_left[env_ids] = torch.where(trigger, self.cfg.jump_duration_s, self.jump_time_left[env_ids])
        self.has_entered_flight[env_ids] &= ~trigger

    def _update_command(self) -> None:
        super()._update_command()
        self.is_jump_active = self.jump_time_left > 0.0
        self._command[:, 2] = torch.where(self.is_jump_active, self.cfg.jump_height_cmd, self.nominal_height_cmd)


@configclass
class WheelLeggedJumpCommandCfg(WheelLeggedCommandCfg):
    """Configuration for WheelLeggedJumpCommand."""

    class_type: type = WheelLeggedJumpCommand

    rel_standing_envs: float = 0.2
    """Fraction of environments sampled with zero linear and yaw velocity commands."""

    resampling_time_range: tuple[float, float] = (2.5, 4.5)
    """Time interval [s] between command resamples and candidate jump triggers."""

    initial_jump_delay_range: tuple[float, float] = (0.5, 1.5)
    """Initial delay [s] after episode reset before the first jump command candidate."""

    jump_probability: float = 0.70
    """Probability of initiating a jump pulse at each eligible command resample."""

    jump_duration_s: float = 1.0
    """Maximum timeout [s] for a single jump pulse before returning to stance height."""

    cooldown_s: float = 1.5
    """Minimum stance cooldown [s] after landing before another jump can be commanded."""

    jump_height_cmd: float = 0.55
    """Commanded ``h^cmd`` [m] during a jump pulse (strictly above ``L0_MAX``)."""

    contact_sensor_name: str = "contact_forces"
    """Scene contact sensor name used to detect post-flight touchdown."""

    wheel_body_pattern: str = ".*_wheel"
    """Regex matching wheel bodies on the contact sensor."""

    contact_force_threshold: float = 1.0
    """Normal force threshold [N] distinguishing wheel ground contact from flight."""

    @configclass
    class Ranges(WheelLeggedCommandCfg.Ranges):
        """Sampling ranges for jump-capable locomotion commands."""

        lin_vel_x: tuple[float, float] = (-1.5, 1.5)
        ang_vel_z: tuple[float, float] = (-1.5, 1.5)
        leg_length: tuple[float, float] = (0.25, 0.34)

    ranges: Ranges = Ranges()
