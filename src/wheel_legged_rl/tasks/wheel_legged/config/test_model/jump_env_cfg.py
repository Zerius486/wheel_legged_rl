# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the test-model v1.0 wheel-legged jumping task."""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from wheel_legged_rl.assets import (
    HIP_JOINT_NAMES,
    HIP_PEAK_EFFORT_LIMIT,
    TEST_MODEL_CFG,
    WHEEL_JOINT_NAMES,
)

from ... import mdp
from .env_cfg import (
    ActionsCfg,
    RewardsCfg,
    TerminationsCfg,
    WheelLeggedEnvCfg,
    WheelLeggedSceneCfg,
)


@configclass
class WheelLeggedJumpSceneCfg(WheelLeggedSceneCfg):
    """Scene configuration enabling DM8009P peak hip torque and contact sensing."""

    robot: ArticulationCfg = TEST_MODEL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=TEST_MODEL_CFG.spawn.replace(activate_contact_sensors=True),
        actuators={
            **TEST_MODEL_CFG.actuators,
            "hips": TEST_MODEL_CFG.actuators["hips"].replace(
                effort_limit_sim=HIP_PEAK_EFFORT_LIMIT,
                stiffness=60.0,
                damping=1.5,
            ),
        },
    )

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class JumpCommandsCfg:
    """Jump-capable 3D command specifications ``(v_x^cmd, yaw_rate^cmd, h^cmd)``."""

    base_command = mdp.WheelLeggedJumpCommandCfg(asset_name="robot")


@configclass
class JumpActionsCfg(ActionsCfg):
    """Action specifications widening hip stroke scale to ``0.5 rad`` for takeoff."""

    hip_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(HIP_JOINT_NAMES),
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class JumpRewardsCfg(RewardsCfg):
    """Jump reward shaping adapted from open-source wheel-legged jump implementations."""

    track_leg_length = RewTerm(
        func=mdp.track_leg_length_stance,
        weight=0.7,
        params={
            "std": 0.05,
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    track_leg_length_fine = RewTerm(
        func=mdp.track_leg_length_stance,
        weight=0.3,
        params={
            "std": 0.02,
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-15.0)
    lin_vel_z = RewTerm(
        func=mdp.lin_vel_z_stance_l2,
        weight=-1.5,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    hip_coordination = RewTerm(
        func=mdp.hip_coordination_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES + WHEEL_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    jump_upward_vel = RewTerm(
        func=mdp.jump_upward_vel,
        weight=2.5,
        params={"command_name": "base_command", "max_vel_z": 2.5},
    )
    jump_takeoff_extension = RewTerm(
        func=mdp.jump_takeoff_extension,
        weight=1.0,
        params={
            "target_l0": 0.42,
            "std": 0.08,
            "command_name": "base_command",
            "min_vel_z": 0.20,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    jump_flight = RewTerm(
        func=mdp.jump_flight,
        weight=2.0,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    uncommanded_flight = RewTerm(
        func=mdp.uncommanded_flight,
        weight=-4.0,
        params={
            "command_name": "base_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    jump_air_time = RewTerm(
        func=mdp.jump_air_time,
        weight=4.0,
        params={
            "command_name": "base_command",
            "max_air_time": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    jump_flight_height = RewTerm(
        func=mdp.jump_flight_height,
        weight=6.0,
        params={
            "target_height": 0.55,
            "std": 0.18,
            "command_name": "base_command",
            "min_height": 0.34,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    jump_leg_tuck = RewTerm(
        func=mdp.jump_leg_tuck,
        weight=2.5,
        params={
            "target_l0": 0.20,
            "std": 0.08,
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_wheel"]),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*_thleg.*", ".*_llleg.*"]),
        },
    )


@configclass
class JumpTerminationsCfg(TerminationsCfg):
    """Terminations preventing belly-collapse reward hacking during jump training."""

    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.18},
    )
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*_thleg.*"]),
        },
    )


@configclass
class WheelLeggedJumpEnvCfg(WheelLeggedEnvCfg):
    """Configuration for the ``test-model v1.0`` jumping RL environment."""

    scene: WheelLeggedJumpSceneCfg = WheelLeggedJumpSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: JumpCommandsCfg = JumpCommandsCfg()
    actions: JumpActionsCfg = JumpActionsCfg()
    rewards: JumpRewardsCfg = JumpRewardsCfg()
    terminations: JumpTerminationsCfg = JumpTerminationsCfg()
