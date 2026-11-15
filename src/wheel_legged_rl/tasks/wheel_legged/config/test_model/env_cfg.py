# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the test-model v1.0 wheel-legged locomotion task."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from wheel_legged_rl.assets import (
    GAIN_RANDOMIZATION_RANGE,
    HIP_JOINT_NAMES,
    PASSIVE_KNEE_JOINT_NAMES,
    TEST_MODEL_CFG,
    WHEEL_JOINT_NAMES,
)
from wheel_legged_rl.models import HISTORY_LENGTH

from ... import mdp


@configclass
class WheelLeggedPhysicsCfg(PresetCfg):
    """Physics presets for the ``test-model v1.0`` wheel-legged robot environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=256,
            nconmax=64,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
            iterations=50,
            ls_iterations=50,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


@configclass
class WheelLeggedSceneCfg(InteractiveSceneCfg):
    """Configuration for the ``test-model v1.0`` robot scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = TEST_MODEL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class CommandsCfg:
    """Command specifications: ``(v_x^cmd, yaw_rate^cmd, h^cmd)`` in ``R^3``."""

    base_command = mdp.WheelLeggedCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
    )


@configclass
class ActionsCfg:
    """Action specifications: 4 hip joint position targets + 2 wheel velocity targets (``R^6``)."""

    hip_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(HIP_JOINT_NAMES),
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=list(WHEEL_JOINT_NAMES),
        scale=20.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Asymmetric observation specifications matching ``README.md``."""

    @configclass
    class PolicyCfg(ObsGroup):
        """5-frame flattened proprioceptive history ``s_t = [o_{t-4}; ...; o_t] in R^125``."""

        proprio = ObsTerm(
            func=mdp.proprio_obs,
            noise=mdp.ProprioUniformNoiseCfg(),
            history_length=HISTORY_LENGTH,
            flatten_history_dim=True,
            params={
                "command_name": "base_command",
                "hip_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(HIP_JOINT_NAMES),
                    preserve_order=True,
                ),
                "joint_vel_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(HIP_JOINT_NAMES + WHEEL_JOINT_NAMES),
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self) -> None:
            """Enable sensor noise and concatenate proprioceptive policy observations."""
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations ``c_t in R^144``."""

        privileged = ObsTerm(
            func=mdp.critic_obs,
            params={
                "command_name": "base_command",
                "hip_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(HIP_JOINT_NAMES),
                    preserve_order=True,
                ),
                "passive_knee_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(PASSIVE_KNEE_JOINT_NAMES),
                    preserve_order=True,
                ),
                "joint_vel_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(HIP_JOINT_NAMES + WHEEL_JOINT_NAMES),
                    preserve_order=True,
                ),
            },
        )

        def __post_init__(self) -> None:
            """Disable sensor noise and concatenate privileged critic observations."""
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for reset and domain randomization events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_hip_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
            "stiffness_distribution_params": GAIN_RANDOMIZATION_RANGE,
            "damping_distribution_params": GAIN_RANDOMIZATION_RANGE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_wheel_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(WHEEL_JOINT_NAMES),
                preserve_order=True,
            ),
            "damping_distribution_params": GAIN_RANDOMIZATION_RANGE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.3, 0.3)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the ``test-model v1.0`` environment matching ``README.md``."""

    track_lin_vel_x = RewTerm(
        func=mdp.track_lin_vel_x,
        weight=1.0,
        params={"std": 0.5, "command_name": "base_command"},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z,
        weight=1.0,
        params={"std": 0.5, "command_name": "base_command"},
    )
    track_leg_length = RewTerm(
        func=mdp.track_leg_length,
        weight=0.7,
        params={
            "std": 0.05,
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    track_leg_length_fine = RewTerm(
        func=mdp.track_leg_length,
        weight=0.3,
        params={
            "std": 0.02,
            "command_name": "base_command",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    leg_symmetry = RewTerm(
        func=mdp.leg_symmetry_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    hip_coordination = RewTerm(
        func=mdp.hip_coordination_l2,
        weight=-0.5,
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
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES + WHEEL_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES + WHEEL_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(HIP_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the ``test-model v1.0`` MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0)},
    )


@configclass
class WheelLeggedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the ``test-model v1.0`` wheel-legged robot RL environment."""

    scene: WheelLeggedSceneCfg = WheelLeggedSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Set simulation timing, physics solver preset, and default camera pose."""
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(2.5, 2.5, 1.5))
        self.sim.dt = 1 / 200  # T_s = dt * decimation = 10 ms (100 Hz)
        self.sim.render_interval = self.decimation
        self.sim.physics = WheelLeggedPhysicsCfg()
