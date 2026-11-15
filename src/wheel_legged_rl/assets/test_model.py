# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configuration for the test-model v1.0 parallel 5-bar wheel-legged robot."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

MODEL_NAME: str = "test-model v1.0"

TEST_MODEL_ASSET_DIR: Path = Path(__file__).resolve().parent / "test_model_v1.0"
TEST_MODEL_USD_PATH: Path = TEST_MODEL_ASSET_DIR / "robot.usda"

# Actuator specifications (baseline gains before reset-time domain randomization).
HIP_EFFORT_LIMIT: float = 20.0  # N·m
HIP_VELOCITY_LIMIT: float = 10.471975511965978  # 100 rpm in rad/s
HIP_STIFFNESS: float = 30.0  # N·m/rad
HIP_DAMPING: float = 1.0  # N·m·s/rad

WHEEL_MOTOR_CONT_TORQUE: float = 0.16  # bare motor max continuous torque [N·m]
WHEEL_MOTOR_NO_LOAD_RPM: float = 9400.0  # bare motor no-load speed [rpm]
WHEEL_GEAR_RATIO: float = 268.0 / 17.0  # gearbox reduction ratio (~15.7647:1)
WHEEL_EFFORT_LIMIT: float = WHEEL_MOTOR_CONT_TORQUE * WHEEL_GEAR_RATIO  # ~2.5224 N·m at wheel
WHEEL_VELOCITY_LIMIT: float = (WHEEL_MOTOR_NO_LOAD_RPM * 2.0 * math.pi / 60.0) / WHEEL_GEAR_RATIO  # ~62.4411 rad/s
WHEEL_DAMPING: float = 0.5  # N·m·s/rad

GAIN_RANDOMIZATION_RANGE: tuple[float, float] = (0.8, 1.2)

HIP_JOINT_NAMES: tuple[str, ...] = (
    "left_thleg1_body",
    "left_thleg2_body",
    "right_thleg1_body",
    "right_thleg2_body",
)
PASSIVE_KNEE_JOINT_NAMES: tuple[str, ...] = (
    "left_llleg1_th",
    "left_llleg2_thleg2",
    "right_llleg1_thleg1",
    "right_llleg2_thleg2",
)
WHEEL_JOINT_NAMES: tuple[str, ...] = (
    "left_wheel_llleg2",
    "right_wheel_llleg2",
)

TEST_MODEL_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TEST_MODEL_USD_PATH),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            "left_thleg1_body": 0.0,
            "left_llleg1_th": 0.0,
            "left_thleg2_body": 0.0,
            "left_llleg2_thleg2": 0.0,
            "left_wheel_llleg2": 0.0,
            "right_thleg1_body": 0.0,
            "right_llleg1_thleg1": 0.0,
            "right_thleg2_body": 0.0,
            "right_llleg2_thleg2": 0.0,
            "right_wheel_llleg2": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_thleg1_body", ".*_thleg2_body"],
            effort_limit_sim=HIP_EFFORT_LIMIT,
            velocity_limit_sim=HIP_VELOCITY_LIMIT,
            stiffness=HIP_STIFFNESS,
            damping=HIP_DAMPING,
        ),
        "knees_passive": ImplicitActuatorCfg(
            joint_names_expr=[".*_llleg1_.*", ".*_llleg2_.*"],
            effort_limit_sim=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_llleg2"],
            effort_limit_sim=WHEEL_EFFORT_LIMIT,
            velocity_limit_sim=WHEEL_VELOCITY_LIMIT,
            stiffness=0.0,
            damping=WHEEL_DAMPING,
        ),
    },
)
