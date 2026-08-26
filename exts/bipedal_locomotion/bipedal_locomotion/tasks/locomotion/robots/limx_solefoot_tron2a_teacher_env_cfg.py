"""Unified teacher and multi-teacher environments for SF_TRON2A."""

from __future__ import annotations

import math
import os
import sys

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from bipedal_locomotion.tasks.locomotion import mdp

from .limx_solefoot_tron2a_env_cfg import SF_TRON2A_BaseEnvCfg


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../../training_terrain"))
from teacher_training_terrain import (  # noqa: E402
    CONTINUOUS_TEACHER_TERRAIN_CFG,
    GAP_TEACHER_TERRAIN_CFG,
    MULTI_TEACHER_TERRAIN_CFG,
    OBSTACLE_TEACHER_TERRAIN_CFG,
    STAIRS_TEACHER_TERRAIN_CFG,
)


@configclass
class TeacherIdObsCfg(ObsGroup):
    """Training-only metadata consumed by ``MultiTeacherRunner``."""

    teacher_id = ObsTerm(func=mdp.terrain_skill_id)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


def _set_unified_teacher_interface(cfg: SF_TRON2A_BaseEnvCfg) -> None:
    """Apply the exact observation, command and control contract shared by all teachers."""
    cfg.scene.env_spacing = 10.0
    cfg.episode_length_s = 24.0
    cfg.events.reset_robot_base.func = mdp.reset_root_state_uniform
    cfg.events.reset_robot_base.params["pose_range"] = {
        "x": (-0.15, 0.15),
        "y": (-0.12, 0.12),
        "yaw": (-0.08, 0.08),
    }
    cfg.events.reset_robot_base.params["velocity_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }
    cfg.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)

    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.70)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.commands.gait_command.ranges.frequencies = (0.75, 1.00)
    cfg.commands.gait_command.ranges.offsets = (0.5, 0.5)
    cfg.commands.gait_command.ranges.durations = (0.5, 0.5)
    cfg.commands.gait_command.ranges.swing_height = (0.12, 0.22)

    # Every teacher sees the same forward-looking grid.  In particular, the
    # continuous teacher keeps the scanner enabled on flat terrain.
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        # Keep the scan coordinates identical to the deployed Camp task, not
        # just the same tensor dimension.
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        # Match the existing Camp policy shape exactly so the distilled student
        # can be loaded into the standard Camp PPO runner.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    cfg.terminations.knee_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="knee_.*"),
            "threshold": 1.0,
        },
    )

    # Common reward skeleton.  Specialist configs below only add the contact
    # strategy shaping that is unique to their terrain family.
    cfg.rewards.track_lin_vel_x_exp.weight = 1.5
    cfg.rewards.track_lin_vel_y_exp.weight = 0.4
    cfg.rewards.track_ang_vel_z_exp.weight = 0.3
    cfg.rewards.keep_balance.weight = 0.1
    cfg.rewards.flat_orientation_l2.weight = -1.5
    cfg.rewards.ang_vel_xy_l2.weight = -0.15
    cfg.rewards.lin_vel_z_l2.weight = -0.10
    cfg.rewards.feet_air_time.weight = 0.1
    cfg.rewards.forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=1.5,
        params={"heading_target": 0.0},
    )
    cfg.rewards.forward_stagnation = RewTerm(
        func=mdp.forward_stagnation,
        weight=-0.4,
        params={"min_forward_speed": 0.08, "command_name": "base_velocity"},
    )
    cfg.rewards.heading_alignment = RewTerm(
        func=mdp.heading_alignment_exp,
        weight=1.0,
        params={"std": 0.35, "target_heading": 0.0},
    )
    cfg.rewards.corridor_penalty = RewTerm(
        func=mdp.corridor_penalty,
        weight=-1.5,
        params={"corridor_half_width": 0.8},
    )
    cfg.rewards.base_height_exp = RewTerm(
        func=mdp.support_foot_adaptive_height,
        weight=0.6,
        params={
            "std": math.sqrt(0.05),
            "stand_height": 0.60,
            "foot_radius": 0.074,
            "contact_threshold": 1.0,
            "foot_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
        },
    )


def _set_teacher_curriculum(cfg: SF_TRON2A_BaseEnvCfg, up: float, down: float) -> None:
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.curriculum.terrain_levels = CurrTerm(
        func=mdp.terrain_levels_forward,
        params={"move_up_distance": up, "move_down_distance": down},
    )


def _configure_play(cfg: SF_TRON2A_BaseEnvCfg) -> None:
    cfg.scene.num_envs = 32
    cfg.scene.terrain.max_init_terrain_level = 9
    cfg.curriculum.terrain_levels = None
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.55, 0.55)


@configclass
class SF_TRON2A_ContinuousTeacherEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = CONTINUOUS_TEACHER_TERRAIN_CFG
        _set_unified_teacher_interface(self)
        _set_teacher_curriculum(self, up=12.0, down=4.0)


@configclass
class SF_TRON2A_ContinuousTeacherEnvCfg_PLAY(SF_TRON2A_ContinuousTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class SF_TRON2A_StairsTeacherEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = STAIRS_TEACHER_TERRAIN_CFG
        _set_unified_teacher_interface(self)
        _set_teacher_curriculum(self, up=10.0, down=3.0)
        self.rewards.gait_reward.weight = 0.35
        self.rewards.gait_reward.params["tracking_contacts_shaped_height"] = 0.2
        self.rewards.foot_joint_orientation.weight = -0.02


@configclass
class SF_TRON2A_StairsTeacherEnvCfg_PLAY(SF_TRON2A_StairsTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class SF_TRON2A_ObstacleTeacherEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = OBSTACLE_TEACHER_TERRAIN_CFG
        _set_unified_teacher_interface(self)
        _set_teacher_curriculum(self, up=10.0, down=3.0)
        self.rewards.gait_reward.weight = 0.35
        self.rewards.gait_reward.params["tracking_contacts_shaped_height"] = 0.25
        self.rewards.touchdown_step_length = RewTerm(
            func=mdp.touchdown_forward_step_length,
            weight=0.4,
            params={
                "target_length": 0.36,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.swing_foot_forward_tracking = RewTerm(
            func=mdp.swing_foot_forward_tracking,
            weight=0.4,
            params={
                "target_length": 0.36,
                "std": 0.25,
                "command_name": "base_velocity",
                "gait_command_name": "gait_command",
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )


@configclass
class SF_TRON2A_ObstacleTeacherEnvCfg_PLAY(SF_TRON2A_ObstacleTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class SF_TRON2A_GapTeacherEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = GAP_TEACHER_TERRAIN_CFG
        _set_unified_teacher_interface(self)
        _set_teacher_curriculum(self, up=7.5, down=3.0)
        self.rewards.gait_reward.weight = 0.35
        self.rewards.gait_reward.params["tracking_contacts_shaped_height"] = 0.2
        self.rewards.touchdown_step_length = RewTerm(
            func=mdp.touchdown_forward_step_length,
            weight=0.8,
            params={
                "target_length": 0.50,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.swing_foot_forward_tracking = RewTerm(
            func=mdp.swing_foot_forward_tracking,
            weight=1.0,
            params={
                "target_length": 0.50,
                "std": 0.25,
                "command_name": "base_velocity",
                "gait_command_name": "gait_command",
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.base_height_exp.func = mdp.highest_support_foot_height


@configclass
class SF_TRON2A_GapTeacherEnvCfg_PLAY(SF_TRON2A_GapTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_play(self)


@configclass
class SF_TRON2A_MultiTeacherEnvCfg(SF_TRON2A_BaseEnvCfg):
    """Balanced four-family environment used only for online distillation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = MULTI_TEACHER_TERRAIN_CFG
        _set_unified_teacher_interface(self)
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 9
        self.curriculum.terrain_levels = None
        self.observations.teacher = TeacherIdObsCfg()
