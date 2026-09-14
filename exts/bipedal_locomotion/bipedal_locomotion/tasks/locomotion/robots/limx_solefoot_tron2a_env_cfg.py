import math
import sys
import os

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_tron2a_cfg import SOLEFOOT_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.limx_base_env_cfg import SF_TRON2A_EnvCfg

# 将 training_terrain 加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../../training_terrain"))
from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG
from stairs_training_terrain import STAIRS_TRAINING_TERRAIN_CFG
from gap_training_terrain import GAP_EVAL_TERRAIN_CFG, GAP_TRAINING_TERRAIN_CFG


def _camp_centerline_command(lin_vel_x: tuple[float, float]) -> mdp.CenterlineVelocityCommandCfg:
    """Create the Camp forward command with closed-loop centerline steering."""
    return mdp.CenterlineVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        debug_vis=False,
        resampling_time_range=(10.0, 10.0),
        lookahead_distance=2.0,
        centerline_deadband=0.05,
        ranges=mdp.CenterlineVelocityCommandCfg.Ranges(
            lin_vel_x=lin_vel_x,
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


######################
# SF_TRON2A Base Environment
######################


@configclass
class SF_TRON2A_BaseEnvCfg(SF_TRON2A_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_Link"]

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class SF_TRON2A_BaseEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# SF_TRON2A Blind Flat Environment
############################


@configclass
class SF_TRON2A_BlindFlatEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class SF_TRON2A_BlindFlatEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


############################
# SF_TRON2A Camp Terrain Environment
############################


@configclass
class SF_TRON2A_CampEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 接入 Camp 训练地形
        self.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0

        # 启用地形难度分层（row 0 最简单 -> row 9 最难）
        self.scene.terrain.terrain_generator.curriculum = True

        # 从地形 flat_patches 采样出生位置并朝向赛道 +X。教师只在接近
        # +X 的朝向上训练，因此 Camp 微调也保持相同的初始分布，避免策略
        # 在学会地形能力之前先处理大角度转向。
        self.events.reset_robot_base.func = mdp.reset_root_state_from_terrain
        self.events.reset_robot_base.params["pose_range"] = {"yaw": (-0.08, 0.08)}
        self.events.reset_robot_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)

        # 保持前向速度，同时根据相对赛道中心线的横向误差实时生成转向 command。
        # command 维度仍是 [vx, vy, wz]，不会改变现有策略网络的输入尺寸。
        self.commands.base_velocity = _camp_centerline_command((0.5, 1.0))
        # Camp 需同时兼顾沟壑长步和高台近距离落脚，保留更灵活的步频与抬脚高度。
        self.commands.gait_command.ranges.frequencies = (0.75, 1.00)
        self.commands.gait_command.ranges.swing_height = (0.20, 0.45)

        # Camp 专项任务奖励：降低停住的生存收益，并直接奖励触地步长。
        self.rewards.keep_balance.weight = 0.1
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=2.0,
            params={"heading_target": 0.0},
        )
        self.rewards.feet_air_time.weight = 0.1
        # 降低步长权重，鼓励机器人跨过沟壑时迈出更大步长，避免原地抬腿刷高度奖励。
        self.rewards.touchdown_step_length = RewTerm(
            func=mdp.touchdown_forward_step_length,
            weight=0.30,
            params={
                "target_length": 0.50,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        # 防止基座在摆动脚跨过沟壑之前先冲出双脚支撑区。
        self.rewards.base_projection_at_feet_midpoint = RewTerm(
            func=mdp.base_projection_at_feet_midpoint,
            weight=0.30, # 避免过度约束机器人在沟壑前伸直腿刷高度奖励
            params={
                "std": 0.20,
                "feet_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )

        # 配置 height_scanner（policy + critic 地形感知）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # 走廊惩罚：约束机器人不偏离赛道中心太远（评测赛道较窄，需提前适应）
        self.rewards.corridor_penalty = RewTerm(
            func=mdp.corridor_penalty,
            weight=-3.0,
            params={"corridor_half_width": 1.0},
        )

        # 基于当前支撑脚高度调整 base 目标，避免在障碍前伸直腿刷高度奖励。
        self.rewards.base_height_exp = RewTerm(
            func=mdp.support_foot_adaptive_height,
            weight=1.0,
            params={
                "std": math.sqrt(0.05),
                "stand_height": 0.60,
                "foot_radius": 0.074,
                "contact_threshold": 1.0,
                "foot_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            },
        )


@configclass
class SF_TRON2A_CampDistillFineTuneEnvCfg(SF_TRON2A_CampEnvCfg):
    """Camp adaptation task that minimizes changes to a distilled gait."""

    def __post_init__(self):
        super().__post_init__()

        # Make command following the dominant new behavior. In particular, the
        # yaw term teaches the policy to obey CenterlineVelocityCommand instead
        # of learning an unrelated fixed-world-heading behavior.
        self.rewards.track_lin_vel_x_exp.weight = 4.0
        self.rewards.track_lin_vel_y_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.forward_progress.weight = 1.0
        self.rewards.keep_balance.weight = 0.2

        # The distilled specialists already contain useful foot trajectories.
        # Retain only mild gait timing/support shaping and remove terms that
        # would push PPO to invent a new long-step gait throughout Camp.
        self.rewards.gait_reward.weight = 0.2
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.touchdown_step_length = None
        self.rewards.base_projection_at_feet_midpoint.weight = 0.15


@configclass
class SF_TRON2A_CampEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # Play 保持固定赛道起点出生，不使用 Camp 训练环境的 flat-patch
        # 随机采样。terrain origin 已由地形生成器设置在起始平地内。
        self.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0
        self.events.reset_robot_base.func = mdp.reset_root_state_uniform
        self.events.reset_robot_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)
        self.curriculum.terrain_levels = None


        # 恒定前进速度 1.0 m/s，并使用与 Camp 训练一致的中心线纠偏 command。
        self.commands.base_velocity = _camp_centerline_command((1.0, 1.0))
        self.commands.gait_command.ranges.frequencies = (0.75, 1.00)
        self.commands.gait_command.ranges.swing_height = (0.20, 0.45)

        # 与蒸馏教师和 Camp 训练任务保持完全相同的地形观测坐标。
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class SF_TRON2A_CampDistillFineTuneEnvCfg_PLAY(SF_TRON2A_CampEnvCfg_PLAY):
    """Play configuration for checkpoints produced by Camp distillation fine-tuning."""

    pass

############################
# SF_TRON2A Gap Terrain Environment
############################


@configclass
class SF_TRON2A_GapEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = GAP_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_gaps,
            params={"move_up_distance": 7.5, "move_down_distance": 3.0},
        )

        # Fixed approach position: the first gap is roughly two metres ahead.
        self.events.reset_robot_base.func = mdp.reset_root_state_uniform
        self.events.reset_robot_base.params["pose_range"] = {
            "x": (-0.10, 0.10),
            "y": (-0.10, 0.10),
            "yaw": (-0.05, 0.05),
        }
        self.events.reset_robot_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)

        # Kneeling on the gap edge is a failed crossing, not a recoverable pose.
        self.terminations.knee_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="knee_.*"),
                "threshold": 1.0,
            },
        )

        # Twenty seconds is enough to cross several gaps and gives the terrain
        # curriculum much faster feedback than the 100-second general task.
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.40, 0.70)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.gait_command.ranges.frequencies = (0.65, 0.85)
        self.commands.gait_command.ranges.swing_height = (0.12, 0.20)

        # Gap-specific shaping: the swing-foot trajectory and support geometry
        # dominate; forward progress remains useful but cannot reward a body dive.
        self.rewards.track_lin_vel_x_exp.weight = 2.0
        self.rewards.track_lin_vel_y_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.3
        self.rewards.keep_balance.weight = 0.05
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=1.5,
            params={"heading_target": 0.0},
        )
        self.rewards.forward_stagnation = RewTerm(
            func=mdp.forward_stagnation,
            weight=-0.5,
            params={"min_forward_speed": 0.08, "command_name": "base_velocity"},
        )
        self.rewards.heading_alignment = RewTerm(
            func=mdp.heading_alignment_exp,
            weight=1.0,
            params={"std": 0.35, "target_heading": 0.0},
        )
        self.rewards.gait_reward.weight = 0.4
        self.rewards.gait_reward.params["tracking_contacts_shaped_height"] = 0.2
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.touchdown_step_length = RewTerm(
            func=mdp.touchdown_forward_step_length,
            weight=1.0,
            params={
                "target_length": 0.50,
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.swing_foot_forward_tracking = RewTerm(
            func=mdp.swing_foot_forward_tracking,
            weight=1.5,
            params={
                "target_length": 0.50,
                "std": 0.25,
                "command_name": "base_velocity",
                "gait_command_name": "gait_command",
                "asset_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.base_projection_at_feet_midpoint = RewTerm(
            func=mdp.base_projection_at_feet_midpoint,
            weight=0.3,
            params={
                "std": 0.20,
                "feet_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.25
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.corridor_penalty = RewTerm(
            func=mdp.corridor_penalty,
            weight=-2.0,
            params={"corridor_half_width": 0.8},
        )
        self.rewards.base_height_exp = RewTerm(
            func=mdp.highest_support_foot_height,
            weight=1.25,
            params={
                "std": math.sqrt(0.02),
                "stand_height": 0.60,
                "foot_radius": 0.074,
                "contact_threshold": 1.0,
                "foot_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            },
        )

        # Shift the standard 1.6 m scan forward: [-0.45, 1.15] m relative
        # to the base instead of [-0.8, 0.8] m.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.35, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )


@configclass
class SF_TRON2A_GapEnvCfg_PLAY(SF_TRON2A_GapEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.terrain = GAP_EVAL_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.commands.base_velocity.ranges.lin_vel_x = (0.60, 0.60)


############################
# SF_TRON2A Stairs Terrain Environment
############################


@configclass
class SF_TRON2A_StairsEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 接入楼梯专项训练地形
        self.scene.terrain = STAIRS_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0

        # 启用地形难度分层（row 0 最简单 -> row 9 最难）
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 1
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.terrain_levels_stairs,
            params={"move_up_distance": 7.5, "move_down_distance": 2.5},
        )

        # 始终从上楼入口附近出生，确保课程进展相对 terrain origin 可测。
        self.events.reset_robot_base.func = mdp.reset_root_state_uniform
        self.events.reset_robot_base.params["pose_range"] = {
            "x": (-0.2, 0.2),
            "y": (-0.2, 0.2),
            "yaw": (-0.1, 0.1),
        }
        self.events.reset_robot_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)

        # 前向偏置指令（专注前向穿越楼梯）
        self.commands.base_velocity.ranges.lin_vel_x = (0.25, 0.60)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.gait_command.ranges.swing_height = (0.12, 0.26)

        # 楼梯专项奖励：只在机身正对楼梯时给完整前进奖励，避免侧身投机。
        self.rewards.track_lin_vel_x_exp.weight = 1.0
        self.rewards.track_lin_vel_y_exp.weight = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.keep_balance.weight = 0.1
        self.rewards.forward_progress = RewTerm(
            func=mdp.forward_progress,
            weight=2.0,
            params={"heading_target": 0.0},
        )
        self.rewards.heading_alignment = RewTerm(
            func=mdp.heading_alignment_exp,
            weight=1.2,
            params={"std": 0.35, "target_heading": 0.0},
        )
        self.rewards.gait_reward.weight = 0.3
        self.rewards.gait_reward.params["tracking_contacts_shaped_height"] = 0.2
        self.rewards.lin_vel_z_l2.weight = -0.1
        self.rewards.foot_joint_orientation.weight = -0.02

        # 配置 height_scanner（policy + critic 地形感知）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # 走廊惩罚：约束机器人不偏离赛道中心太远
        self.rewards.corridor_penalty = RewTerm(
            func=mdp.corridor_penalty,
            weight=-1.0,
            params={"corridor_half_width": 1.0},
        )

        # 只有支撑脚真正踩上台阶后才提高 base 目标，防止原地伸腿投机。
        self.rewards.base_height_exp = RewTerm(
            func=mdp.support_foot_adaptive_height,
            weight=0.5,
            params={
                "std": math.sqrt(0.05),
                "stand_height": 0.60,
                "foot_radius": 0.074,
                "contact_threshold": 1.0,
                "foot_cfg": SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
            },
        )


@configclass
class SF_TRON2A_StairsEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # 接入楼梯专项训练地形
        self.scene.terrain = STAIRS_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0

        # 恒定前进速度 0.5 m/s（评测场景）
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # 配置 height_scanner（policy + critic 地形感知）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
