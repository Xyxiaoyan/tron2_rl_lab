import sys
import os

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_tron2a_cfg import WHEELFOOT_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON2A.limx_base_env_cfg import WF_TRON2A_EnvCfg

# 将 training_terrain 加入搜索路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../../training_terrain"))
from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG


######################
# WF_TRON2A Base Environment
######################


@configclass
class WF_TRON2A_BaseEnvCfg(WF_TRON2A_EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = WHEELFOOT_TRON2A_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = ["base_Link"]

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WF_TRON2A_BaseEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg):
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
# WF_TRON2A Blind Flat Environment
############################


@configclass
class WF_TRON2A_BlindFlatEnvCfg(WF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class WF_TRON2A_BlindFlatEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


############################
# WF_TRON2A Camp Terrain Environment
############################


@configclass
class WF_TRON2A_CampEnvCfg(WF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 接入 Camp 训练地形
        self.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0

        # 启用地形难度分层（row 0 最简单 -> row 9 最难）
        self.scene.terrain.terrain_generator.curriculum = True

        # 随机起点：从地形 flat_patches 采样出生位置，自动获取正确 z 坐标
        self.events.reset_robot_base.func = mdp.reset_root_state_from_terrain
        self.events.reset_robot_base.params["pose_range"] = {"yaw": (-0.5, 0.5)}  # 只控制朝向

        # 前向偏置指令（评测要求穿越赛道，让机器人始终沿赛道前进）
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)  # 只向前
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)  # 减小侧向
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)     # 固定朝前

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
            weight=-1.0,
            params={"corridor_half_width": 2.0},
        )


@configclass
class WF_TRON2A_CampEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # 接入 Camp 训练地形
        self.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
        self.scene.env_spacing = 10.0

        # 恒定前进速度 0.5 m/s（评测场景）
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0  # 所有环境都前进，不站立

        # 配置 height_scanner（policy + critic 地形感知）
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
