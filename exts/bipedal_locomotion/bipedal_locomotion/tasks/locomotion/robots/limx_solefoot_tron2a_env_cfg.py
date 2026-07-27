import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.solefoot_tron2a_cfg import SOLEFOOT_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion.cfg.SF_TRON2A.limx_base_env_cfg import SF_TRON2A_EnvCfg
from bipedal_locomotion.tasks.locomotion import mdp


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
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class SF_TRON2A_BlindFlatEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


def _disable_height_scanner(cfg):
    """Keep observation dimensions identical across all curriculum stages."""
    cfg.scene.height_scanner = None
    cfg.observations.critic.height_scan = None
    cfg.curriculum.terrain_levels = None


def _configure_stand_still(cfg):
    """Configure stage 1 for stable standing on a plane."""
    _disable_height_scanner(cfg)
    cfg.episode_length_s = 10.0

    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    cfg.rewards.progress = None
    cfg.rewards.track_lin_vel_x_exp = None
    cfg.rewards.track_lin_vel_y_exp = None
    cfg.rewards.terrain_orientation_penalty = None
    cfg.rewards.swing_foot_height = None
    cfg.rewards.foot_landing_softness = None
    cfg.rewards.gait_reward = None
    cfg.rewards.feet_air_time = None
    cfg.rewards.lateral_deviation = None
    cfg.rewards.keep_balance.weight = 5.0
    cfg.rewards.stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-5.0,
        params={"lin_threshold": 0.05, "ang_threshold": 0.05},
    )
    cfg.rewards.stand_still_reg = RewTerm(
        func=mdp.stand_still_reg,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot"), "exclude_joints_name": []},
    )

    # Zero-perturbation reset for Stage 1: start from the known standing pose.
    cfg.events.reset_robot_base.params["pose_range"]["x"] = (0.0, 0.0)
    cfg.events.reset_robot_base.params["pose_range"]["y"] = (0.0, 0.0)
    cfg.events.reset_robot_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    for k in ("x", "y", "z", "roll", "pitch", "yaw"):
        cfg.events.reset_robot_base.params["velocity_range"][k] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def _configure_flat_walk(cfg):
    """Configure stage 2 for omnidirectional walking on a plane."""
    _disable_height_scanner(cfg)
    cfg.episode_length_s = 20.0

    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    # Retain a small standing fraction so stage 2 does not forget stage 1.
    cfg.commands.base_velocity.rel_standing_envs = 0.1
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
    cfg.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

    # Pure +x progress conflicts with commanded backward motion; velocity tracking
    # is the task signal for omnidirectional locomotion.
    cfg.rewards.progress = None
    cfg.rewards.track_lin_vel_x_exp.weight = 10.0
    cfg.rewards.track_lin_vel_y_exp.weight = 8.0
    cfg.rewards.keep_balance.weight = 1.0
    cfg.rewards.base_height_adaptive = None
    cfg.rewards.terrain_orientation_penalty = None
    cfg.rewards.swing_foot_height = None
    cfg.rewards.foot_landing_softness = None
    cfg.rewards.lateral_deviation = None
    cfg.rewards.base_height_exp = RewTerm(
        func=mdp.base_height_exp,
        weight=0.5,
        params={"std": math.sqrt(0.02), "target_height": 0.72},
    )
    # Lateral travel is intentional in this stage, so do not terminate on y.
    cfg.terminations.lateral_deviation_termination = None

    # --- Gait v2: loosen posture and gait forcing terms for a more natural walk. ---
    # Keep gait_reward but reduce its dominance; drop feet_air_time entirely so
    # the policy is not paid to lift feet every step.
    cfg.rewards.gait_reward.weight = 0.3
    cfg.rewards.feet_air_time = None
    # Let knees/joints move — the strong deviation and orientation penalties from
    # the terrain-crossing config produced a robotic, stiff-leg gait on flat ground.
    cfg.rewards.knee_joint_orientation.weight = -0.1
    cfg.rewards.orientation_exp_knee = None
    cfg.rewards.joint_deviation_l1.weight = -0.05
    cfg.rewards.action_rate_l2.weight = -0.001


def _configure_terrain_walk(cfg):
    """Configure stage 3 while preserving checkpoint-compatible observations."""
    _disable_height_scanner(cfg)
    cfg.episode_length_s = 20.0


############################
# Stage 1: Stable standing
############################


@configclass
class SF_TRON2A_StandStillEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_stand_still(self)


@configclass
class SF_TRON2A_StandStillEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        _configure_stand_still(self)


####################################
# Stage 2: Flat omnidirectional walk
####################################


@configclass
class SF_TRON2A_FlatWalkEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_flat_walk(self)


@configclass
class SF_TRON2A_FlatWalkEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        _configure_flat_walk(self)


#############################
# Stage 3: Terrain locomotion
#############################


@configclass
class SF_TRON2A_TerrainWalkEnvCfg(SF_TRON2A_BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_terrain_walk(self)


@configclass
class SF_TRON2A_TerrainWalkEnvCfg_PLAY(SF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        _configure_terrain_walk(self)
