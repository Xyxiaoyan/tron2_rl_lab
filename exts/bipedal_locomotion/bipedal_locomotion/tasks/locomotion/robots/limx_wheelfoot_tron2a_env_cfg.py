from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_tron2a_cfg import WHEELFOOT_TRON2A_CFG
from bipedal_locomotion.tasks.locomotion.cfg.WF_TRON2A.limx_base_env_cfg import WF_TRON2A_EnvCfg


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
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None


@configclass
class WF_TRON2A_BlindFlatEnvCfg_PLAY(WF_TRON2A_BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None