import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    SF_TRON2AFlatPPORunnerCfg,
    SF_TRON2AFlatWalkPPORunnerCfg,
    SF_TRON2AStandStillPPORunnerCfg,
    SF_TRON2ATerrainWalkPPORunnerCfg,
    WF_TRON2AFlatPPORunnerCfg,
)

from . import limx_solefoot_tron2a_env_cfg, limx_wheelfoot_tron2a_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_sf_tron2a_blind_flat_runner_cfg = SF_TRON2AFlatPPORunnerCfg()
limx_sf_tron2a_stand_still_runner_cfg = SF_TRON2AStandStillPPORunnerCfg()
limx_sf_tron2a_flat_walk_runner_cfg = SF_TRON2AFlatWalkPPORunnerCfg()
limx_sf_tron2a_terrain_walk_runner_cfg = SF_TRON2ATerrainWalkPPORunnerCfg()

limx_wf_tron2a_blind_flat_runner_cfg = WF_TRON2AFlatPPORunnerCfg()


##
# Register Gym environments
##


######################################
# SF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# WF_TRON2A Blind Flat Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_blind_flat_runner_cfg,
    },
)


######################################
# SF_TRON2A Curriculum Environments
######################################


gym.register(
    id="Isaac-Limx-SF-TRON2A-StandStill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_StandStillEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_stand_still_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-StandStill-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_StandStillEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_stand_still_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-FlatWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_FlatWalkEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_flat_walk_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-FlatWalk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_FlatWalkEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_flat_walk_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-TerrainWalk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_TerrainWalkEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_terrain_walk_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-TerrainWalk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_TerrainWalkEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_terrain_walk_runner_cfg,
    },
)