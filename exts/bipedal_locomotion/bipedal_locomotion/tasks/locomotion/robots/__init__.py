import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    SF_TRON2AFlatPPORunnerCfg, WF_TRON2AFlatPPORunnerCfg,
    SF_TRON2ACampPPORunnerCfg, WF_TRON2ACampPPORunnerCfg,
    SF_TRON2AStairsPPORunnerCfg, WF_TRON2AStairsPPORunnerCfg,
    SF_TRON2AGapPPORunnerCfg,
    SF_TRON2AContinuousTeacherPPORunnerCfg,
    SF_TRON2AStairsTeacherPPORunnerCfg,
    SF_TRON2AObstacleTeacherPPORunnerCfg,
    SF_TRON2AGapTeacherPPORunnerCfg,
    SF_TRON2AMultiTeacherRunnerCfg,
)

from . import (
    limx_solefoot_tron2a_env_cfg,
    limx_solefoot_tron2a_teacher_env_cfg,
    limx_wheelfoot_tron2a_env_cfg,
)

##
# Create PPO runners for RSL-RL
##

limx_sf_tron2a_blind_flat_runner_cfg = SF_TRON2AFlatPPORunnerCfg()
limx_wf_tron2a_blind_flat_runner_cfg = WF_TRON2AFlatPPORunnerCfg()

limx_sf_tron2a_camp_runner_cfg = SF_TRON2ACampPPORunnerCfg()
limx_wf_tron2a_camp_runner_cfg = WF_TRON2ACampPPORunnerCfg()

limx_sf_tron2a_stairs_runner_cfg = SF_TRON2AStairsPPORunnerCfg()
limx_wf_tron2a_stairs_runner_cfg = WF_TRON2AStairsPPORunnerCfg()

limx_sf_tron2a_gap_runner_cfg = SF_TRON2AGapPPORunnerCfg()

limx_sf_tron2a_teacher_continuous_runner_cfg = SF_TRON2AContinuousTeacherPPORunnerCfg()
limx_sf_tron2a_teacher_stairs_runner_cfg = SF_TRON2AStairsTeacherPPORunnerCfg()
limx_sf_tron2a_teacher_obstacle_runner_cfg = SF_TRON2AObstacleTeacherPPORunnerCfg()
limx_sf_tron2a_teacher_gap_runner_cfg = SF_TRON2AGapTeacherPPORunnerCfg()
limx_sf_tron2a_multi_teacher_runner_cfg = SF_TRON2AMultiTeacherRunnerCfg()


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
# SF_TRON2A Camp Terrain Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Camp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_CampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_camp_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Camp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_CampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_camp_runner_cfg,
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
# WF_TRON2A Camp Terrain Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Camp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_CampEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_camp_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Camp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_CampEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_camp_runner_cfg,
    },
)


######################################
# SF_TRON2A Stairs Terrain Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_StairsEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_StairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_stairs_runner_cfg,
    },
)


######################################
# SF_TRON2A Gap Terrain Environment
######################################
gym.register(
    id="Isaac-Limx-SF-TRON2A-Gap-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_GapEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_gap_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-TRON2A-Gap-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_env_cfg.SF_TRON2A_GapEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_gap_runner_cfg,
    },
)


######################################
# SF_TRON2A Unified Teacher Environments
######################################
_teacher_tasks = (
    (
        "Continuous",
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_ContinuousTeacherEnvCfg,
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_ContinuousTeacherEnvCfg_PLAY,
        limx_sf_tron2a_teacher_continuous_runner_cfg,
    ),
    (
        "Stairs",
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_StairsTeacherEnvCfg,
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_StairsTeacherEnvCfg_PLAY,
        limx_sf_tron2a_teacher_stairs_runner_cfg,
    ),
    (
        "Obstacle",
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_ObstacleTeacherEnvCfg,
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_ObstacleTeacherEnvCfg_PLAY,
        limx_sf_tron2a_teacher_obstacle_runner_cfg,
    ),
    (
        "Gap",
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_GapTeacherEnvCfg,
        limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_GapTeacherEnvCfg_PLAY,
        limx_sf_tron2a_teacher_gap_runner_cfg,
    ),
)

for _teacher_name, _env_cfg, _play_cfg, _runner_cfg in _teacher_tasks:
    gym.register(
        id=f"Isaac-Limx-SF-TRON2A-Teacher-{_teacher_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _env_cfg,
            "rsl_rl_cfg_entry_point": _runner_cfg,
        },
    )
    gym.register(
        id=f"Isaac-Limx-SF-TRON2A-Teacher-{_teacher_name}-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _play_cfg,
            "rsl_rl_cfg_entry_point": _runner_cfg,
        },
    )

gym.register(
    id="Isaac-Limx-SF-TRON2A-MultiTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_tron2a_teacher_env_cfg.SF_TRON2A_MultiTeacherEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_tron2a_multi_teacher_runner_cfg,
    },
)


######################################
# WF_TRON2A Stairs Terrain Environment
######################################
gym.register(
    id="Isaac-Limx-WF-TRON2A-Stairs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_StairsEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_stairs_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-TRON2A-Stairs-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_tron2a_env_cfg.WF_TRON2A_StairsEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_tron2a_stairs_runner_cfg,
    },
)
