from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel_custom(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """地形课程：根据机器人行走距离调整地形难度。

    与 IsaacLab 的 terrain_levels_vel 相同，但升级阈值从 size[0]/2 改为 size[0]/4，
    使机器人在 20 秒内以 ~1 m/s 行走（约 20m）即可升级到更难的地形行。

    仅适用于 terrain_type="generator" 的地形。
    """
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # 机器人从 spawn 点走出的距离
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # 走得够远 -> 升级到更难的地形（阈值 size[0]/8 = 10m）
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 8
    # 走得不够 -> 降级到更简单的地形
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.25
    move_down *= ~move_up

    # [DEBUG] 诊断课程不升级的问题
    if env.common_step_counter % 500 == 0 and len(env_ids) > 0:
        print(
            f"[DEBUG curriculum] step={env.common_step_counter}  "
            f"n_reset={len(env_ids)}  "
            f"dist_mean={distance.mean().item():.2f}  "
            f"dist_max={distance.max().item():.2f}  "
            f"move_up={move_up.sum().item()}  "
            f"move_down={move_down.sum().item()}  "
            f"terrain_levels_mean={terrain.terrain_levels.float().mean().item():.4f}  "
            f"terrain_origins_is_none={terrain.terrain_origins is None}  "
            f"curriculum_flag={terrain.cfg.terrain_generator.curriculum}  "
            f"size0={terrain.cfg.terrain_generator.size[0]}"
        )

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_stairs(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    move_up_distance: float = 7.5,
    move_down_distance: float = 2.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adjust stair difficulty from forward progress during the finished episode.

    Stair environments reset near their terrain origin. Advancing beyond
    ``move_up_distance`` therefore means the robot has entered and climbed a
    meaningful part of the staircase. Failing before ``move_down_distance``
    lowers the level, while intermediate progress preserves the current level.
    Only forward x progress is used, so lateral motion or retreat cannot promote
    an environment.
    """
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # CurriculumManager also runs during the initial reset, before any policy
    # episode has occurred. Preserve the configured initial level distribution.
    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())

    forward_progress = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = forward_progress > move_up_distance
    move_down = (forward_progress < move_down_distance) & ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_forward(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    move_up_distance: float,
    move_down_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Generic forward-progress curriculum shared by all teacher terrains."""
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    if move_down_distance >= move_up_distance:
        raise ValueError(
            f"move_down_distance ({move_down_distance}) must be below "
            f"move_up_distance ({move_up_distance})."
        )
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())

    progress = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = progress > move_up_distance
    move_down = (progress < move_down_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def terrain_levels_gaps(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    move_up_distance: float = 7.5,
    move_down_distance: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Adjust gap width from forward progress over a completed episode.

    The robot starts about two metres before the first gap.  Reaching
    ``move_up_distance`` means it crossed several gaps and promotes the
    environment.  Failing before ``move_down_distance`` means it did not
    reliably clear the first gap and lowers the level.
    """
    from isaaclab.assets import Articulation
    from isaaclab.terrains import TerrainImporter

    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    if env.common_step_counter == 0:
        return torch.mean(terrain.terrain_levels.float())

    forward_progress = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = forward_progress > move_up_distance
    move_down = (forward_progress < move_down_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)
