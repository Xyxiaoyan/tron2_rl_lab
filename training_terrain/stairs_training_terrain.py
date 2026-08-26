# 楼梯专项训练地形 —— 只含楼梯，用于让机器人专注攻克上下楼梯。
#
# 结构：起始平地 → 上楼梯(N级) → 顶部平台 → 下楼梯(N级) → 结束平地
# 难度随行递增：步高 8~18cm，步数 4~12级，踏面 0.30~0.40m。

from __future__ import annotations

import math

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import FlatPatchSamplingCfg, SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from terrain_base import BetterTerrainGenerator, BetterTerrainImporter
from mixed_terrain import _next_rng, _blend


def make_stairs_track(difficulty: float, cfg: "StairsTrackCfg"):
    """构建一条纯楼梯赛道。返回 (mesh_list, origin)。"""
    rng = _next_rng(cfg.base_seed)
    W = cfg.width
    cy = cfg.size[1] / 2.0
    y0, y1 = cy - W / 2.0, cy + W / 2.0
    meshes: list[trimesh.Trimesh] = []

    def _box(x_center, z_center, ext_x, ext_z, ext_y=W, y_center=cy):
        return trimesh.creation.box(
            (ext_x, ext_y, ext_z),
            trimesh.transformations.translation_matrix((x_center, y_center, z_center)),
        )

    x = 0.0

    # --- 地面底板 (顶面在 z = 0)，待最终长度确定后再定尺寸 ---
    base_idx = len(meshes)
    meshes.append(None)  # 占位符

    # --- 1. 起始平地 ---
    x += _blend(rng, *cfg.start_flat_len_range, difficulty, 0.0)

    # --- 2. 上楼梯 ---
    # Strong difficulty weighting keeps the first curriculum rows genuinely
    # easy while still increasing smoothly across all ten rows.
    n_steps_up = int(round(_blend(rng, *cfg.n_stairs_range, difficulty, 0.8)))
    step_h = _blend(rng, *cfg.step_height_range, difficulty, 0.8)
    step_run = _blend(rng, *cfg.step_run_range, difficulty, 0.0)
    for i in range(n_steps_up):
        top_z = (i + 1) * step_h
        meshes.append(_box(x + (i + 0.5) * step_run, top_z / 2.0, step_run, top_z))
    x += n_steps_up * step_run
    top_z = n_steps_up * step_h

    # --- 3. 顶部平台 ---
    plat_len = _blend(rng, *cfg.platform_len_range, difficulty, 0.0)
    meshes.append(_box(x + plat_len / 2.0, top_z / 2.0, plat_len, top_z))
    x += plat_len

    # --- 4. 下楼梯 (镜像上楼梯，落回 z = 0) ---
    for i in range(n_steps_up):
        z = (n_steps_up - i) * step_h
        meshes.append(_box(x + (i + 0.5) * step_run, z / 2.0, step_run, z))
    x += n_steps_up * step_run

    # --- 5. 结束平地 ---
    x += _blend(rng, *cfg.finish_flat_len_range, difficulty, 0.0)
    total_len = x

    # 填充底板 (顶面在 z = 0)。
    meshes[base_idx] = _box(total_len / 2.0, -0.1, total_len, 0.2)

    origin = np.array([1.0, cy, 0.0])
    return meshes, origin


@configclass
class StairsTrackCfg(SubTerrainBaseCfg):
    """纯楼梯赛道的子地形配置。"""

    function = make_stairs_track

    width: float = 8.0
    base_seed: int = 0

    start_flat_len_range: tuple = (3.0, 6.0)
    finish_flat_len_range: tuple = (3.0, 6.0)
    platform_len_range: tuple = (1.0, 2.5)

    n_stairs_range: tuple = (4, 12)
    step_height_range: tuple = (0.08, 0.18)
    step_run_range: tuple = (0.30, 0.40)

    # 采样平坦位置用于机器人 spawn
    flat_patch_sampling = {
        "init_pos": FlatPatchSamplingCfg(
            num_patches=1000,
            patch_radius=0.5,
            max_height_diff=0.1,
        )
    }


def make_stairs_training_terrain_cfg(
    *,
    width: float = 8.0,
    cell_length: float = 40.0,
    num_rows: int = 10,
    num_cols: int = 10,
    base_seed: int = 0,
    track_cfg: "StairsTrackCfg | None" = None,
) -> TerrainImporterCfg:
    """构建一个由纯楼梯地形格组成的 TerrainImporterCfg。"""
    track = track_cfg or StairsTrackCfg()
    track.width = width
    track.base_seed = base_seed
    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            class_type=BetterTerrainGenerator,
            seed=base_seed,
            size=(cell_length, max(width, 8.0)),
            border_width=0.0,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={"track": track},
        ),
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.6)),
        debug_vis=False,
    )


# 预置配置：10x10 楼梯地形，难度随行递增
STAIRS_TRAINING_TERRAIN_CFG = make_stairs_training_terrain_cfg(
    width=8.0,
    cell_length=40.0,
    num_rows=10,
    num_cols=10,
    base_seed=0,
)
