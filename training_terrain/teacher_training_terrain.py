"""Unified terrain families for SF_TRON2A teacher training and distillation."""

from __future__ import annotations

import math

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from gap_training_terrain import GapTrackCfg
from mixed_terrain import _blend, _next_rng
from stairs_training_terrain import StairsTrackCfg
from terrain_base import BetterTerrainGenerator, BetterTerrainGeneratorCfg, BetterTerrainImporter


SKILL_CONTINUOUS = 0
SKILL_STAIRS = 1
SKILL_OBSTACLE = 2
SKILL_GAP = 3
SKILL_NAMES = ("continuous", "stairs", "obstacle", "gap")


def _box(x: float, y: float, z: float, sx: float, sy: float, sz: float) -> trimesh.Trimesh:
    return trimesh.creation.box(
        (sx, sy, sz),
        trimesh.transformations.translation_matrix((x, y, z)),
    )


def _ramp_prism(x_lo: float, x_hi: float, z_lo: float, z_hi: float, y0: float, y1: float):
    profile = (
        [(x_lo, z_lo), (x_hi, z_lo), (x_hi, z_hi)]
        if z_hi >= z_lo
        else [(x_lo, z_lo), (x_hi, z_hi), (x_lo, z_hi)]
    )
    vertices = np.asarray(
        [[x, y0, z] for x, z in profile] + [[x, y1, z] for x, z in profile], dtype=float
    )
    n = len(profile)
    faces = []
    for index in range(n):
        next_index = (index + 1) % n
        faces.extend(
            ([index, next_index, next_index + n], [index, next_index + n, index + n])
        )
    faces.extend(([0, 1, 2], [n, n + 2, n + 1]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)
    mesh.fix_normals()
    return mesh


def make_continuous_track(difficulty: float, cfg: "ContinuousTrackCfg"):
    """Flat lead-in followed by an up-ramp, plateau, down-ramp and flat exit."""
    rng = _next_rng(cfg.base_seed)
    width = cfg.width
    center_y = cfg.size[1] / 2.0
    y0, y1 = center_y - width / 2.0, center_y + width / 2.0
    x = 0.0
    meshes: list[trimesh.Trimesh] = []

    start = _blend(rng, *cfg.start_flat_len_range, difficulty, 0.0)
    x += start
    angle = math.radians(_blend(rng, *cfg.ramp_angle_deg_range, difficulty, 0.85))
    run_up = _blend(rng, *cfg.ramp_run_range, difficulty, 0.0)
    rise = run_up * math.tan(angle)
    meshes.append(_ramp_prism(x, x + run_up, 0.0, rise, y0, y1))
    x += run_up

    plateau = _blend(rng, *cfg.plateau_len_range, difficulty, 0.0)
    meshes.append(_box(x + plateau / 2.0, center_y, rise / 2.0, plateau, width, rise))
    x += plateau

    run_down = _blend(rng, *cfg.ramp_run_range, difficulty, 0.0)
    meshes.append(_ramp_prism(x, x + run_down, rise, 0.0, y0, y1))
    x += run_down
    x += _blend(rng, *cfg.finish_flat_len_range, difficulty, 0.0)

    meshes.insert(0, _box(x / 2.0, center_y, -0.1, x, width, 0.2))
    return meshes, np.asarray([1.0, center_y, 0.0])


@configclass
class ContinuousTrackCfg(SubTerrainBaseCfg):
    function = make_continuous_track
    width: float = 8.0
    base_seed: int = 1100
    start_flat_len_range: tuple[float, float] = (3.0, 5.0)
    finish_flat_len_range: tuple[float, float] = (3.0, 5.0)
    ramp_angle_deg_range: tuple[float, float] = (3.0, 18.0)
    ramp_run_range: tuple[float, float] = (2.5, 4.5)
    plateau_len_range: tuple[float, float] = (1.5, 3.5)


def make_obstacle_track(difficulty: float, cfg: "ObstacleTrackCfg"):
    """Full-width bumps and isolated raised platforms that cannot be bypassed."""
    rng = _next_rng(cfg.base_seed)
    width = cfg.width
    center_y = cfg.size[1] / 2.0
    x = _blend(rng, *cfg.start_flat_len_range, difficulty, 0.0)
    meshes: list[trimesh.Trimesh | None] = [None]

    num_bumps = int(round(_blend(rng, *cfg.num_bumps_range, difficulty, 0.6)))
    for _ in range(num_bumps):
        height = _blend(rng, *cfg.bump_height_range, difficulty, 0.85)
        thickness = _blend(rng, *cfg.bump_thickness_range, difficulty, 0.0)
        meshes.append(_box(x + thickness / 2.0, center_y, height / 2.0, thickness, width, height))
        x += thickness + _blend(rng, *cfg.flat_spacing_range, difficulty, 0.0)

    num_platforms = int(round(_blend(rng, *cfg.num_platforms_range, difficulty, 0.5)))
    for _ in range(num_platforms):
        height = _blend(rng, *cfg.platform_height_range, difficulty, 0.9)
        length = _blend(rng, *cfg.platform_len_range, difficulty, 0.0)
        meshes.append(_box(x + length / 2.0, center_y, height / 2.0, length, width, height))
        x += length + _blend(rng, *cfg.flat_spacing_range, difficulty, 0.0)

    x += _blend(rng, *cfg.finish_flat_len_range, difficulty, 0.0)
    meshes[0] = _box(x / 2.0, center_y, -0.1, x, width, 0.2)
    return meshes, np.asarray([1.0, center_y, 0.0])


@configclass
class ObstacleTrackCfg(SubTerrainBaseCfg):
    function = make_obstacle_track
    width: float = 8.0
    base_seed: int = 2200
    start_flat_len_range: tuple[float, float] = (3.0, 4.5)
    finish_flat_len_range: tuple[float, float] = (3.0, 4.5)
    flat_spacing_range: tuple[float, float] = (1.0, 2.0)
    num_bumps_range: tuple[int, int] = (1, 3)
    bump_height_range: tuple[float, float] = (0.03, 0.20)
    bump_thickness_range: tuple[float, float] = (0.10, 0.30)
    num_platforms_range: tuple[int, int] = (1, 3)
    platform_height_range: tuple[float, float] = (0.08, 0.42)
    platform_len_range: tuple[float, float] = (0.60, 1.10)


def _terrain_importer(generator_cfg: BetterTerrainGeneratorCfg, max_init_level: int) -> TerrainImporterCfg:
    return TerrainImporterCfg(
        class_type=BetterTerrainImporter,
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator_cfg,
        max_init_terrain_level=max_init_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.58, 0.58)),
        debug_vis=False,
    )


def make_single_teacher_terrain_cfg(name: str, num_rows: int = 10, num_cols: int = 10):
    configs = {
        "continuous": ContinuousTrackCfg(),
        "stairs": StairsTrackCfg(base_seed=3300),
        "obstacle": ObstacleTrackCfg(),
        "gap": GapTrackCfg(base_seed=4400),
    }
    if name not in configs:
        raise ValueError(f"Unknown teacher terrain '{name}'. Expected one of {tuple(configs)}.")
    generator_cfg = BetterTerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(45.0, 8.0),
        border_width=0.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={name: configs[name]},
        skill_id_by_terrain={name: SKILL_NAMES.index(name)},
    )
    return _terrain_importer(generator_cfg, max_init_level=1)


def make_multi_teacher_terrain_cfg(num_rows: int = 10, num_cols: int = 8):
    if num_cols % len(SKILL_NAMES) != 0:
        raise ValueError(f"num_cols must be divisible by {len(SKILL_NAMES)}, got {num_cols}.")
    sub_terrains = {
        "continuous": ContinuousTrackCfg(proportion=1.0),
        "stairs": StairsTrackCfg(proportion=1.0, base_seed=3300),
        "obstacle": ObstacleTrackCfg(proportion=1.0),
        "gap": GapTrackCfg(proportion=1.0, base_seed=4400),
    }
    generator_cfg = BetterTerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(45.0, 8.0),
        border_width=0.0,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains=sub_terrains,
        skill_id_by_terrain={name: index for index, name in enumerate(SKILL_NAMES)},
    )
    return _terrain_importer(generator_cfg, max_init_level=num_rows - 1)


CONTINUOUS_TEACHER_TERRAIN_CFG = make_single_teacher_terrain_cfg("continuous")
STAIRS_TEACHER_TERRAIN_CFG = make_single_teacher_terrain_cfg("stairs")
OBSTACLE_TEACHER_TERRAIN_CFG = make_single_teacher_terrain_cfg("obstacle")
GAP_TEACHER_TERRAIN_CFG = make_single_teacher_terrain_cfg("gap")
MULTI_TEACHER_TERRAIN_CFG = make_multi_teacher_terrain_cfg()
