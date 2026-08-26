"""Gap-only curriculum terrain for SF_TRON2A locomotion training.

Each track contains a flat approach followed by five full-width gaps separated
by landing platforms.  Difficulty controls gap width; the floor below the gaps
is deliberately deep so that the policy must place a foot on the far side
instead of stepping down into the gap.
"""

from __future__ import annotations

import numpy as np
import trimesh

import isaaclab.sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from terrain_base import BetterTerrainGenerator, BetterTerrainImporter
from mixed_terrain import _blend, _next_rng


def make_gap_track(difficulty: float, cfg: "GapTrackCfg"):
    """Build one gap track and return its meshes and robot origin."""
    rng = _next_rng(cfg.base_seed)
    width = cfg.width
    center_y = cfg.size[1] / 2.0
    meshes: list[trimesh.Trimesh] = []

    def _box(x_center: float, z_center: float, extent_x: float, extent_z: float):
        return trimesh.creation.box(
            (extent_x, width, extent_z),
            trimesh.transformations.translation_matrix((x_center, center_y, z_center)),
        )

    surface_thickness = cfg.surface_thickness
    x = 0.0

    # Stable approach so the gait can settle before the first gap.
    start_length = _blend(rng, *cfg.start_flat_len_range, difficulty, 0.0)
    meshes.append(_box(start_length / 2.0, -surface_thickness / 2.0, start_length, surface_thickness))
    x += start_length

    for gap_index in range(cfg.num_gaps):
        # Strong difficulty weighting makes the first rows genuinely easy and
        # the last rows approach the configured maximum width.
        gap_width = _blend(rng, *cfg.gap_width_range, difficulty, 0.9)
        x += gap_width

        landing_length = _blend(rng, *cfg.landing_len_range, difficulty, 0.0)
        if gap_index == cfg.num_gaps - 1:
            landing_length += _blend(rng, *cfg.finish_flat_len_range, difficulty, 0.0)
        meshes.append(
            _box(
                x + landing_length / 2.0,
                -surface_thickness / 2.0,
                landing_length,
                surface_thickness,
            )
        )
        x += landing_length

    total_length = x

    # A low catch floor gives the height scanner finite hits while keeping the
    # gap too deep to solve by simply stepping down.
    meshes.insert(
        0,
        _box(
            total_length / 2.0,
            -cfg.gap_depth - surface_thickness / 2.0,
            total_length,
            surface_thickness,
        ),
    )

    # The first gap begins roughly two metres in front of the reset origin.
    origin = np.array([1.0, center_y, 0.0])
    return meshes, origin


@configclass
class GapTrackCfg(SubTerrainBaseCfg):
    """Configuration for a repeated full-width gap track."""

    function = make_gap_track

    width: float = 8.0
    base_seed: int = 0
    num_gaps: int = 5
    gap_width_range: tuple[float, float] = (0.10, 0.38)
    start_flat_len_range: tuple[float, float] = (2.8, 3.2)
    landing_len_range: tuple[float, float] = (1.2, 1.6)
    finish_flat_len_range: tuple[float, float] = (2.5, 3.5)
    gap_depth: float = 1.0
    surface_thickness: float = 0.20


def make_gap_training_terrain_cfg(
    *,
    width: float = 8.0,
    cell_length: float = 25.0,
    num_rows: int = 10,
    num_cols: int = 10,
    base_seed: int = 0,
    difficulty_range: tuple[float, float] = (0.0, 1.0),
) -> TerrainImporterCfg:
    """Create a grid of gap-only tracks."""
    track = GapTrackCfg(width=width, base_seed=base_seed)
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
            difficulty_range=difficulty_range,
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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.55)),
        debug_vis=False,
    )


GAP_TRAINING_TERRAIN_CFG = make_gap_training_terrain_cfg()

# A single maximum-difficulty row for deterministic play/evaluation.
GAP_EVAL_TERRAIN_CFG = make_gap_training_terrain_cfg(
    num_rows=1,
    difficulty_range=(1.0, 1.0),
)
