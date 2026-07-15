from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, Camera, TiledCamera, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_joint_torque(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_mass.to(device)


def robot_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia.to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase follows ``gait_command.gait_indices`` (same integrator as GaitReward).

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Use the same integrated phase as GaitReward / gait_command (not episode_length_buf).
    command_term = env.command_manager.get_term("gait_command")
    gait_indices = command_term.gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def feet_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


def joint_pos_rel_exclude_wheel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                wheel_joints_name: list[str] = ["wheel_[RL]_Joint"] 
                                ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return asset.data.joint_pos[:, pos_idx_exclude_wheel] - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]


# ---- 评测传感器观测函数 ----

def sensor_image_bundle(
    env: ManagerBasedEnv,
    head_sensor_cfg: SceneEntityCfg = SceneEntityCfg("head_camera"),
    down_sensor_cfg: SceneEntityCfg = SceneEntityCfg("down_camera"),
) -> torch.Tensor:
    """Bundled downsampled images for sensor encoder input.

    Returns a flat tensor combining head-RGB, head-depth, down-RGB, down-depth,
    each downsampled to a fixed resolution. The SensorEncoder will reshape internally.

    Returns:
        (N, D) flat tensor where D = 2 * (3+1) * H * W  (2 cameras × 4 ch × 24 × 32 = 6144)
    """
    import torch.nn.functional as F

    RESIZE_H, RESIZE_W = 24, 32  # small spatial for manageability
    imgs = []
    for sensor_cfg in (head_sensor_cfg, down_sensor_cfg):
        camera: TiledCamera = env.scene.sensors[sensor_cfg.name]
        rgb = camera.data.output["rgb"].clone()  # (N, H, W, 3)
        depth = camera.data.output["distance_to_image_plane"].clone().unsqueeze(-1)  # (N, H, W, 1)
        depth[torch.isinf(depth)] = 0.0

        # downsample: (N, H, W, C) → (N, C, H, W) → interpolate → (N, C, h, w) → flatten
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
        depth = depth.permute(0, 3, 1, 2).float() / 10.0  # normalize depth to ~[0,1]

        rgb_small = F.interpolate(rgb, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False)
        depth_small = F.interpolate(depth, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False)

        imgs.append(rgb_small.flatten(start_dim=1))   # (N, 3 * 24 * 32)
        imgs.append(depth_small.flatten(start_dim=1))  # (N, 1 * 24 * 32)

    return torch.cat(imgs, dim=-1)  # (N, 2 * 4 * 24 * 32) = (N, 6144)


def sensor_lidar_bundle(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    target_rows: int = 24,
    target_cols: int = 72,
) -> torch.Tensor:
    """Downsampled LiDAR range data for sensor encoder input.

    Uses average-pooling over the (channels × rays) grid to reduce dimensionality.

    Args:
        target_rows: Downsampled vertical channels.
        target_cols: Downsampled horizontal resolution.

    Returns:
        (N, target_rows * target_cols) flat tensor.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    n_envs = sensor.data.ray_hits_w.shape[0]

    # ray_hits_w: (N, num_rays, 3) = (N, 96*360, 3) → take z (height)
    z_hits = sensor.data.ray_hits_w[..., 2].clone()  # (N, 96*360)

    # Reshape to (N, 96, 360) and downsample
    z_hits = z_hits.view(n_envs, 96, 360)
    z_hits = z_hits.view(n_envs, target_rows, 96 // target_rows, target_cols, 360 // target_cols)
    z_hits = z_hits.mean(dim=(2, 4))  # (N, target_rows, target_cols)
    # Normalize by max_range
    z_hits = z_hits / 30.0  # max_distance
    z_hits = z_hits.clamp(-1.0, 1.0)
    return z_hits.flatten(start_dim=1)  # (N, target_rows * target_cols)


# ---- 原有评测传感器观测函数（保留用于调试） ----
