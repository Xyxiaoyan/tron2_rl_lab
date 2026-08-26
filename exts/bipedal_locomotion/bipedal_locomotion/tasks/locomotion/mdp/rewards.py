"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
import isaaclab.utils.math as math_utils
from bipedal_locomotion.utils.math import CubicSpline


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def forward_progress(
    env: ManagerBasedRLEnv,
    heading_target: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励机器人沿赛道方向（世界 x 轴）的位移增量。

    每步计算当前 x 位置与上一步 x 位置之差（= 位移增量）。
    这等价于速度 × dt，但直接基于位置变化，无法通过原地踏步欺骗。
    正增量（前进）给奖励，零/负增量（原地/后退）给惩罚。如果指定 ``heading_target``，
    正向奖励会乘以机身朝向与目标朝向的一致性，防止侧着身体刷前进奖励。

    自动检测 reset（位置突变 > 1m）并重置缓存，避免跨 episode 的错误增量。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos_x = asset.data.root_pos_w[:, 0]

    # 首次调用：初始化缓存，返回 0
    if not hasattr(env, "_prev_pos_x"):
        env._prev_pos_x = pos_x.clone()
        return torch.zeros(env.num_envs, device=env.device)

    # 检测 reset：位置突变 > 1m 说明机器人被传送到新起点
    pos_jump = torch.abs(pos_x - env._prev_pos_x) > 1.0
    # 对 reset 的环境，重置缓存为当前位置（本帧增量=0）
    env._prev_pos_x = torch.where(pos_jump, pos_x, env._prev_pos_x)

    # 位移增量 = 当前 - 上一步
    delta_x = pos_x - env._prev_pos_x
    env._prev_pos_x = pos_x.clone()

    positive_progress = torch.clamp(delta_x * 50.0, max=1.5)
    if heading_target is not None:
        # cos(yaw error) 在正对赛道时为 1，侧向或背对赛道时为 0。
        _, _, base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
        yaw_error = normalize_angle(base_yaw - heading_target)
        heading_scale = torch.clamp(torch.cos(yaw_error), min=0.0)
        positive_progress = positive_progress * heading_scale

    # 正增量给奖励，原地/后退继续给完整惩罚。
    reward = torch.where(
        delta_x > 0.0,
        positive_progress,                        # 前进：放大增量（dt=0.02，×50 ≈ 速度）
        torch.clamp(delta_x * 150.0, min=-0.5),   # 原地/后退：放大惩罚（比之前更狠）
    )
    return reward


def heading_alignment_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_heading: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励机身 yaw 朝向跟踪固定的世界坐标系朝向。

    用于楼梯等行进方向固定的地形；与朝向加权的前进奖励配合，
    避免策略以侧身姿态沿赛道移动。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    _, _, base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_error = normalize_angle(base_yaw - target_heading)
    return torch.exp(-torch.square(yaw_error) / std**2)


def corridor_penalty(
    env: ManagerBasedRLEnv,
    corridor_half_width: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机器人偏离赛道中心过远。

    在走廊宽度内无惩罚，超出后线性惩罚。用于约束机器人在赛道内行走，
    为较窄的评测赛道做准备。

    Args:
        corridor_half_width: 走廊半宽（从赛道中心到边缘的距离），超出此距离开始惩罚。
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    y_deviation = torch.abs(asset.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1])
    return torch.clamp(y_deviation - corridor_half_width, min=0.0)


def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_x_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the x-axis linear velocity command in the gravity-aligned (yaw) robot frame using an exponential kernel.

    Split-out of :func:`track_lin_vel_xy_yaw_frame_exp` so that the longitudinal (x) and
    lateral (y) tracking rewards can carry independent weights / kernel widths.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 0] - vel_yaw[:, 0]
    )
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_y_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of the y-axis linear velocity command in the gravity-aligned (yaw) robot frame using an exponential kernel.

    Split-out of :func:`track_lin_vel_xy_yaw_frame_exp` so the lateral channel can be
    tuned (weight / std) independently from the longitudinal channel — useful when
    pure-lateral motion is under-trained or when symmetry constraints make the
    network insensitive to vy commands.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 1] - vel_yaw[:, 1]
    )
    return torch.exp(-lin_vel_error / std**2)


def joint_powers_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def joint_deviation_from_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints deviating from their default (initial) positions using L2 kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_error), dim=1)


def leg_symmetry(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward regulate abad joint position."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)


def same_feet_x_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward regulate abad joint position."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    return feet_x_distance


def contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    asset: Articulation = env.scene[asset_cfg.name]
    robot_links_mass = asset.root_physx_view.get_masses()
    robot_mass = torch.sum(robot_links_mass, dim=-1, keepdim=True)
    robot_mass = robot_mass.to(env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - robot_mass * 9.8
    return torch.sum(violation.clip(min=0.0), dim=1)


def distance_aligned(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_dist: float,
    max_dist: float,
    desired_dist: float,
    std: float,
    command_name: str = "base_velocity",
    vy_max: float = 0.8,
    decay_power: float = 1.0,
) -> torch.Tensor:
    """Penalize feet distance from the desired distance."""
    asset: RigidObject = env.scene[asset_cfg.name]

    left_idx = asset_cfg.body_ids[0]
    right_idx = asset_cfg.body_ids[1]
    base_quat = asset.data.root_quat_w
    heading_aligned = math_utils.yaw_quat(base_quat)

    left_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, left_idx])
    right_pos = math_utils.quat_apply_inverse(heading_aligned, asset.data.body_pos_w[:, right_idx])

    distance_y = torch.abs(left_pos[:, 1] - right_pos[:, 1])
    d_min = torch.where(distance_y < min_dist, min_dist - distance_y, torch.tensor(0.0, device=distance_y.device))
    d_max = torch.where(distance_y > max_dist, distance_y - max_dist, torch.tensor(0.0, device=distance_y.device))

    reward_1 = torch.exp(-(d_min + d_max) / std**2)
    reward_2 = torch.exp(-torch.square((distance_y - desired_dist) / std**2))

    vy = env.command_manager.get_command(command_name)[:, 1]
    x = torch.clamp(torch.abs(vy) / vy_max, 0.0, 1.0)
    vy_weight = (1.0 - x) ** decay_power

    return (reward_1 + vy_weight * reward_2) / 2


def terrain_adaptive_height(
    env: ManagerBasedRLEnv,
    std: float,
    stand_height: float = 0.60,
    lookahead: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """地形自适应高度奖励。

    根据机器人前方地形高度动态调整目标 base 高度：
    - 前方是台阶/上坡（地形升高）→ 目标高度升高，奖励机器人爬上去
    - 前方是下坡/下台阶（地形下降）→ 目标高度降低，奖励机器人降下来
    - 前方平坦 → 目标高度 = stand_height（与固定高度奖励一致）

    Args:
        stand_height: 平地上的目标 base 高度（相对脚下地面）。
        lookahead: 前方 lookahead 米内的最高点作为目标地形高度。
    """
    from isaaclab.sensors import RayCaster

    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    # 射线命中点（世界坐标）和传感器位置
    hits_w = sensor.data.ray_hits_w  # (num_envs, num_rays, 3)
    sensor_pos = sensor.data.pos_w   # (num_envs, 3)

    # 转到机器人局部坐标（只看 yaw），区分前方/后方
    base_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)  # (num_envs, 4)
    hits_rel = hits_w - sensor_pos.unsqueeze(1)  # 相对传感器
    hits_local = math_utils.quat_apply_inverse(
        base_yaw_quat.unsqueeze(1).expand(-1, hits_rel.shape[1], -1).reshape(-1, 4),
        hits_rel.reshape(-1, 3),
    ).reshape(hits_rel.shape)

    # 筛选前方点（local x > lookahead），取这些点的世界 z
    is_ahead = hits_local[..., 0] > lookahead
    hits_z = hits_w[..., 2]
    # 前方最高点：用 where 而非 -1e6 填充，避免无前方点时产生极端值
    # 无前方点时退回当前地面（delta=0）
    ground_z = torch.median(hits_z, dim=1)[0]  # (num_envs,) 当前脚下地面
    max_ahead_z = torch.where(
        is_ahead.any(dim=1),
        torch.where(is_ahead, hits_z, torch.full_like(hits_z, -1e4)).max(dim=1)[0],
        ground_z,  # 无前方点时退回地面
    )

    # 前方地形相对当前地面的高度差，clamp 到合理范围避免 NaN
    terrain_delta = torch.clamp(max_ahead_z - ground_z, min=-1.0, max=1.0)

    # 目标 base 高度 = 当前地面高度 + stand_height + 软化的前方高度差
    soft_delta = terrain_delta * torch.sigmoid(torch.abs(terrain_delta) * 5.0)
    target_height = ground_z + stand_height + soft_delta

    # 指数奖励：base 接近目标高度
    base_z = asset.data.root_pos_w[:, 2]
    height_error = torch.square(base_z - target_height)
    reward = torch.exp(-height_error / std**2)
    # 防止 NaN 传播（传感器未初始化时返回 0）
    return torch.nan_to_num(reward, nan=0.0)


def support_foot_adaptive_height(
    env: ManagerBasedRLEnv,
    std: float,
    stand_height: float = 0.60,
    foot_radius: float = 0.074,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
) -> torch.Tensor:
    """Reward base height relative to the feet that currently support the robot.

    A forward terrain scan must not directly raise the target base height: doing
    so lets the policy collect reward by straightening its legs in front of a
    step. Instead, the target rises only after a foot actually contacts a higher
    support surface. During double support the two support heights are averaged;
    when neither foot is in contact, the lower foot provides a stable fallback.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_z = asset.data.body_pos_w[:, foot_cfg.body_ids, 2]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_contact_force = torch.norm(contact_forces, dim=-1).max(dim=1)[0]
    contacts = max_contact_force > contact_threshold

    # The tracked ankle body origin is at the center of the rounded foot, so
    # subtract the radius to estimate the supporting terrain surface.
    feet_ground_z = feet_z - foot_radius
    contact_weights = contacts.to(feet_ground_z.dtype)
    num_contacts = contact_weights.sum(dim=1)
    support_ground_z = torch.sum(feet_ground_z * contact_weights, dim=1) / num_contacts.clamp(min=1.0)

    # In flight there is no measured support surface. Referencing the lower foot
    # keeps the reward about leg extension rather than an unseen forward step.
    fallback_ground_z = torch.min(feet_ground_z, dim=1)[0]
    support_ground_z = torch.where(num_contacts > 0.0, support_ground_z, fallback_ground_z)

    target_height = support_ground_z + stand_height
    base_z = asset.data.root_pos_w[:, 2]
    height_error = torch.square(base_z - target_height)
    reward = torch.exp(-height_error / std**2)
    return torch.nan_to_num(reward, nan=0.0)


def highest_support_foot_height(
    env: ManagerBasedRLEnv,
    std: float,
    stand_height: float = 0.60,
    foot_radius: float = 0.074,
    contact_threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="ankle_pitch_.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="ankle_pitch_.*"),
) -> torch.Tensor:
    """Reward base height above the highest supporting foot.

    This gap-specific variant prevents a foot hanging into a gap from lowering
    the base-height target.  If neither foot is in contact, the higher foot is
    used as a conservative fallback instead of legitimizing a crouch around the
    lower foot.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_ground_z = asset.data.body_pos_w[:, foot_cfg.body_ids, 2] - foot_radius
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    max_contact_force = torch.norm(contact_forces, dim=-1).max(dim=1)[0]
    contacts = max_contact_force > contact_threshold

    negative_inf = torch.full_like(feet_ground_z, -torch.inf)
    contacted_heights = torch.where(contacts, feet_ground_z, negative_inf)
    highest_contact = torch.max(contacted_heights, dim=1)[0]
    highest_foot = torch.max(feet_ground_z, dim=1)[0]
    has_contact = torch.any(contacts, dim=1)
    support_ground_z = torch.where(has_contact, highest_contact, highest_foot)

    target_height = support_ground_z + stand_height
    height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    reward = torch.exp(-height_error / std**2)
    return torch.nan_to_num(reward, nan=0.0)


def forward_stagnation(
    env: ManagerBasedRLEnv,
    min_forward_speed: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a penalty magnitude when commanded forward motion stalls.

    The term is zero once world-x speed reaches ``min_forward_speed`` and rises
    linearly to one at zero or backward speed.  It is intended for fixed +x
    tracks and should be configured with a negative reward weight.
    """
    if min_forward_speed <= 0.0:
        raise ValueError(f"Expected min_forward_speed > 0, got {min_forward_speed}.")
    asset: RigidObject = env.scene[asset_cfg.name]
    forward_speed = asset.data.root_lin_vel_w[:, 0]
    penalty = torch.clamp((min_forward_speed - forward_speed) / min_forward_speed, min=0.0, max=1.0)
    moving_command = env.command_manager.get_command(command_name)[:, 0] > 0.1
    return penalty * moving_command.to(penalty.dtype)


def stand_still(
    env,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize linear and angular motion when command velocities are near zero."""
    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


def base_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.72,
) -> torch.Tensor:
    """Penalize base height from the target height using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height_error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-base_height_error / std**2)


def base_projection_at_feet_midpoint(
    env: ManagerBasedRLEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="wheel_.*")
) -> torch.Tensor:
    """Reward base projection at the feet midpoint."""
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, feet_cfg.body_ids, :2]
    midpoint_xy = torch.mean(feet_pos_w, dim=1)
    base_xy = asset.data.root_pos_w[:, :2]
    error_sq = torch.sum(torch.square(base_xy - midpoint_xy), dim=1)
    return torch.exp(-error_sq / std**2)


class FeetSlidePenaltyWrapper:
    """A wrapper class for calculating feet slide penalty."""
    def __init__(self):
        self.count = 0
        self.friction = None
        self.__name__ = "feet_slide_penalty"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize feet slide."""

        asset: RigidObject | Articulation = env.scene[asset_cfg.name]

        if self.count <= 1:
            self.friction = asset.root_physx_view.get_material_properties()[..., 0].to(device=asset.device)
            self.num_shapes_per_body = []
            for link_path in asset.root_physx_view.link_paths[0]:
                link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)

            # sample material properties from the given ranges
            body_count = 0
            self.body_ids = []
            for body_ids, valid in enumerate(self.num_shapes_per_body):
                if valid:
                    if isinstance(asset_cfg.body_ids, slice):
                        asset_cfg.body_ids = list(range(len(asset_cfg.body_ids)))
                    if body_ids in asset_cfg.body_ids:
                        self.body_ids.append(body_count)
                    body_count += 1

        self.count += 1

        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
        )
        asset = env.scene[asset_cfg.name]
        feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
        reward = torch.sum(torch.square(feet_vel.norm(dim=-1)) * contacts * self.friction[:, self.body_ids], dim=1)

        return reward


feet_slide_penalty = FeetSlidePenaltyWrapper()


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def stand_still_reg(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exclude_joints_name: list[str] = [r"J\d"],
) -> torch.Tensor:
    """Regulate the stand still"""
    asset: Articulation = env.scene[asset_cfg.name]

    exclude_joints_idx = asset.find_joints(exclude_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    vel_idx_exclude_arm = [i for i in all_joints_idx if i not in exclude_joints_idx]

    joint_vel = asset.data.joint_vel[:, vel_idx_exclude_arm]

    # stand still env , set to zero
    is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore

    not_standing_env_ids = (~is_standing_env).nonzero(as_tuple=False).flatten()

    reward = torch.sum(torch.abs(joint_vel), dim=1)

    reward[not_standing_env_ids] = 0.0

    return reward


class TorquesSmoothnessPenaltyWrapper:
    """A wrapper class for calculating torques smoothness penalty."""

    def __init__(self):
        self.prev_torques = None
        self.__name__ = "torques_smoothness_penalty"

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalize large instantaneous changes in the torques output"""
        asset: Articulation = env.scene[asset_cfg.name]
        torques = asset.data.applied_torque[:, asset_cfg.joint_ids].clone()
        if self.prev_torques is None:
            self.prev_torques = torques
            return torch.zeros(torques.shape[0], device=torques.device)
        reward = torch.sum(torch.square(torques - self.prev_torques), dim=1)
        self.prev_torques = torques
        return reward


torques_smoothness_penalty = TorquesSmoothnessPenaltyWrapper()


def body_orientation_yaw_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_yaw: float = 0.0,
) -> torch.Tensor:
    """Penalize body orientation yaw error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    num_body = len(asset_cfg.body_ids)
    base_quat = asset.data.root_quat_w
    inverse_base_quat = math_utils.quat_inv(base_quat).unsqueeze(1).expand(-1, num_body, -1)
    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    body_quat_b = math_utils.quat_mul(inverse_base_quat, body_quat_w).flatten(0, 1)
    r, p, y = math_utils.euler_xyz_from_quat(body_quat_b.squeeze(1))

    yaw_error = normalize_angle(y - target_yaw)

    ruler_angle = torch.stack([normalize_angle(r), normalize_angle(p), yaw_error], dim=-1).reshape(-1, num_body, 3)
    quat_mismatch = torch.exp(-torch.abs(ruler_angle[:, :, 2]) * 10)
    return torch.mean(quat_mismatch, dim=1)


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    # stand still env , set to zero
    is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore
    no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()
    reward[no_gait_env_ids] = 0.0
    return reward


def touchdown_forward_step_length(
    env: ManagerBasedRLEnv,
    target_length: float,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward a forward separation between the landing foot and the support foot.

    The reward is evaluated only on the first frame of a foot contact.  A landing
    ``target_length`` metres ahead of the other foot receives one; shorter steps
    receive a proportional reward and backward landings receive a penalty.  Using
    the robot yaw frame makes the term independent of its world position.
    """
    if target_length <= 0.0:
        raise ValueError(f"Expected target_length > 0, got {target_length}.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    base_pos_w = asset.data.root_pos_w[:, :3].unsqueeze(1)
    base_yaw = math_utils.yaw_quat(asset.data.root_quat_w)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_yaw.unsqueeze(1).expand(-1, feet_pos_w.shape[1], -1).reshape(-1, 4),
        (feet_pos_w - base_pos_w).reshape(-1, 3),
    ).reshape(feet_pos_w.shape)

    # This term is defined for a biped: the opposite foot is the current support
    # reference when a new foot touches down.
    if feet_pos_b.shape[1] != 2:
        raise ValueError(
            f"touchdown_forward_step_length expects exactly two feet, got {feet_pos_b.shape[1]}."
        )
    feet_x = feet_pos_b[:, :, 0]
    forward_separation = feet_x - torch.flip(feet_x, dims=[1])
    step_score = torch.clamp(forward_separation / target_length, min=-1.0, max=1.0)
    reward = torch.sum(first_contact.to(step_score.dtype) * step_score, dim=1)

    # Do not encourage stepping when the commanded longitudinal velocity is zero.
    moving_forward = env.command_manager.get_command(command_name)[:, 0] > 0.1
    return reward * moving_forward.to(reward.dtype)


def swing_foot_forward_tracking(
    env: ManagerBasedRLEnv,
    target_length: float,
    std: float,
    command_name: str,
    gait_command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track a front-to-back foot-separation trajectory throughout swing.

    At lift-off the swing foot should be behind the support foot; during swing
    its desired separation moves continuously forward and reaches
    ``target_length`` before touchdown.  Unlike a touchdown-only term, this
    provides dense guidance even before the policy has ever cleared a gap.
    """
    if target_length <= 0.0 or std <= 0.0:
        raise ValueError(f"Expected target_length and std > 0, got {target_length=} and {std=}.")

    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if feet_pos_w.shape[1] != 2:
        raise ValueError(f"swing_foot_forward_tracking expects exactly two feet, got {feet_pos_w.shape[1]}.")

    base_pos_w = asset.data.root_pos_w[:, :3].unsqueeze(1)
    base_yaw = math_utils.yaw_quat(asset.data.root_quat_w)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_yaw.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 4),
        (feet_pos_w - base_pos_w).reshape(-1, 3),
    ).reshape(feet_pos_w.shape)
    feet_x = feet_pos_b[:, :, 0]
    actual_separation = feet_x - torch.flip(feet_x, dims=[1])

    gait_command = env.command_manager.get_command(gait_command_name)
    gait_phase = env.command_manager.get_term(gait_command_name).gait_indices
    phase_offset = gait_command[:, 1]
    contact_duration = gait_command[:, 2].unsqueeze(1).expand(-1, 2)
    foot_phase = torch.remainder(
        torch.stack((gait_phase, gait_phase + phase_offset), dim=1),
        1.0,
    )
    swing_mask = foot_phase >= contact_duration
    swing_progress = torch.clamp(
        (foot_phase - contact_duration) / (1.0 - contact_duration),
        min=0.0,
        max=1.0,
    )
    desired_separation = target_length * (2.0 * swing_progress - 1.0)
    tracking_score = torch.exp(-torch.square(actual_separation - desired_separation) / std**2)
    swing_count = torch.sum(swing_mask.to(tracking_score.dtype), dim=1).clamp(min=1.0)
    reward = torch.sum(tracking_score * swing_mask.to(tracking_score.dtype), dim=1) / swing_count

    moving_forward = env.command_manager.get_command(command_name)[:, 0] > 0.1
    return reward * moving_forward.to(reward.dtype)


def joint_orientation_l1_symmetric(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(joint_error), dim=-1)
    return reward


def joint_orientation_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    knee_joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(knee_joint_error) * (knee_joint_error > 0), dim=-1)
    return reward


def weighted_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    deviation_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    weighted_joint_deviation = torch.zeros_like(asset.data.joint_pos)

    for joint_name, w in deviation_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_joint_deviation[:, joint_idx] = torch.abs(angle[:, joint_idx]) * w
    return torch.sum(weighted_joint_deviation, dim=1)


def weighted_joint_power_l1(
    env: ManagerBasedRLEnv,
    power_weight: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint power applied on the articulation using L1 kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    weighted_power = torch.zeros_like(asset.data.applied_torque)

    for joint_name, w in power_weight.items():
        joint_idx = asset.find_joints(joint_name)[0]
        weighted_power[:, joint_idx] = (
            torch.abs(asset.data.applied_torque[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )

    return torch.sum(weighted_power, dim=1)


def feet_angle_slide(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the angler velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    feet_ang = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    command = env.command_manager.get_command(command_name)
    feet_ang_reward = torch.sum(torch.abs(feet_ang) * contacts, dim=1)
    reward = torch.where(torch.abs(command[:, 2]) > 0.1, feet_ang_reward, feet_ang_reward * 0.1)
    return reward


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]


        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.height_scale = float(cfg.params["tracking_contacts_shaped_height"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.height_sigma = cfg.params["gait_height_sigma"]
        self.stand_height = float(cfg.params.get("stand_height", 0.60))
        self.touch_down_vel = float(cfg.params["touch_down_vel"])
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt
        self.use_reference_motion = cfg.params["use_reference_motion"]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        tracking_contacts_shaped_height,
        gait_force_sigma,
        gait_vel_sigma,
        gait_height_sigma,
        touch_down_vel,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        use_reference_motion,
        stand_height=0.60,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """
        gait_params = env.command_manager.get_command(self.command_name)  # type: ignore
        gait_indices = env.command_manager.get_term(self.command_name).gait_indices  # type: ignore

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Update foot height targets
        self.compute_desired_foot_height(gait_params, gait_indices)

        # Force-based reward
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        total_reward = force_reward

        # Velocity-based reward
        if self.vel_scale != 0:
            foot_velocities = self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids]
            velocity_reward = self._compute_velocity_reward(
                foot_velocities, self.des_foot_velocity_z, desired_contact_states
            )
            total_reward += velocity_reward

        # Height-based reward
        if self.height_scale != 0:
            foot_heights = self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2]
            base_height = self.asset.data.root_pos_w[:, 2]
            height_reward = self._compute_height_reward(
                foot_heights,
                self.des_foot_height,
                desired_contact_states,
                base_height,
                stand_height,
            )
            total_reward += height_reward

        # stand still env , set to zero
        is_standing_env = env.command_manager.get_term("base_velocity").is_standing_env  # type: ignore

        no_gait_env_ids = is_standing_env.nonzero(as_tuple=False).flatten()

        total_reward[no_gait_env_ids] = 0.0
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all(
            (offsets >= 0) & (offsets <= 1)
        ), "Offsets must be between 0 and 1"
        assert torch.all(
            (durations > 0) & (durations < 1)
        ), "Durations must be between 0 and 1"

        # use gait indices from command
        command_term = self._env.command_manager.get_term("gait_command")  # type: ignore
        gait_indices = command_term.gait_indices  # type: ignore

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = torch.distributions.normal.Normal(
            0, self.kappa_gait_probs
        ).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (
            1 - smoothing_cdf_start(foot_indices - 1.5)
        )

        return desired_contact_states

    def compute_desired_foot_height(self, gait_params, gait_indices):
        """Calculate desired foot height for the current timestep."""
        frequencies = gait_params[:, 0]
        mask_0 = (gait_indices < 0.25) & (gait_indices >= 0.0)  # lift up
        mask_1 = (gait_indices < 0.5) & (gait_indices >= 0.25)  # touch down
        mask_2 = (gait_indices < 0.75) & (gait_indices >= 0.5)  # lift up
        mask_3 = (gait_indices <= 1.0) & (gait_indices >= 0.75)  # touch down
        swing_start_time = torch.zeros(self.num_envs, device=self.device)
        swing_start_time[mask_1] = 0.25 / frequencies[mask_1]
        swing_start_time[mask_2] = 0.5 / frequencies[mask_2]
        swing_start_time[mask_3] = 0.75 / frequencies[mask_3]
        swing_end_time = swing_start_time + 0.25 / frequencies
        swing_start_pos = torch.ones(self.num_envs, device=self.device)
        swing_start_pos[mask_0] = 0.0
        swing_start_pos[mask_2] = 0.0
        swing_end_pos = torch.ones(self.num_envs, device=self.device)
        swing_end_pos[mask_1] = 0.0
        swing_end_pos[mask_3] = 0.0
        swing_end_vel = torch.ones(self.num_envs, device=self.device)
        swing_end_vel[mask_0] = 0.0
        swing_end_vel[mask_2] = 0.0
        swing_end_vel[mask_1] = self.touch_down_vel
        swing_end_vel[mask_3] = self.touch_down_vel

        # generate desire foot z trajectory
        swing_height = gait_params[:, 3]

        start = {
            'time': swing_start_time,
            'position': swing_start_pos * swing_height,
            'velocity': torch.zeros(self.num_envs, device=self.device),
        }
        end = {
            'time': swing_end_time,
            'position': swing_end_pos * swing_height,
            'velocity': swing_end_vel,
        }
        cubic_spline = CubicSpline(start, end)
        self.des_foot_height = cubic_spline.position(gait_indices / frequencies)
        self.des_foot_velocity_z = cubic_spline.velocity(gait_indices / frequencies)

    def _compute_force_reward(
        self, forces: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (
                    1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma)
                )
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(
                    -forces[:, i] ** 2 / self.force_sigma
                )

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(
        self, foot_velocities: torch.Tensor, des_foot_velocities_z: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity-based reward component."""
        foot_velocity_norm = torch.norm(foot_velocities, dim=-1)
        reward = torch.zeros_like(foot_velocity_norm[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * (
                    1 - torch.exp(-foot_velocity_norm[:, i] ** 2 / self.vel_sigma)
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * (
                        1 - torch.exp(-((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2) / self.vel_sigma)
                    )
        else:  # Positive scale means reward movement during swing
            for i in range(foot_velocity_norm.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(
                    -foot_velocity_norm[:, i] ** 2 / self.vel_sigma
                )
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    reward += swing_phase * torch.exp(
                        -((foot_velocities[:, i, 2] - des_foot_velocities_z) ** 2) / self.vel_sigma
                    )

        return (reward / foot_velocity_norm.shape[1]) * self.vel_scale

    def _compute_height_reward(
        self,
        foot_heights: torch.Tensor,
        des_foot_height: torch.Tensor,
        desired_contacts: torch.Tensor,
        base_height: torch.Tensor,
        stand_height: float,
    ) -> torch.Tensor:
        """Compute foot-height reward relative to the current base height.

        The nominal stance-foot height is ``base_height - stand_height``. During
        swing, the reference gait trajectory is added on top of that nominal
        height. This makes the target follow the robot up and down terrain instead
        of incorrectly anchoring the feet to world z=0.
        """
        reward = torch.zeros_like(foot_heights[:, 0])
        nominal_foot_height = base_height - stand_height
        if self.height_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    desired_swing_height = nominal_foot_height + des_foot_height
                    reward += swing_phase * (
                        1 - torch.exp(-(foot_heights[:, i] - desired_swing_height) ** 2 / self.height_sigma)
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * (
                    1 - torch.exp(-(foot_heights[:, i] - nominal_foot_height) ** 2 / self.height_sigma)
                )
        else:  # Positive scale means reward movement during swing
            for i in range(foot_heights.shape[1]):
                if self.use_reference_motion:
                    swing_phase = 1 - desired_contacts[:, i]
                    desired_swing_height = nominal_foot_height + des_foot_height
                    reward += swing_phase * torch.exp(
                        -(foot_heights[:, i] - desired_swing_height) ** 2 / self.height_sigma
                    )
                stand_phase = desired_contacts[:, i]
                reward += stand_phase * torch.exp(
                    -(foot_heights[:, i] - nominal_foot_height) ** 2 / self.height_sigma
                )

        return (reward / foot_heights.shape[1]) * self.height_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """A reward term for penalizing large instantaneous changes in the network action output.

    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        current_action = env.action_manager.action.clone()
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)
        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0
        return penalty
