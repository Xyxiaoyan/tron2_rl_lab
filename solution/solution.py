"""Submission solution for TRON2 SF (legged) — TronCamp Locomotion.

Combined model (policy.pt) takes:
  - obs_history (1, 420): 10 frames × 42-dim obs (for MLP_Encoder)
  - obs_vec (1, 42): current obs [ang_vel, proj_gravity, joint_pos, joint_vel,
    last_action, gait_phase, gait_cmd]
  - commands (1, 3): velocity commands from proprio[6:9]
  - image_bundle (1, 6144): 2 cameras × 4 ch × 24 × 32 (head+down, rgb+depth)
  - lidar_bundle (1, 1728): 24 × 72 downsampled LiDAR heights

Internally: encoder(420)→3 + sensor(96) + obs(42) + cmd(3) = 144 → ActorMLP → 10 leg actions
Arm joints (8) are zeroed (not trained).
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn.functional as F

POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policy.pt")

# Training gait command defaults (from limx_base_env_cfg.py)
GAIT_FREQ = 0.9
GAIT_OFFSET = 0.5
GAIT_DURATION = 0.5
GAIT_SWING_HEIGHT = 0.12

# Number of leg joints trained
NUM_LEG_JOINTS = 10
NUM_ARM_JOINTS = 8
ACTION_DIM = NUM_LEG_JOINTS + NUM_ARM_JOINTS  # 18

# Sensor preprocessing constants (match training observations.py)
IMG_H, IMG_W = 24, 32          # downsampled image size
LIDAR_ROWS, LIDAR_COLS = 24, 72  # downsampled lidar grid
LIDAR_CHANNELS = 96
LIDAR_MAX_DIST = 30.0


def _process_camera(rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Process one camera's RGB-D to match training format.

    Args:
        rgb: (1, H, W, 3) uint8 or float
        depth: (1, H, W, 1) float32 (meters, inf→0 for TRON2)

    Returns:
        (1, 4*24*32) = (1, 3072) flattened [rgb(3*24*32), depth(1*24*32)]
    """
    # RGB: normalize to [0,1], permute to (1, 3, H, W)
    rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
    # Depth: replace inf/nan with 0 (matches training observations.py), then normalize
    depth = depth.permute(0, 3, 1, 2).float()
    depth[torch.isinf(depth)] = 0.0
    depth[torch.isnan(depth)] = 0.0
    depth = depth / 10.0

    # Downsample to (24, 32)
    rgb_small = F.interpolate(rgb, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    depth_small = F.interpolate(depth, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
    # Guard against any NaN from interpolation
    depth_small = torch.nan_to_num(depth_small, nan=0.0, posinf=0.0, neginf=0.0)

    # Flatten and concat: (1, 3*24*32 + 1*24*32) = (1, 3072)
    return torch.cat([rgb_small.flatten(1), depth_small.flatten(1)], dim=-1)


def _process_lidar(extero: torch.Tensor) -> torch.Tensor:
    """Process LiDAR height scan to match training format.

    Args:
        extero: (1, 34560) = 96 channels × 360 rays, flattened

    Returns:
        (1, 1728) = 24 × 72 downsampled and normalized
    """
    n = extero.shape[0]
    # Replace inf/nan in lidar hits (ray misses can return inf/large values)
    extero = torch.nan_to_num(extero, nan=0.0, posinf=0.0, neginf=0.0)
    # Reshape to (1, 96, 360)
    z_hits = extero.view(n, LIDAR_CHANNELS, -1)  # (1, 96, 360)

    # Adaptive average-pool to (24, 72) — matches training sensor_lidar_bundle
    z_hits = z_hits.unsqueeze(1)  # (1, 1, 96, 360)
    z_hits = F.adaptive_avg_pool2d(z_hits, (LIDAR_ROWS, LIDAR_COLS))  # (1, 1, 24, 72)
    z_hits = z_hits.squeeze(1)  # (1, 24, 72)

    # Normalize by max_range and clamp
    z_hits = z_hits / LIDAR_MAX_DIST
    z_hits = z_hits.clamp(-1.0, 1.0)
    return z_hits.flatten(1)  # (1, 1728)


class AlgSolution:
    def __init__(self):
        self.device = torch.device("cpu")
        self.policy = None

        # --- Load combined model (encoder + sensor_encoder + actor) ---
        if not os.path.exists(POLICY_PATH):
            print("[solution] WARNING: policy.pt not found, running with zeros")
        else:
            self.policy = torch.jit.load(POLICY_PATH, map_location=self.device).eval()

        # --- State buffers ---
        self.obs_history = None
        self.last_leg_action = torch.zeros(1, NUM_LEG_JOINTS, device=self.device)
        self.step_count = 0

    def reset(self, **kwargs):
        """Called at episode start."""
        self.obs_history = None
        self.last_leg_action = torch.zeros(1, NUM_LEG_JOINTS, device=self.device)
        self.step_count = 0

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        """Match training action scale (0.25 for leg, 0.5 for arm)."""
        return {
            "leg": {"mode": "position", "scale": 0.25},
            "arm": {"mode": "position", "scale": 0.5},
        }

    def predicts(self, obs, current_score) -> dict:
        proprio = torch.as_tensor(obs["proprio"], dtype=torch.float32, device=self.device)
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)

        # --- Extract components from 66-dim proprio ---
        ang_vel = proprio[:, 3:6]
        proj_gravity = proprio[:, 9:12]
        joint_pos = proprio[:, 12:12 + NUM_LEG_JOINTS]
        joint_vel = proprio[:, 30:30 + NUM_LEG_JOINTS]

        # --- Gait phase (self-integrated, matches training) ---
        phase = (self.step_count * GAIT_FREQ * 0.02) % 1.0
        gait_phase = torch.tensor(
            [[math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]],
            dtype=torch.float32, device=self.device,
        )
        gait_cmd = torch.tensor(
            [[GAIT_FREQ, GAIT_OFFSET, GAIT_DURATION, GAIT_SWING_HEIGHT]],
            dtype=torch.float32, device=self.device,
        )

        # --- Build 42-dim training obs ---
        obs_vec = torch.cat([
            ang_vel, proj_gravity, joint_pos, joint_vel,
            self.last_leg_action, gait_phase, gait_cmd,
        ], dim=-1)

        # --- Update 10-frame history buffer ---
        if self.obs_history is None:
            self.obs_history = obs_vec.repeat(1, 10)
        else:
            self.obs_history = torch.cat([self.obs_history[:, 42:], obs_vec], dim=-1)

        # --- Process sensor data ---
        images = obs.get("image") or {}
        head_rgb = images.get("head_rgb")
        head_depth = images.get("head_depth")
        down_rgb = images.get("down_rgb")
        down_depth = images.get("down_depth")
        extero = obs.get("extero")

        image_bundle = None
        lidar_bundle = None

        # Build image bundle if all cameras present
        if all(x is not None for x in [head_rgb, head_depth, down_rgb, down_depth]):
            head_rgb = torch.as_tensor(head_rgb, dtype=torch.float32, device=self.device)
            head_depth = torch.as_tensor(head_depth, dtype=torch.float32, device=self.device)
            down_rgb = torch.as_tensor(down_rgb, dtype=torch.float32, device=self.device)
            down_depth = torch.as_tensor(down_depth, dtype=torch.float32, device=self.device)

            head_cam = _process_camera(head_rgb, head_depth)   # (1, 3072)
            down_cam = _process_camera(down_rgb, down_depth)   # (1, 3072)
            image_bundle = torch.cat([head_cam, down_cam], dim=-1)  # (1, 6144)

        # Build lidar bundle if extero present
        if extero is not None:
            extero = torch.as_tensor(extero, dtype=torch.float32, device=self.device)
            lidar_bundle = _process_lidar(extero)  # (1, 1728)

        # --- Fallback to zeros if sensors missing ---
        if image_bundle is None:
            image_bundle = torch.zeros(1, 6144, device=self.device)
        if lidar_bundle is None:
            lidar_bundle = torch.zeros(1, 1728, device=self.device)

        # --- Velocity commands from proprio ---
        commands = proprio[:, 6:9]

        # --- Inference ---
        if self.policy is not None:
            with torch.inference_mode():
                leg_action = self.policy(self.obs_history, obs_vec, commands,
                                         image_bundle, lidar_bundle)
        else:
            leg_action = torch.zeros(1, NUM_LEG_JOINTS, device=self.device)

        # Guard against NaN/Inf in action output (causes JSON serialization error)
        leg_action = torch.nan_to_num(leg_action, nan=0.0, posinf=0.0, neginf=0.0)
        # Clamp to reasonable range (training used clip_actions=1.0)
        leg_action = leg_action.clamp(-1.0, 1.0)

        # --- Build 18-dim full action (leg 10 + arm 8) ---
        arm_action = torch.zeros(1, NUM_ARM_JOINTS, device=self.device)
        full_action = torch.cat([leg_action, arm_action], dim=-1)

        # Save for next step's obs
        self.last_leg_action = leg_action
        self.step_count += 1

        action_list = full_action.squeeze(0).cpu().tolist()
        return {"action": action_list, "giveup": False}
