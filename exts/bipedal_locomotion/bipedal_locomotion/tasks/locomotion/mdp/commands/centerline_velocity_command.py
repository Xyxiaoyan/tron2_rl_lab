"""Velocity command that steers a robot back toward its terrain-track centerline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import CenterlineVelocityCommandCfg


class CenterlineVelocityCommand(UniformVelocityCommand):
    """Generate a forward command with closed-loop centerline heading correction.

    Terrain tracks are assumed to run along world ``+X`` and each environment's
    origin is assumed to lie on its track centerline. The target heading points
    from the robot toward a look-ahead point on that centerline. The inherited
    heading controller converts this target into the third command component,
    so the policy command remains ``[v_x, v_y, w_z]``.
    """

    cfg: CenterlineVelocityCommandCfg

    def __init__(self, cfg: CenterlineVelocityCommandCfg, env: ManagerBasedEnv):
        if not cfg.heading_command:
            raise ValueError("CenterlineVelocityCommand requires heading_command=True.")
        if cfg.lookahead_distance <= 0.0:
            raise ValueError("lookahead_distance must be greater than zero.")
        if cfg.centerline_deadband < 0.0:
            raise ValueError("centerline_deadband must be non-negative.")

        super().__init__(cfg, env)
        self.metrics["error_centerline"] = torch.zeros(self.num_envs, device=self.device)

    def _centerline_error(self) -> torch.Tensor:
        """Return signed lateral displacement from each environment centerline."""
        return self.robot.data.root_pos_w[:, 1] - self._env.scene.env_origins[:, 1]

    def _update_metrics(self):
        super()._update_metrics()
        max_command_step = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["error_centerline"] += torch.abs(self._centerline_error()) / max_command_step

    def _update_command(self):
        lateral_error = self._centerline_error()

        # Aim at a point ``lookahead_distance`` ahead on the world-X centerline.
        # Positive lateral error therefore produces a negative target heading,
        # and vice versa. A deadband avoids constant small yaw oscillations.
        lookahead = torch.full_like(lateral_error, self.cfg.lookahead_distance)
        centerline_heading = torch.atan2(-lateral_error, lookahead)
        centerline_heading = torch.where(
            torch.abs(lateral_error) <= self.cfg.centerline_deadband,
            torch.zeros_like(centerline_heading),
            centerline_heading,
        )
        self.heading_target[:] = centerline_heading

        super()._update_command()
