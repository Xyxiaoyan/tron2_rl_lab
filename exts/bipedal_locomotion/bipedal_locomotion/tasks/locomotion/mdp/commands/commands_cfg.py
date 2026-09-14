from dataclasses import MISSING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .centerline_velocity_command import CenterlineVelocityCommand
from .gait_command import GaitCommand  # Import the GaitCommand class


@configclass
class CenterlineVelocityCommandCfg(UniformVelocityCommandCfg):
    """Forward velocity command with closed-loop track-centerline steering."""

    class_type: type = CenterlineVelocityCommand

    lookahead_distance: float = 2.0
    """Distance in world X used to form the centerline look-ahead target [m]."""

    centerline_deadband: float = 0.05
    """Lateral error below which the target heading is exactly world +X [m]."""


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""
