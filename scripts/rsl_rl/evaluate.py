"""Script to evaluate a trained RL agent checkpoint with RSL-RL.

Runs a fixed number of episodes, reports metrics, applies gate thresholds,
and exits (no infinite loop).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import json
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent checkpoint with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the model checkpoint (.pt).")
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "camp"],
                    help="Terrain type: flat (plane) or camp (TRON_CAMP generator).")
parser.add_argument("--gate_stage", type=int, default=None, choices=[1, 2, 3],
                    help="Apply acceptance gate thresholds for the given stage. Exit 0 if passed, 1 if failed.")
parser.add_argument("--output_json", type=str, default=None, help="Optional path to save metrics JSON.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras (SF env always has head/down TiledCamera + lidar).
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401


def main():
    # Parse env cfg
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )

    # Terrain selection (mirror train.py semantics)
    if args_cli.terrain == "camp":
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training_terrain")))
        from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG, TRON2_SPAWN_Z

        env_cfg.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
        _init_z = env_cfg.scene.robot.init_state.pos[2]
        _z_offset = TRON2_SPAWN_Z - _init_z
        env_cfg.events.reset_robot_base.params["pose_range"]["z"] = (_z_offset, _z_offset)
        env_cfg.scene.env_spacing = 10.0
        print(f"[INFO] Camp terrain: init_state.pos.z={_init_z}, pose_range z offset={_z_offset:.4f}, spawn z={TRON2_SPAWN_Z}")
    else:
        print("[INFO] Using flat terrain")

    # Resolve checkpoint path
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", ".")
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, None, None)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path) if resume_path else "."

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during evaluation.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
    sensor_encoder = ppo_runner.get_inference_sensor_encoder(device=env.unwrapped.device)

    # Collect metrics
    num_episodes = 0
    episode_rewards = []
    episode_lengths = []
    commanded_velocities = []
    actual_velocities = []
    stand_still_penalties = []

    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1) if obs_history is not None else None
    commands = obs_dict.get("commands")
    sensor_latent = None

    ep_reward = torch.zeros(env.num_envs, device=agent_cfg.device)
    ep_len = torch.zeros(env.num_envs, device=agent_cfg.device)
    ep_vel_cmd = torch.zeros(env.num_envs, 3, device=agent_cfg.device)
    ep_vel_act = torch.zeros(env.num_envs, 3, device=agent_cfg.device)
    ep_vel_count = torch.zeros(env.num_envs, device=agent_cfg.device)
    done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=agent_cfg.device)

    total_steps = 0
    max_steps = args_cli.num_episodes * 500  # safety cap

    while simulation_app.is_running():
        with torch.inference_mode():
            # Sensor encoder
            if sensor_encoder is not None and "sensor" in obs_dict:
                sensor_obs = obs_dict["sensor"]
                sensor_img = sensor_obs["image_bundle"]
                sensor_lid = sensor_obs["lidar_bundle"]
                sensor_latent = sensor_encoder(sensor_img, sensor_lid)

            # Actor forward
            est = encoder(obs_history)
            actor_inputs = [est]
            if sensor_latent is not None:
                actor_inputs.append(sensor_latent)
            actor_inputs.extend([obs, commands])
            actions = policy(torch.cat(actor_inputs, dim=-1).detach())

            # Env step
            obs_dict, rewards, dones, infos = env.step(actions)
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1) if obs_history is not None else None
            commands = obs_dict.get("commands")

            # Track per-episode metrics
            ep_reward += rewards
            ep_len += 1
            if commands is not None:
                ep_vel_cmd += commands[:, :3]
            # Root linear velocity from encoder estimate (3-dim)
            ep_vel_act += est[:, :3]
            ep_vel_count += 1

            # Detect episode end
            done_mask = dones > 0
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).flatten()
                for idx in done_ids:
                    episode_rewards.append(ep_reward[idx].item())
                    episode_lengths.append(ep_len[idx].item())
                    if ep_vel_count[idx] > 0:
                        commanded_velocities.append(ep_vel_cmd[idx].cpu() / ep_vel_count[idx])
                        actual_velocities.append(ep_vel_act[idx].cpu() / ep_vel_count[idx])
                    # Track stand-still penalty for Stage 1 evaluation
                    if "stand_still_penalty" in infos:
                        stand_still_penalties.append(infos["stand_still_penalty"][idx].item() if hasattr(infos["stand_still_penalty"], "item") else infos["stand_still_penalty"][idx])
                    ep_reward[idx] = 0
                    ep_len[idx] = 0
                    ep_vel_cmd[idx] = 0
                    ep_vel_act[idx] = 0
                    ep_vel_count[idx] = 0
                    num_episodes += 1

            total_steps += 1
            if num_episodes >= args_cli.num_episodes or total_steps > max_steps:
                break

    env.close()

    # Compute aggregate metrics
    n = len(episode_rewards)
    if n == 0:
        print("[WARN] No episodes completed. Metrics unavailable.")
        return

    mean_reward = np.mean(episode_rewards)
    mean_ep_len = np.mean(episode_lengths)
    mean_ep_len_sec = mean_ep_len * env_cfg.decimation * env_cfg.sim.dt

    mean_vel_cmd = np.mean(commanded_velocities, axis=0) if commanded_velocities else np.zeros(3)
    mean_vel_act = np.mean(actual_velocities, axis=0) if actual_velocities else np.zeros(3)
    vel_tracking_error = np.linalg.norm(mean_vel_cmd - mean_vel_act) if len(commanded_velocities) > 0 else float("nan")

    metrics = {
        "num_episodes": n,
        "mean_reward": round(mean_reward, 2),
        "mean_episode_length": round(mean_ep_len, 1),
        "mean_episode_length_sec": round(mean_ep_len_sec, 2),
        "mean_vel_cmd_x": round(mean_vel_cmd[0], 3),
        "mean_vel_cmd_y": round(mean_vel_cmd[1], 3),
        "mean_vel_cmd_z": round(mean_vel_cmd[2], 3),
        "mean_vel_act_x": round(mean_vel_act[0], 3),
        "mean_vel_act_y": round(mean_vel_act[1], 3),
        "mean_vel_act_z": round(mean_vel_act[2], 3),
        "vel_tracking_error": round(vel_tracking_error, 3),
    }

    if stand_still_penalties:
        metrics["mean_stand_still_penalty"] = round(np.mean(stand_still_penalties), 4)

    # Print summary
    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")
    print("=" * 60)

    # Save JSON
    if args_cli.output_json:
        os.makedirs(os.path.dirname(args_cli.output_json) or ".", exist_ok=True)
        with open(args_cli.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Metrics saved to: {args_cli.output_json}")

    # Gate thresholds
    if args_cli.gate_stage is not None:
        passed = True
        if args_cli.gate_stage == 1:
            if mean_ep_len_sec < 8.0:
                print(f"[FAIL] Stage 1 gate: mean_episode_length_sec={mean_ep_len_sec:.2f} < 8.0")
                passed = False
            if mean_reward < -10.0:
                print(f"[FAIL] Stage 1 gate: mean_reward={mean_reward:.2f} < -10.0")
                passed = False
            if "mean_stand_still_penalty" in metrics and metrics["mean_stand_still_penalty"] > 0.1:
                print(f"[FAIL] Stage 1 gate: stand_still_penalty={metrics['mean_stand_still_penalty']:.4f} > 0.1")
                passed = False
        elif args_cli.gate_stage == 2:
            if mean_ep_len_sec < 15.0:
                print(f"[FAIL] Stage 2 gate: mean_episode_length_sec={mean_ep_len_sec:.2f} < 15.0")
                passed = False
            if abs(mean_vel_act[0]) < 0.3:
                print(f"[FAIL] Stage 2 gate: forward velocity={mean_vel_act[0]:.3f} < 0.3")
                passed = False
            if vel_tracking_error > 0.5:
                print(f"[FAIL] Stage 2 gate: vel_tracking_error={vel_tracking_error:.3f} > 0.5")
                passed = False
        elif args_cli.gate_stage == 3:
            if mean_ep_len_sec < 12.0:
                print(f"[FAIL] Stage 3 gate: mean_episode_length_sec={mean_ep_len_sec:.2f} < 12.0")
                passed = False
            if abs(mean_vel_act[0]) < 0.1:
                print(f"[FAIL] Stage 3 gate: forward velocity={mean_vel_act[0]:.3f} < 0.1")
                passed = False

        if passed:
            print("[PASS] Gate thresholds passed.")
            # Exit 0 — success
            sys.exit(0)
        else:
            print("[FAIL] Gate thresholds NOT passed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()