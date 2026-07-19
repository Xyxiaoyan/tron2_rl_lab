"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Relative path to checkpoint file.")
parser.add_argument(
    "--terrain",
    type=str,
    default="flat",
    choices=["flat", "tron_camp", "humanoid_camp"],
    help="Terrain type for evaluation: flat (default), tron_camp, or humanoid_camp.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from rsl_rl.runner import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg,DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg, export_mlp_as_onnx, export_policy_as_jit


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    env_cfg.seed = agent_cfg.seed

    # override terrain if requested (camp terrain evaluation)
    # 重要：必须与 train.py 中的 camp 地形接入方式完全一致，否则训练/评估分布不匹配
    if args_cli.terrain != "flat":
        import sys as _sys
        _terrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training_terrain"))
        if _terrain_dir not in _sys.path:
            _sys.path.insert(0, _terrain_dir)
        if args_cli.terrain == "tron_camp":
            from tron_camp_training_terrain import TRON_CAMP_TRAINING_TERRAIN_CFG, TRON2_SPAWN_Z
            env_cfg.scene.terrain = TRON_CAMP_TRAINING_TERRAIN_CFG
            spawn_z = TRON2_SPAWN_Z
        elif args_cli.terrain == "humanoid_camp":
            from humanoid_camp_training_terrain import HUMANOID_CAMP_TRAINING_TERRAIN_CFG, OLI_SPAWN_Z
            env_cfg.scene.terrain = HUMANOID_CAMP_TRAINING_TERRAIN_CFG
            spawn_z = OLI_SPAWN_Z
        # 与 train.py 一致：让 spawn z = TRON2_SPAWN_Z（脚刚好接触地面，不从高空落下）
        # reset_root_state_uniform: positions = default_root_state + env_origins + rand_samples
        #   default_root_state.z = init_state.pos.z
        #   rand_samples.z = pose_range["z"] = TRON2_SPAWN_Z - init_state.pos.z
        _init_z = env_cfg.scene.robot.init_state.pos[2]
        _z_offset = spawn_z - _init_z
        env_cfg.events.reset_robot_base.params["pose_range"]["z"] = (_z_offset, _z_offset)
        # 增大环境间距，避免地形格子重叠（与 train.py 一致）
        env_cfg.scene.env_spacing = 10.0
        print(f"[INFO] Using terrain: {args_cli.terrain} (init_z={_init_z}, z_offset={_z_offset:.4f}, spawn z={spawn_z})")
    else:
        print("[INFO] Using terrain: flat")

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)
    sensor_encoder = ppo_runner.get_inference_sensor_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            ppo_runner.alg.actor_critic, export_model_dir
        )
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor, 
            export_model_dir, 
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )
        export_mlp_as_onnx(
            ppo_runner.alg.encoder,
            export_model_dir,
            "encoder",
            ppo_runner.alg.encoder.num_input_dim,
        )
    # reset environment
    obs_dict = env.get_observations()
    obs = obs_dict["policy"]
    obs_history = obs_dict.get("obsHistory")
    obs_history = obs_history.flatten(start_dim=1)
    commands = obs_dict.get("commands")
    sensor_latent = None

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # encode sensor if available
            if sensor_encoder is not None and "sensor" in obs_dict:
                sensor_obs = obs_dict["sensor"]
                sensor_img = sensor_obs["image_bundle"]
                sensor_lid = sensor_obs["lidar_bundle"]
                sensor_latent = sensor_encoder(sensor_img, sensor_lid)
            # agent stepping
            est = encoder(obs_history)
            actor_inputs = [est]
            if sensor_latent is not None:
                actor_inputs.append(sensor_latent)
            actor_inputs.extend([obs, commands])
            actions = policy(torch.cat(actor_inputs, dim=-1).detach())
            # env stepping
            obs_dict, _, _, infos = env.step(actions)
            obs = obs_dict["policy"]
            obs_history = obs_dict.get("obsHistory")
            obs_history = obs_history.flatten(start_dim=1)
            commands = obs_dict.get("commands") 

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = True
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
