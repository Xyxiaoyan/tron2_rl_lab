"""Distill four unified SF_TRON2A teacher policies into one student."""

import argparse
import os
import sys

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rsl_rl")))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Online multi-teacher distillation for SF_TRON2A.")
parser.add_argument("--task", default="Isaac-Limx-SF-TRON2A-MultiTeacher-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--distill_iterations", type=int, default=None)
parser.add_argument("--save_interval", type=int, default=None)
parser.add_argument("--teacher_continuous", required=True)
parser.add_argument("--teacher_stairs", required=True)
parser.add_argument("--teacher_obstacle", required=True)
parser.add_argument("--teacher_gap", required=True)
parser.add_argument("--student_init", default=None, help="Defaults to the continuous teacher checkpoint.")
parser.add_argument("--resume_distillation", default=None)
parser.add_argument("--beta_start", type=float, default=1.0)
parser.add_argument("--beta_end", type=float, default=0.25)
parser.add_argument("--updates_per_iteration", type=int, default=20)
parser.add_argument("--distill_batch_size", type=int, default=4096)
parser.add_argument("--replay_capacity_per_skill", type=int, default=25000)
parser.add_argument("--distill_learning_rate", type=float, default=1.0e-4)
parser.add_argument("--estimation_coef", type=float, default=0.25)
parser.add_argument(
    "--specialist_relief_start",
    type=float,
    default=0.03,
    help="Local height relief in metres where a specialist starts replacing the continuous teacher.",
)
parser.add_argument(
    "--specialist_relief_full",
    type=float,
    default=0.10,
    help="Local height relief in metres where the terrain specialist is used fully.",
)
parser.add_argument(
    "--student_probe_fraction",
    type=float,
    default=0.10,
    help="Fraction of environments always controlled by the student for stability measurement.",
)
parser.add_argument(
    "--max_probe_error",
    type=float,
    default=0.08,
    help="Probe Smooth-L1 threshold above which teacher control is automatically restored.",
)
parser.add_argument(
    "--min_specialist_samples_for_best",
    type=int,
    default=1000,
    help="Required active samples from every specialist before model_best.pt selection starts.",
)
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runner import MultiTeacherRunner

# Importing the extension registers all Gym tasks.
import bipedal_locomotion.tasks  # noqa: F401,E402


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval
    iterations = args_cli.distill_iterations or agent_cfg.max_iterations

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Distillation logs: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)
    env.seed(agent_cfg.seed)

    teacher_paths = {
        "continuous": os.path.abspath(args_cli.teacher_continuous),
        "stairs": os.path.abspath(args_cli.teacher_stairs),
        "obstacle": os.path.abspath(args_cli.teacher_obstacle),
        "gap": os.path.abspath(args_cli.teacher_gap),
    }
    runner = MultiTeacherRunner(
        env=env,
        train_cfg=agent_cfg.to_dict(),
        teacher_checkpoints=teacher_paths,
        log_dir=log_dir,
        device=agent_cfg.device,
        replay_capacity_per_skill=args_cli.replay_capacity_per_skill,
        learning_rate=args_cli.distill_learning_rate,
        estimation_coef=args_cli.estimation_coef,
        specialist_relief_start=args_cli.specialist_relief_start,
        specialist_relief_full=args_cli.specialist_relief_full,
    )
    if args_cli.resume_distillation:
        runner.resume(os.path.abspath(args_cli.resume_distillation))
    else:
        runner.initialize_student(os.path.abspath(args_cli.student_init or args_cli.teacher_continuous))

    runner.learn(
        num_iterations=iterations,
        beta_start=args_cli.beta_start,
        beta_end=args_cli.beta_end,
        updates_per_iteration=args_cli.updates_per_iteration,
        batch_size=args_cli.distill_batch_size,
        student_probe_fraction=args_cli.student_probe_fraction,
        max_probe_error=args_cli.max_probe_error,
        min_specialist_samples_for_best=args_cli.min_specialist_samples_for_best,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
