"""Online multi-teacher behavior distillation with DAgger-style control mixing."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from ..modules import ActorCritic, MLP_Encoder
from ..storage.distillation_storage import BalancedDistillationReplay


TEACHER_NAMES = ("continuous", "stairs", "obstacle", "gap")


def specialist_blend_weights(
    terrain_profile: torch.Tensor,
    skill_ids: torch.Tensor,
    relief_start: float,
    relief_full: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Blend from the continuous teacher only when terrain relief is visible.

    All specialist tracks contain flat lead-in segments.  Those segments are
    indistinguishable to the deployable student, so asking four independently
    fine-tuned teachers for four different actions creates an impossible
    regression target.  Local scan relief provides a deployable routing signal:
    flat observations use the continuous teacher, while visible geometry
    smoothly activates the terrain-family specialist.
    """
    if terrain_profile.ndim != 2 or terrain_profile.shape[1] < 2:
        raise ValueError(f"Expected a 2-D terrain profile with at least two rays, got {terrain_profile.shape}.")
    if relief_start < 0.0 or relief_full <= relief_start:
        raise ValueError(
            f"Expected 0 <= relief_start < relief_full, got {relief_start=} and {relief_full=}.")
    profile = torch.nan_to_num(terrain_profile, nan=0.0, posinf=1.0, neginf=-1.0)
    relief = profile.amax(dim=1) - profile.amin(dim=1)
    weights = torch.clamp((relief - relief_start) / (relief_full - relief_start), 0.0, 1.0)
    weights = torch.where(skill_ids == 0, torch.zeros_like(weights), weights)
    return weights, relief


def _actor_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.startswith("actor.") or key == "logstd"}


def _load_actor_encoder_strict(
    actor_critic: ActorCritic,
    encoder: MLP_Encoder,
    checkpoint_path: str,
    load_critic: bool = False,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint or "encoder_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing model or encoder state dictionaries.")

    expected_actor = _actor_state(actor_critic.state_dict())
    checkpoint_actor = _actor_state(checkpoint["model_state_dict"])
    missing = sorted(set(expected_actor) - set(checkpoint_actor))
    unexpected = sorted(set(checkpoint_actor) - set(expected_actor))
    mismatched = {
        key: (tuple(checkpoint_actor[key].shape), tuple(expected_actor[key].shape))
        for key in expected_actor.keys() & checkpoint_actor.keys()
        if checkpoint_actor[key].shape != expected_actor[key].shape
    }
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Incompatible teacher checkpoint '{checkpoint_path}'. "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
        )
    actor_critic.load_state_dict(checkpoint_actor, strict=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)

    if load_critic:
        current = actor_critic.state_dict()
        compatible = {
            key: value
            for key, value in checkpoint["model_state_dict"].items()
            if key.startswith("critic.") and key in current and value.shape == current[key].shape
        }
        actor_critic.load_state_dict(compatible, strict=False)
    return checkpoint


@dataclass
class FrozenTeacher:
    name: str
    encoder: MLP_Encoder
    actor_critic: ActorCritic

    @torch.inference_mode()
    def action(self, obs: torch.Tensor, history: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.inference(history)
        actor_input = torch.cat((latent, obs, commands), dim=-1)
        return self.actor_critic.act_inference(actor_input)


class MultiTeacherRunner:
    """Train one student against four frozen policies in a balanced terrain grid."""

    def __init__(
        self,
        env,
        train_cfg: dict,
        teacher_checkpoints: dict[str, str],
        log_dir: str,
        device: str = "cpu",
        replay_capacity_per_skill: int = 25_000,
        learning_rate: float = 1.0e-4,
        estimation_coef: float = 0.25,
        max_grad_norm: float = 1.0,
        specialist_relief_start: float = 0.03,
        specialist_relief_full: float = 0.10,
    ):
        self.env = env
        self.device = device
        self.log_dir = log_dir
        self.cfg = copy.deepcopy(train_cfg)
        self.estimation_coef = estimation_coef
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.specialist_relief_start = specialist_relief_start
        self.specialist_relief_full = specialist_relief_full
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg["save_interval"])
        self.current_iteration = 0
        self.best_probe_error = float("inf")
        self.online_error_ema = None
        self.specialist_active_counts = [0] * len(TEACHER_NAMES)

        observation_dict = env.get_observations()
        for key in ("policy", "obsHistory", "commands", "critic", "teacher", "terrain_route"):
            if key not in observation_dict:
                raise KeyError(f"Multi-teacher environment is missing observation group '{key}'.")
        initial_skill_ids = observation_dict["teacher"].flatten().to(dtype=torch.long)
        present_skills = set(torch.unique(initial_skill_ids).tolist())
        expected_skills = set(range(len(TEACHER_NAMES)))
        if present_skills != expected_skills:
            raise RuntimeError(
                f"The distillation batch must contain all four skills; got {sorted(present_skills)}. "
                "Use at least 8 environments and keep num_envs divisible by 8."
            )
        self.obs_dim = observation_dict["policy"].shape[1]
        self.history_dim = observation_dict["obsHistory"].flatten(start_dim=1).shape[1]
        self.command_dim = observation_dict["commands"].shape[1]
        self.critic_dim = observation_dict["critic"].shape[1] + self.command_dim
        self.action_dim = env.num_actions

        encoder_cfg = copy.deepcopy(self.cfg["encoder"])
        encoder_cfg["num_input_dim"] = self.history_dim
        policy_cfg = copy.deepcopy(self.cfg["policy"])
        self.student_encoder = MLP_Encoder(**encoder_cfg).to(device)
        self.student_actor_critic = ActorCritic(
            self.obs_dim + self.student_encoder.num_output_dim + self.command_dim,
            self.critic_dim,
            self.action_dim,
            **policy_cfg,
        ).to(device)

        absent = set(TEACHER_NAMES) - set(teacher_checkpoints)
        if absent:
            raise ValueError(f"Missing teacher checkpoints for: {sorted(absent)}")
        self.teachers: list[FrozenTeacher] = []
        for name in TEACHER_NAMES:
            teacher_encoder = MLP_Encoder(**copy.deepcopy(encoder_cfg)).to(device)
            teacher_actor = ActorCritic(
                self.obs_dim + teacher_encoder.num_output_dim + self.command_dim,
                self.critic_dim,
                self.action_dim,
                **copy.deepcopy(policy_cfg),
            ).to(device)
            _load_actor_encoder_strict(teacher_actor, teacher_encoder, teacher_checkpoints[name])
            teacher_encoder.eval().requires_grad_(False)
            teacher_actor.eval().requires_grad_(False)
            self.teachers.append(FrozenTeacher(name, teacher_encoder, teacher_actor))

        self.optimizer = torch.optim.Adam(
            list(self.student_actor_critic.actor.parameters()) + list(self.student_encoder.parameters()),
            lr=learning_rate,
        )
        self.replay = BalancedDistillationReplay(
            num_skills=len(TEACHER_NAMES),
            capacity_per_skill=replay_capacity_per_skill,
            obs_dim=self.obs_dim,
            history_dim=self.history_dim,
            command_dim=self.command_dim,
            action_dim=self.action_dim,
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
        env.reset()

    def initialize_student(self, checkpoint_path: str) -> None:
        _load_actor_encoder_strict(
            self.student_actor_critic,
            self.student_encoder,
            checkpoint_path,
            load_critic=True,
        )
        print(f"[INFO] Initialized student from {checkpoint_path}")

    def resume(self, checkpoint_path: str) -> None:
        checkpoint = _load_actor_encoder_strict(
            self.student_actor_critic,
            self.student_encoder,
            checkpoint_path,
            load_critic=True,
        )
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        self.current_iteration = int(checkpoint.get("iter", 0))
        infos = checkpoint.get("infos", {})
        # distill.py creates a new log directory on resume.  Select and save a
        # fresh best checkpoint for this continuation rather than referring to
        # a model_best.pt that only exists in the previous directory.
        self.best_probe_error = float("inf")
        self.online_error_ema = infos.get("online_error_ema")
        saved_counts = infos.get("specialist_active_counts", [0] * len(TEACHER_NAMES))
        if len(saved_counts) != len(TEACHER_NAMES):
            raise RuntimeError(f"Invalid specialist_active_counts in checkpoint: {saved_counts}")
        self.specialist_active_counts = [int(value) for value in saved_counts]
        print(f"[INFO] Resumed distillation at iteration {self.current_iteration}")

    def _teacher_actions(self, obs, history, commands, skill_ids, terrain_profile):
        continuous_actions = self.teachers[0].action(obs, history, commands)
        actions = continuous_actions.clone()
        blend_weights, relief = specialist_blend_weights(
            terrain_profile,
            skill_ids,
            self.specialist_relief_start,
            self.specialist_relief_full,
        )
        for skill_id, teacher in enumerate(self.teachers[1:], start=1):
            mask = skill_ids == skill_id
            if mask.any():
                specialist_actions = teacher.action(obs[mask], history[mask], commands[mask])
                weight = blend_weights[mask].unsqueeze(1)
                actions[mask] = torch.lerp(continuous_actions[mask], specialist_actions, weight)
        invalid = (skill_ids < 0) | (skill_ids >= len(self.teachers))
        if invalid.any():
            values = torch.unique(skill_ids[invalid]).tolist()
            raise RuntimeError(f"Invalid terrain teacher ids returned by environment: {values}")
        return actions, blend_weights, relief

    def _student_actions(self, obs, history, commands):
        # Use forward rather than encode so behavior cloning can train the
        # history encoder even when the PPO config uses output_detach=True.
        latent = self.student_encoder(history)
        actor_input = torch.cat((latent, obs, commands), dim=-1)
        return self.student_actor_critic.actor(actor_input), latent

    def _update(self, batch_size: int, num_updates: int):
        mean_bc = 0.0
        mean_estimation = 0.0
        for _ in range(num_updates):
            obs, history, commands, targets, velocity_targets, skill_ids = self.replay.balanced_batch(
                batch_size, self.device
            )
            predictions, latent = self._student_actions(obs, history, commands)
            skill_losses = []
            for skill_id in range(len(self.teachers)):
                mask = skill_ids == skill_id
                if mask.any():
                    skill_losses.append(F.smooth_l1_loss(predictions[mask], targets[mask]))
            bc_loss = torch.stack(skill_losses).mean()
            if latent.shape[1] >= 3:
                estimation_loss = F.mse_loss(latent[:, :3], velocity_targets)
            else:
                estimation_loss = torch.zeros((), device=self.device)
            loss = bc_loss + self.estimation_coef * estimation_loss

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(self.student_actor_critic.actor.parameters()) + list(self.student_encoder.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()
            mean_bc += float(bc_loss.item())
            mean_estimation += float(estimation_loss.item())
        return mean_bc / num_updates, mean_estimation / num_updates

    def learn(
        self,
        num_iterations: int,
        beta_start: float = 1.0,
        beta_end: float = 0.25,
        updates_per_iteration: int = 20,
        batch_size: int = 4096,
        student_probe_fraction: float = 0.10,
        max_probe_error: float = 0.08,
        min_specialist_samples_for_best: int = 1_000,
    ) -> None:
        if not 0.0 <= beta_end <= beta_start <= 1.0:
            raise ValueError("Expected 0 <= beta_end <= beta_start <= 1.")
        if num_iterations <= 0 or updates_per_iteration <= 0 or batch_size <= 0:
            raise ValueError("num_iterations, updates_per_iteration and batch_size must be positive.")
        if not 0.0 < student_probe_fraction < 1.0:
            raise ValueError("student_probe_fraction must be between zero and one.")
        if max_probe_error <= 0.0:
            raise ValueError("max_probe_error must be positive.")
        if min_specialist_samples_for_best <= 0:
            raise ValueError("min_specialist_samples_for_best must be positive.")
        observations = self.env.get_observations()
        start_iteration = self.current_iteration
        final_iteration = start_iteration + num_iterations

        for iteration in range(start_iteration, final_iteration):
            started = time.time()
            fraction = (iteration - start_iteration) / max(1, num_iterations - 1)
            scheduled_beta = beta_start + fraction * (beta_end - beta_start)
            beta = scheduled_beta
            # If student-only probe environments become unstable, temporarily
            # restore more teacher control instead of continuing the collapse.
            if self.online_error_ema is not None and self.online_error_ema > max_probe_error:
                recovery_beta = beta_end + (self.online_error_ema - max_probe_error) / (2.0 * max_probe_error)
                beta = max(beta, min(1.0, recovery_beta))
            # Keep one controller for a rollout (and resample on episode reset)
            # rather than blending joint targets at every control step.
            teacher_control = torch.rand(self.env.num_envs, device=self.device) < beta
            student_probe = torch.rand(self.env.num_envs, device=self.device) < student_probe_fraction
            rollout_skill_ids = observations["teacher"].flatten().to(self.device, dtype=torch.long)
            for skill_id in range(len(self.teachers)):
                candidates = torch.nonzero(rollout_skill_ids == skill_id, as_tuple=False).flatten()
                if candidates.numel() and not student_probe[candidates].any():
                    student_probe[candidates[0]] = True
            # Always retain student-controlled rollouts.  They measure actual
            # deployment error even while the safeguard raises teacher control.
            teacher_control[student_probe] = False
            teacher_fraction_sum = 0.0
            action_error_sum = 0.0
            probe_error_sum = 0.0
            probe_error_steps = 0
            specialist_weight_sum = 0.0
            relief_sum = 0.0

            for _ in range(self.num_steps_per_env):
                obs = observations["policy"].to(self.device)
                history = observations["obsHistory"].flatten(start_dim=1).to(self.device)
                commands = observations["commands"].to(self.device)
                critic = observations["critic"].to(self.device)
                skill_ids = observations["teacher"].flatten().to(self.device, dtype=torch.long)
                terrain_profile = observations["terrain_route"].to(self.device)

                with torch.inference_mode():
                    teacher_actions, specialist_weights, relief = self._teacher_actions(
                        obs,
                        history,
                        commands,
                        skill_ids,
                        terrain_profile,
                    )
                    student_actions, _ = self._student_actions(obs, history, commands)
                self.replay.add(obs, history, commands, teacher_actions, critic[:, :3], skill_ids)
                executed = torch.where(teacher_control.unsqueeze(1), teacher_actions, student_actions)
                observations, _, dones, _ = self.env.step(executed)

                done_mask = dones.flatten().to(device=self.device, dtype=torch.bool)
                if done_mask.any():
                    num_done = int(done_mask.sum().item())
                    teacher_control[done_mask] = torch.rand(num_done, device=self.device) < beta
                    teacher_control[done_mask & student_probe] = False
                per_env_error = F.smooth_l1_loss(student_actions, teacher_actions, reduction="none").mean(dim=1)
                teacher_fraction_sum += float(teacher_control.float().mean().item())
                action_error_sum += float(per_env_error.mean().item())
                if student_probe.any():
                    probe_error_sum += float(per_env_error[student_probe].mean().item())
                    probe_error_steps += 1
                specialist_weight_sum += float(specialist_weights.mean().item())
                relief_sum += float(relief.mean().item())
                for skill_id in range(1, len(self.teachers)):
                    active = (skill_ids == skill_id) & (specialist_weights >= 0.5)
                    self.specialist_active_counts[skill_id] += int(active.sum().item())

            mean_bc, mean_estimation = self._update(batch_size, updates_per_iteration)
            elapsed = time.time() - started
            teacher_fraction = teacher_fraction_sum / self.num_steps_per_env
            action_error = action_error_sum / self.num_steps_per_env
            probe_error = probe_error_sum / max(1, probe_error_steps)
            specialist_weight = specialist_weight_sum / self.num_steps_per_env
            mean_relief = relief_sum / self.num_steps_per_env
            if self.online_error_ema is None:
                self.online_error_ema = probe_error
            else:
                self.online_error_ema = 0.95 * self.online_error_ema + 0.05 * probe_error
            self.current_iteration = iteration + 1

            self.writer.add_scalar("Distillation/bc_loss", mean_bc, iteration)
            self.writer.add_scalar("Distillation/online_action_error", action_error, iteration)
            self.writer.add_scalar("Distillation/student_probe_error", probe_error, iteration)
            self.writer.add_scalar("Distillation/student_probe_error_ema", self.online_error_ema, iteration)
            self.writer.add_scalar("Distillation/velocity_estimation", mean_estimation, iteration)
            self.writer.add_scalar("DAgger/beta", beta, iteration)
            self.writer.add_scalar("DAgger/scheduled_beta", scheduled_beta, iteration)
            self.writer.add_scalar("DAgger/teacher_control_fraction", teacher_fraction, iteration)
            self.writer.add_scalar("Routing/specialist_weight", specialist_weight, iteration)
            self.writer.add_scalar("Routing/mean_relief", mean_relief, iteration)
            self.writer.add_scalar("Replay/size", self.replay.total_size, iteration)
            self.writer.add_scalar("Perf/iteration_seconds", elapsed, iteration)
            print(
                f"[distill {iteration:05d}] bc={mean_bc:.5f} online={action_error:.5f} "
                f"probe={probe_error:.5f}/{self.online_error_ema:.5f} vel={mean_estimation:.5f} "
                f"beta={beta:.3f} teacher={teacher_fraction:.3f} specialist={specialist_weight:.3f} "
                f"replay={self.replay.total_size} time={elapsed:.2f}s"
            )
            specialist_coverage_ready = all(
                count >= min_specialist_samples_for_best for count in self.specialist_active_counts[1:]
            )
            self.writer.add_scalar("Routing/specialist_coverage_ready", float(specialist_coverage_ready), iteration)
            if specialist_coverage_ready and probe_error + 1.0e-4 < self.best_probe_error:
                self.best_probe_error = probe_error
                self.save(os.path.join(self.log_dir, "model_best.pt"))
            if iteration % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))
                self.save_replay(os.path.join(self.log_dir, "teacher_replay.pt"))

        self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))
        if self.best_probe_error == float("inf"):
            print(
                "[WARNING] Specialist coverage was insufficient to rank checkpoints; "
                "saving the final student as model_best.pt."
            )
            self.best_probe_error = probe_error
            self.save(os.path.join(self.log_dir, "model_best.pt"))
        self.save_replay(os.path.join(self.log_dir, "teacher_replay.pt"))

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.student_actor_critic.state_dict(),
                "encoder_state_dict": self.student_encoder.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter": self.current_iteration,
                "infos": {
                    "training_type": "multi_teacher_distillation",
                    "teacher_names": TEACHER_NAMES,
                    "obs_dim": self.obs_dim,
                    "history_dim": self.history_dim,
                    "command_dim": self.command_dim,
                    "action_dim": self.action_dim,
                    "best_probe_error": self.best_probe_error,
                    "online_error_ema": self.online_error_ema,
                    "specialist_relief_start": self.specialist_relief_start,
                    "specialist_relief_full": self.specialist_relief_full,
                    "specialist_active_counts": self.specialist_active_counts,
                },
            },
            path,
        )

    def save_replay(self, path: str) -> None:
        self.replay.save(path)
