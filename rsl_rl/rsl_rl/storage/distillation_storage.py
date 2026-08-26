"""Balanced CPU replay storage for multi-teacher action distillation."""

from __future__ import annotations

import torch


class BalancedDistillationReplay:
    """Fixed-capacity ring buffers with equal capacity for every teacher."""

    def __init__(
        self,
        num_skills: int,
        capacity_per_skill: int,
        obs_dim: int,
        history_dim: int,
        command_dim: int,
        action_dim: int,
    ):
        if num_skills <= 0 or capacity_per_skill <= 0:
            raise ValueError("num_skills and capacity_per_skill must be positive.")
        self.num_skills = num_skills
        self.capacity_per_skill = capacity_per_skill
        self.obs = torch.empty(num_skills, capacity_per_skill, obs_dim, dtype=torch.float32)
        self.history = torch.empty(num_skills, capacity_per_skill, history_dim, dtype=torch.float32)
        self.commands = torch.empty(num_skills, capacity_per_skill, command_dim, dtype=torch.float32)
        self.actions = torch.empty(num_skills, capacity_per_skill, action_dim, dtype=torch.float32)
        self.velocity_targets = torch.empty(num_skills, capacity_per_skill, 3, dtype=torch.float32)
        self.sizes = torch.zeros(num_skills, dtype=torch.long)
        self.write_indices = torch.zeros(num_skills, dtype=torch.long)

    def add(self, obs, history, commands, actions, velocity_targets, skill_ids) -> None:
        tensors = (obs, history, commands, actions, velocity_targets, skill_ids)
        obs, history, commands, actions, velocity_targets, skill_ids = (
            tensor.detach().to("cpu") for tensor in tensors
        )
        skill_ids = skill_ids.to(dtype=torch.long).flatten()
        for skill_id in range(self.num_skills):
            mask = skill_ids == skill_id
            count = int(mask.sum().item())
            if count == 0:
                continue
            selected = (obs[mask], history[mask], commands[mask], actions[mask], velocity_targets[mask])
            # A single large rollout may exceed one skill's full capacity.  A
            # random subset avoids keeping only one contiguous time interval.
            if count > self.capacity_per_skill:
                indices = torch.randperm(count)[: self.capacity_per_skill]
                selected = tuple(tensor[indices] for tensor in selected)
                count = self.capacity_per_skill

            start = int(self.write_indices[skill_id].item())
            first = min(count, self.capacity_per_skill - start)
            second = count - first
            destinations = (
                self.obs[skill_id],
                self.history[skill_id],
                self.commands[skill_id],
                self.actions[skill_id],
                self.velocity_targets[skill_id],
            )
            for destination, source in zip(destinations, selected):
                destination[start : start + first].copy_(source[:first])
                if second:
                    destination[:second].copy_(source[first:])
            self.write_indices[skill_id] = (start + count) % self.capacity_per_skill
            self.sizes[skill_id] = min(self.capacity_per_skill, int(self.sizes[skill_id]) + count)

    @property
    def total_size(self) -> int:
        return int(self.sizes.sum().item())

    def balanced_batch(self, batch_size: int, device: str):
        available = [skill for skill in range(self.num_skills) if self.sizes[skill] > 0]
        if not available:
            raise RuntimeError("Cannot sample an empty distillation replay.")
        per_skill = max(1, batch_size // len(available))
        batches = [[] for _ in range(5)]
        skill_batch = []
        for skill in available:
            size = int(self.sizes[skill].item())
            indices = torch.randint(size, (per_skill,))
            sources = (self.obs, self.history, self.commands, self.actions, self.velocity_targets)
            for batch, source in zip(batches, sources):
                batch.append(source[skill, indices])
            skill_batch.append(torch.full((per_skill,), skill, dtype=torch.long))
        result = tuple(torch.cat(batch, dim=0).to(device) for batch in batches)
        return (*result, torch.cat(skill_batch, dim=0).to(device))

    def state_dict(self) -> dict:
        chunks = {"obs": [], "history": [], "commands": [], "actions": [], "velocity_targets": []}
        skill_ids = []
        for skill in range(self.num_skills):
            size = int(self.sizes[skill].item())
            if size == 0:
                continue
            for name in chunks:
                chunks[name].append(getattr(self, name)[skill, :size].clone())
            skill_ids.append(torch.full((size,), skill, dtype=torch.long))
        return {
            **{name: torch.cat(values, dim=0) for name, values in chunks.items()},
            "skill_ids": torch.cat(skill_ids, dim=0),
            "num_skills": self.num_skills,
        }

    def save(self, path: str) -> None:
        if self.total_size == 0:
            raise RuntimeError("Refusing to save an empty distillation replay.")
        torch.save(self.state_dict(), path)
