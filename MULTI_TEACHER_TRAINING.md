# SF_TRON2A 多教师训练

这套流水线使用四个观测和动作接口完全一致的教师：连续地面、楼梯、障碍和沟壑。训练标签只供蒸馏 Runner 选择教师，不进入学生策略观测。最终产物仍是普通的 `ActorCritic + MLP_Encoder` checkpoint，可直接由 `play.py` 或 Camp PPO 加载。

## 1. 训练连续地面教师

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Teacher-Continuous-v0 \
  --num_envs 4096 \
  --headless
```

## 2. 从连续教师初始化专项教师

将 `<CONTINUOUS_CHECKPOINT>` 替换为上一步生成的 checkpoint。

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Teacher-Stairs-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path <CONTINUOUS_CHECKPOINT> \
  --headless

python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Teacher-Obstacle-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path <CONTINUOUS_CHECKPOINT> \
  --headless

python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Teacher-Gap-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path <CONTINUOUS_CHECKPOINT> \
  --headless
```

教师评估任务分别在任务名中加入 `-Play`，例如：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Limx-SF-TRON2A-Teacher-Gap-Play-v0 \
  --checkpoint_path <GAP_CHECKPOINT>
```

## 3. 检查教师兼容性

```bash
python scripts/rsl_rl/check_teacher_compatibility.py \
  --teacher_continuous <CONTINUOUS_CHECKPOINT> \
  --teacher_stairs <STAIRS_CHECKPOINT> \
  --teacher_obstacle <OBSTACLE_CHECKPOINT> \
  --teacher_gap <GAP_CHECKPOINT>
```

正常输出应包含：

```text
actor_input=235, history_input=420, action_dim=10
```

旧的 Blind-Flat checkpoint 没有高度扫描，不能作为这套流水线的连续教师。

## 4. 在线蒸馏和 DAgger

RTX 4070 8GB 建议先使用 2048 个环境；如果显存不足再降至 1024。

```bash
python scripts/rsl_rl/distill.py \
  --task Isaac-Limx-SF-TRON2A-MultiTeacher-v0 \
  --num_envs 2048 \
  --distill_iterations 5000 \
  --teacher_continuous <CONTINUOUS_CHECKPOINT> \
  --teacher_stairs <STAIRS_CHECKPOINT> \
  --teacher_obstacle <OBSTACLE_CHECKPOINT> \
  --teacher_gap <GAP_CHECKPOINT> \
  --beta_start 1.0 \
  --beta_end 0.0 \
  --headless
```

输出目录：

```text
logs/rsl_rl/sf_tron_2a_multi_teacher/<RUN>/
├── model_<ITER>.pt
└── teacher_replay.pt
```

`model_<ITER>.pt` 是学生策略，`teacher_replay.pt` 是四种能力等量保存的防遗忘回放。

## 5. Camp PPO 微调

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Camp-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path <DISTILLED_STUDENT_CHECKPOINT> \
  --imitation_replay <TEACHER_REPLAY> \
  --imitation_coef 0.1 \
  --imitation_batch_size 2048 \
  --headless
```

建议先用 `0.1`。如果 Camp 奖励提高缓慢可降到 `0.05`；如果沟壑或楼梯能力明显遗忘，可升到 `0.15～0.2`。

## 任务列表

```text
Isaac-Limx-SF-TRON2A-Teacher-Continuous-v0
Isaac-Limx-SF-TRON2A-Teacher-Stairs-v0
Isaac-Limx-SF-TRON2A-Teacher-Obstacle-v0
Isaac-Limx-SF-TRON2A-Teacher-Gap-v0
Isaac-Limx-SF-TRON2A-MultiTeacher-v0
```

四个教师任务都有对应的 `-Play-v0` 任务。
