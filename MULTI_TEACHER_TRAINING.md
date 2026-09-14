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
  --beta_end 0.25 \
  --distill_learning_rate 1e-4 \
  --student_probe_fraction 0.10 \
  --max_probe_error 0.08 \
  --min_specialist_samples_for_best 1000 \
  --specialist_relief_start 0.03 \
  --specialist_relief_full 0.10 \
  --headless
```

新版蒸馏不再按“整条赛道类型”直接切换教师。高度扫描起伏小于 3 cm 时，
四类赛道都使用 Continuous 教师，保住统一的平地步态；3～10 cm 之间平滑混合，
超过 10 cm 后才完全使用楼梯、障碍或沟壑教师。路由所用的高度信息不进入学生
额外观测，学生仍然只依赖原有 policy 输入。

`student_probe_fraction=0.10` 会保留 10% 环境始终由学生控制。若这些环境上的
动作误差超过 `max_probe_error`，Runner 会自动提高教师控制比例，阻止学生继续
偏离。`beta_end=0.25` 是教师控制下限，不建议在纯蒸馏阶段降到零。只有三个
专项教师各自在明显地形上累计至少 1000 个样本后，才开始更新 `model_best.pt`，
避免起始平地的零误差过早锁定最佳模型。

不要从旧的 `model_10000.pt` 恢复新版蒸馏；旧模型已经发生平地步态遗忘，应从
Continuous 教师重新初始化一个新运行。

输出目录：

```text
logs/rsl_rl/sf_tron_2a_multi_teacher/<RUN>/
├── model_best.pt
├── model_<ITER>.pt
└── teacher_replay.pt
```

`model_best.pt` 是学生探针误差最低的策略，优先用于评估和后续 PPO；
`model_<ITER>.pt` 是定期快照，`teacher_replay.pt` 是四种地形等量保存的防遗忘回放。

## 5. Camp 蒸馏策略专项 PPO 微调

这里使用独立的 `Camp-Distill-Finetune` 任务。它采用低学习率、低熵和较小 PPO
裁剪范围，主要学习速度/转向指令、中心线纠偏以及不同地形之间的衔接；教师回放
用于限制策略漂移，避免重新塑造已有的楼梯、沟壑和基础步态。

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Limx-SF-TRON2A-Camp-Distill-Finetune-v0 \
  --num_envs 4096 \
  --resume True \
  --checkpoint_path <MULTI_TEACHER_RUN>/model_best.pt \
  --imitation_replay <TEACHER_REPLAY> \
  --imitation_coef 0.1 \
  --imitation_batch_size 2048 \
  --headless
```

专项微调默认额外训练 3000 次迭代、每 100 次保存一次。建议先用 `0.1`；
如果指令跟踪学习过慢可降到 `0.05`，如果沟壑或楼梯能力明显遗忘可升到
`0.15～0.2`。评估时使用 `Isaac-Limx-SF-TRON2A-Camp-Distill-Finetune-Play-v0`。

## 任务列表

```text
Isaac-Limx-SF-TRON2A-Teacher-Continuous-v0
Isaac-Limx-SF-TRON2A-Teacher-Stairs-v0
Isaac-Limx-SF-TRON2A-Teacher-Obstacle-v0
Isaac-Limx-SF-TRON2A-Teacher-Gap-v0
Isaac-Limx-SF-TRON2A-MultiTeacher-v0
```

四个教师任务都有对应的 `-Play-v0` 任务。
