# tron2_rl_lab

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 LimX **TRON2A** 双足机器人强化学习训练栈，使用 PPO 训练 locomotion 策略。本仓库专注于基础形态的平地（Flat）训练，支持 SF（sole-foot）与 WF（wheel-foot）两种机器人变体。

## 仓库结构

```
.
├── exts/bipedal_locomotion/   # Isaac Lab extension：env/asset/MDP/robot cfg
├── rsl_rl/                    # 项目内 vendored 的 rsl_rl fork（PPO + on-policy runner）
├── scripts/rsl_rl/            # 训练 / play 入口（train.py / play.py / cli_args.py）
└── logs/                      # 训练日志与模型权重
```

## 环境要求

- **Isaac Sim 5.1** + **Isaac Lab 2.3.1**
- Python 3.10
- GPU（推荐 ≥ 12 GB 显存，4096 envs 训练）

## 安装

```bash
# 1. clone 仓库
git clone <repo-url> tron1_rl_lab
cd tron1_rl_lab

# 2. editable install extension 与 vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

## IsaacSim 训练

任务 ID 均在 [exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py](exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/robots/__init__.py) 中注册。

### 1. 训练模型

```bash
# === Solefoot (SF) ===
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-v0 --num_envs 4096 --headless

# === Wheelfoot (WF) ===
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-TRON2A-Blind-Flat-v0 --num_envs 4096 --headless
```

*常用选项：*
- `--checkpoint_path <path>`: 从某 .pt 恢复。
- `--video`: 开启录像。
- `--max_iterations N`: 覆盖最大迭代数。

日志路径：`logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/`

### 2. 运行推理 (Play)

用 `-Play-v0` 后缀的任务 ID。Play cfg 使用更少 env、关闭域随机化、简化地形。

```bash
# Solefoot (SF)
python scripts/rsl_rl/play.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-Play-v0 --num_envs 32

# Wheelfoot (WF)
python scripts/rsl_rl/play.py --task Isaac-Limx-WF-TRON2A-Blind-Flat-Play-v0 --num_envs 32
```

*注意：* 默认加载最新 checkpoint，指定路径可用 `--checkpoint_path`。

### 3. Resume 续训

必须显式带 `--resume True` 才会加载 checkpoint。

```bash
# 方式 A：直接给 .pt 路径（推荐）
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-v0 --resume True --checkpoint_path <path_to_model>

# 方式 B：按 run 名查找
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-TRON2A-Blind-Flat-v0 --resume True --load_run <run_name>
```

## 机器人形态

| 形态 | 末端 | task id 前缀 |
|---|---|---|
| SF_TRON2A | sole foot (ankle pitch) | `Isaac-Limx-SF-TRON2A-...` |
| WF_TRON2A | wheel | `Isaac-Limx-WF-TRON2A-...` |

## 架构概览

项目分为三个主要部分：

1. **`exts/bipedal_locomotion/`** — Isaac Lab extension。包含 env / asset / MDP / robot cfg。
2. **`rsl_rl/`** — vendored fork。`scripts/rsl_rl/train.py` 会优先加载此路径下的算法库。
3. **`scripts/rsl_rl/`** — 训练与推理的入口脚本。

### 任务 Wiring 流程

以 `Isaac-Limx-SF-TRON2A-Blind-Flat-v0` 为例：

1. **Gym 注册**：在 `tasks/locomotion/robots/__init__.py` 中将环境配置与 PPO 配置绑定到任务 ID。
2. **环境配置 (Env cfg)**：在 `tasks/locomotion/robots/limx_solefoot_tron2a_env_cfg.py` 中定义资产加载与 MDP 规则。
3. **资产配置 (Asset cfg)**：在 `assets/config/solefoot_tron2a_cfg.py` 中指定内置的 USD 路径及执行器（Actuator）参数。

## MuJoCo 仿真与实机部署

- [MuJoCo 仿真仓库](https://github.com/limx-tron2/tron2_mujoco_sim)
- [Python实机部署代码](https://github.com/limx-tron2/tron2_rl_deploy_python)

### MuJoCo 部署效果（SF / WF）

<p align="center">
  <img src="doc/mujoco_sf.gif" alt="MuJoCo SF" width="48%" />
  <img src="doc/mujoco_wf.gif" alt="MuJoCo WF" width="48%" />
</p>

- 无法直接预览时可下载查看：
  - [mujoco_sf.gif](doc/mujoco_sf.gif)
  - [mujoco_wf.gif](doc/mujoco_wf.gif)

## Gazebo 仿真与实机部署

- [Gazebo 仿真仓库](https://github.com/limx-tron2/tron2_gazebo_ros)
- [ROS实机部署代码](https://github.com/limx-tron2/tron2_rl_deploy_ros)

### Gazebo 部署效果（SF / WF）

<p align="center">
  <img src="doc/gazebo_sf.gif" alt="Gazebo SF" width="48%" />
  <img src="doc/gazebo_wf.gif" alt="Gazebo WF" width="48%" />
</p>

- 无法直接预览时可下载查看：
  - [gazebo_sf.gif](doc/gazebo_sf.gif)
  - [gazebo_wf.gif](doc/gazebo_wf.gif)

## 实机部署效果（办公室场景）

<p align="center">
  <img src="doc/real_wf.GIF" alt="TRON2A 实机部署效果 1" width="48%" />
  <img src="doc/real_sf.GIF" alt="TRON2A 实机部署效果 2" width="48%" />
</p>

## 实机运行注意事项（强烈建议）

启动与落地流程建议固定为以下顺序，避免切策略瞬间冲击：

1. **将机器人吊起**，确保双脚不承重，先检查关节状态、急停和通信是否正常。
2. **先进入 IK 模式**，确认逆解控制稳定、姿态与期望一致。
3. **再缓慢将机器人放到地面**，观察接触是否平稳、是否出现异常抖动。
4. **最后切换到 walk 策略**，先低速小步验证，再逐步提升速度与动作幅度。

如出现异常（突发抖动、姿态发散、落地后冲击过大），请立即急停并回到吊起状态重新检查。

## License

[Apache 2.0](LICENCE)。

## Reference

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)
