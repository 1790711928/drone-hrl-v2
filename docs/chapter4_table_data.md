# 第四章三线表基础参数整理

提取日期：2026-10-27  
范围：仅读取环境定义、配置、README、SAC/PPO 训练脚本和模型构建接口。  
本报告不检查图片、热力图、胜率或消融结果，不重新训练、不运行完整评估，也不修改现有实验文件。

## 参数提取摘要

- 表4-1 的环境约束和场景代表性初始化可以从代码直接提取；随机训练的实际运行 seed 和实际参数仍需单独确认。
- 表4-2 的 SAC 显式参数可以整理，但 learning rate、gamma、tau、replay buffer、entropy coefficient 和网络结构没有在项目脚本中显式指定，不能把 Stable-Baselines3 默认值写成已确认实验参数。
- 表4-3 的 PPO 显式参数可以整理；GAE lambda、clip range、entropy/value 系数及网络结构没有显式指定。
- README 的观测维度为 21，当前 `src/training/sac_env.py` 的 `OBS_KEYS` 为 25；正式写作应优先采用当前代码，并在论文中确认版本。

状态定义：`CONFIRMED`=代码/配置明确值；`DEFAULT`=训练脚本默认值；`UNVERIFIED`=存在配置差异或无法确认实际运行版本；`MISSING`=当前代码未提供该参数。

## 表4-1：无人机与仿真环境基础参数

### 代表性静态场景初始化

以下是 `src/env/scenarios.py` 中的代码定义，只能作为代表性初始化；训练脚本使用 `randomize_reset=True`，因此不能写成所有回合固定值。

| 参数 | 我方无人机 | 对方无人机 | 状态/来源 |
|---|---:|---:|---|
| 后向场景位置 (X,Y,Z) | (0, 0, 12) | (-6, 0, 12) | CONFIRMED；`src/env/scenarios.py` |
| 后向场景 Pitch | 0 | 0 | CONFIRMED；同上 |
| 后向场景 Yaw | 0.2 | 0 | CONFIRMED；同上 |
| 后向场景 Speed | 10 | 11.5 | CONFIRMED；同上 |
| 后向场景 Distance | 6.000 | — | 由静态位置计算，代表性值 |
| 后向场景 Azimuth | — | 180° | 由静态位置计算，代表性值 |
| 侧向场景位置 (X,Y,Z) | (0, 0, 10) | (-8, 8, 10) | CONFIRMED；`src/env/scenarios.py` |
| 侧向场景 Pitch | 0 | 0 | CONFIRMED |
| 侧向场景 Yaw | 0 | -0.3 | CONFIRMED |
| 侧向场景 Speed | 9.5 | 11 | CONFIRMED |
| 侧向场景 Distance | 11.314 | — | 代表性静态值 |
| 侧向场景 Azimuth | — | 135° | 代表性静态值 |
| 边界场景位置 (X,Y,Z) | (46, 0, 8) | (30, -10, 8) | CONFIRMED；`src/env/scenarios.py` |
| 边界场景 Pitch | 0 | 0 | CONFIRMED |
| 边界场景 Yaw | 0.8 | 0.5 | CONFIRMED |
| 边界场景 Speed | 9 | 10.5 | CONFIRMED |
| 边界场景 Distance | 18.868 | — | 代表性静态值 |
| 边界场景 Azimuth | — | -147.99° | 代表性静态值 |
| 垂向场景位置 (X,Y,Z) | (0, 0, 18) | (-3, 1, 3) | CONFIRMED；`src/env/scenarios.py` |
| 垂向场景 Pitch | 0.16 | 0.18 | CONFIRMED |
| 垂向场景 Yaw | 0 | 0.05 | CONFIRMED |
| 垂向场景 Speed | 10 | 11 | CONFIRMED |
| 垂向场景 Distance | 15.330 | — | 代表性静态值 |
| 垂向场景 Azimuth | — | 161.57° | 代表性静态值 |

随机初始化：逃逸方和追击方 x/y 分别在静态值上加减 3.0，z 加减 1.5，速度加减 1.0 后裁剪到各自速度范围，yaw 加减 0.15，pitch 加减 0.08。该范围来自 `src/env/pursuit_escape_env.py`，状态为 `CONFIRMED`；某次实际实验是否采用该模式及其 seed 为 `UNVERIFIED`。

### 环境基础约束

| 参数 | 值 | 状态/来源 |
|---|---:|---|
| 仿真空间维度 | 3D | CONFIRMED；`configs/env.yaml` |
| x 范围 | [-50, 50] | CONFIRMED；`configs/env.yaml` |
| y 范围 | [-50, 50] | CONFIRMED；`configs/env.yaml` |
| z 范围 | [0, 50] | CONFIRMED；`configs/env.yaml` |
| 时间步长 | 0.1 s | CONFIRMED；`configs/env.yaml` |
| episode 最大步数 | 600 | CONFIRMED；`configs/env.yaml` / `termination.py` |
| 捕获距离 | distance < 2.5 | CONFIRMED |
| 捕获持续条件 | 连续 4 步 | CONFIRMED；`k_capture` |
| 安全距离 | distance > 26.0 | CONFIRMED |
| 逃逸持续条件 | 连续 8 步，且 closing_speed≤0、视线条件满足 | CONFIRMED；`termination.py` |
| 越界判定 | 逃逸方位置超出任一 x/y/z 边界 | CONFIRMED；`in_bounds()` |
| 追击速度比例 | 1.2 | DEFAULT；`configs/env.yaml`/`EnvConfig` |
| 逃逸方速度范围 | [2.0,20.0] | CONFIRMED |
| 追击方速度范围 | [2.0,25.0] | CONFIRMED |
| 最大 yaw rate | 1.0 | CONFIRMED |
| 最大 pitch rate | 0.7 | CONFIRMED |

## 表4-2：低层 SAC 训练参数

四个专家均由 `src/training/train_lowlevel_all.py` 调用 `src/training/train_lowlevel.py`，脚本未定义专家专属超参数。下表中的“默认”不是实际运行值的证明。

| 参数 | 四个 SAC 专家统一值 | 状态/来源 |
|---|---:|---|
| 算法 | SAC | CONFIRMED；`train_lowlevel.py` |
| 专家名称 | pi1_rear / pi2_flank / pi3_boundary / pi4_vertical | CONFIRMED |
| learning rate | 未显式设置 | MISSING；由 SB3 默认决定，依赖版本未固定 |
| gamma | 未显式设置 | MISSING |
| tau | 未显式设置 | MISSING |
| batch size | 256 | CONFIRMED；脚本显式参数 |
| replay buffer size | 未显式设置 | MISSING |
| warm-up steps (`learning_starts`) | 1,000 | CONFIRMED |
| entropy coefficient | 未显式设置 | MISSING |
| 自动温度调节 | 未显式设置，不能确认 | UNVERIFIED |
| actor 网络结构 | `MlpPolicy`，具体层结构未在项目代码声明 | UNVERIFIED |
| critic 网络结构 | `MlpPolicy`，具体层结构未在项目代码声明 | UNVERIFIED |
| 每回合最大步数 | 环境最大 600 步 | CONFIRMED；环境定义 |
| 默认训练步数 | 50,000 | DEFAULT；命令行默认 |
| 默认训练 episode 数 | 未固定，由 episode 长度和终止情况决定 | MISSING |
| 随机种子 | 0 | DEFAULT；命令行默认 |
| checkpoint 间隔 | 5,000 步 | DEFAULT；命令行默认 |
| reward 记录 | `episode_rewards.csv`、`monitor.csv`，另配置 TensorBoard 目录 | CONFIRMED；`training_logging.py` |
| 场景混合 | 目标场景 0.8，其余三个各 0.0667（默认 mix_ratio=0.2） | DEFAULT；训练脚本 |

四个专家是否使用相同超参数：从训练脚本看是相同的，`CONFIRMED`。单个专家专属超参数：未找到，`MISSING`。脚本默认值与实际运行值：可能不同，`UNVERIFIED`；README 示例使用 40,000 步，而当前训练脚本默认 50,000 步。

## 表4-3：高层 PPO 训练参数

| 参数 | PPO 选择器值 | 状态/来源 |
|---|---:|---|
| 算法 | PPO | CONFIRMED；`src/training/train_highlevel.py` |
| learning rate | 3e-4 | CONFIRMED |
| gamma | 0.99 | CONFIRMED |
| GAE lambda | 未显式设置 | MISSING |
| clip range | 未显式设置 | MISSING |
| entropy coefficient | 未显式设置 | MISSING |
| value loss coefficient | 未显式设置 | MISSING |
| batch size | 256 | CONFIRMED |
| rollout length (`n_steps`) | 1,024 | CONFIRMED |
| minibatch 数量 | 未单独设置；由 rollout length/batch size 关系决定 | UNVERIFIED |
| update 次数 | 未单独设置；由 PPO 学习流程决定 | MISSING |
| actor 网络结构 | `MlpPolicy`，项目未显式声明层结构 | UNVERIFIED |
| critic 网络结构 | `MlpPolicy`，项目未显式声明层结构 | UNVERIFIED |
| 默认训练步数 | 80,000 | DEFAULT |
| 默认训练 episode 数 | 未固定 | MISSING |
| 随机种子 | 0 | DEFAULT |
| checkpoint 间隔 | 5,000 步 | DEFAULT |
| reward 记录 | `episode_rewards.csv`、`monitor.csv`、TensorBoard 目录 | CONFIRMED |
| 高层 option duration | 8 个低层步 | DEFAULT |
| switch penalty | 0.02 | DEFAULT |

PPO 与 SAC 对比：训练步数默认不同（PPO 80,000；SAC 50,000）；两者均未固定 episode 数；输入观测当前都来自 25 维共享 schema，但 SAC 输出 3 维连续动作，PPO 输出 4 类离散技能动作；两者均使用 `MlpPolicy`，项目代码没有声明具体网络层结构，因此不能确认两者实际网络是否相同。

## 正文中的观测与动作维度说明

当前代码定义的总观测为 25 维，主要包括相对位置/距离、闭合速度与双方速度、航向角正余弦、俯仰角、视线关系、边界裕度、归一化时间/位置以及机体坐标系威胁方向。低层 SAC 输出 3 维连续动作（加速度、偏航角速度、俯仰角速度）；高层 PPO 输出 4 个离散动作，分别选择 pi1 后向、pi2 侧向、pi3 边界和 pi4 垂向技能。

观测处理：部分变量使用归一化或 sin/cos 编码，位置/边界相关量按环境范围缩放；当前代码没有统一的全向量标准化或整体裁剪声明。README 的“21维”与 `src/training/sac_env.py::OBS_KEYS` 的“25维”不一致。正式论文应优先依据当前代码中的 25 维定义，并在写作前确认论文对应的代码版本和最终运行版本。

## 主要来源文件

- `configs/env.yaml`
- `configs/train_lowlevel.yaml`
- `configs/train_highlevel.yaml`
- `src/env/pursuit_escape_env.py`
- `src/env/scenarios.py`
- `src/env/termination.py`
- `src/training/sac_env.py`
- `src/training/train_lowlevel.py`
- `src/training/train_lowlevel_all.py`
- `src/training/train_highlevel.py`
- `src/training/training_logging.py`
- `README.md`
- `paper_code_mapping.md`

