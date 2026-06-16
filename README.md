# drone-hrl-v2

无人机 3D 追逃（分层强化学习）入门版工程（**Windows 优先**）。

> 你说你的环境是 Windows + Python 程序，所以这份 README 以 Windows 操作为主，Linux/Mac 只放简版。

---


## 0) 你现在跑的 `main` 在做什么？

`python -m src.main` 目前是在跑**演示回合（demo）**，不是强化学习训练：
- 逃跑方动作当前是固定动作（示例动作）。
- 追击方是规则引导（朝向逃跑方）。

如果你要看轨迹图，用：
```powershell
python -m src.main --scenario rear_close_threat --steps 20 --save-plot --plot-path outputs/trajectory.png
```

如果提示 `[plot] skipped: matplotlib is required...`，执行：`python -m pip install -r requirements.txt`。

## 1) 目录结构（搭积木）

```text
drone-hrl-v2/
├─ src/
│  ├─ main.py                      # 新手入口：先跑这个
│  ├─ visualization/
│  │  └─ trajectory.py             # 保存3D轨迹图
│  ├─ env/
│  │  ├─ dynamics.py               # 3D动力学 + 规则追击
│  │  ├─ termination.py            # 终止条件（k步逃脱/k步捕获）
│  │  ├─ scenarios.py              # 4个场景初始条件
│  │  └─ pursuit_escape_env.py     # 环境主循环（reset/step）
│  ├─ training/
│  │  ├─ train_lowlevel.py         # 低层训练入口（smoke）
│  │  └─ train_highlevel.py        # 高层训练入口（TODO）
│  └─ evaluation/
│     └─ metrics.py
├─ configs/
├─ scripts/
│  ├─ setup_windows.bat            # Windows一键初始化（推荐）
│  ├─ setup_windows.ps1            # PowerShell初始化（可选）
│  ├─ run_demo.bat                 # Windows运行demo
│  └─ run_tests.bat                # Windows运行测试
├─ requirements.txt
└─ tests/
```

---

## 2) Windows 快速开始（推荐）

### 2.1 打开 PowerShell 并进入项目
```powershell
Set-Location "E:\python程序\drone_hrl_repo\drone-hrl-v2"
```

> 说明：`cd /d ...` 是 **cmd** 用法，不是 PowerShell 用法。

> 如果你项目目录不一样，改成你自己的路径即可。

### 2.2 执行初始化脚本（优先用 bat，最稳）
```powershell
& .\scripts\setup_windows.bat
```

如果你想用 PowerShell 版本：
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

如果你没有 `py` 启动器，可以这样：
```powershell
.\scripts\setup_windows.ps1 -PythonCmd "python"
```

如果 PowerShell 仍报“字符串缺少终止符”，请直接改用 `setup_windows.bat`（不受该问题影响）。

### 2.3 跑 demo（两种方式）
方式 A（推荐）：
```powershell
& .\scripts\run_demo.bat
```

方式 B（手动）：
```powershell
& .\.venv\Scripts\Activate.ps1
python -m src.main --scenario rear_close_threat --steps 20
```

### 2.4 跑测试
```powershell
& .\scripts\run_tests.bat
```


## 2.5 如果脚本仍失败：手动安装（PowerShell）

```powershell
Set-Location "E:\python程序\drone_hrl_repo\drone-hrl-v2"
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m src.main --scenario rear_close_threat --steps 20
python -m pytest -q
```

如果 `Activate.ps1` 被策略阻止，可先执行：
```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

---

## 3) 你会看到什么输出？

每一步会输出：
- `distance`：追逃距离
- `closing_speed`：闭合速度（<=0 通常表示正在拉开）
- `outcome`：running / escaped / captured / out_of_bounds / timeout
- `escape_streak`：连续满足逃脱条件步数（防瞬时逃脱胜率虚高）

低层 SAC / 高层 PPO 共用 observation（21 维）字段如下（threat-geometry 增强版）：
- 相对位移（归一化）：`dx, dy, dz`
- 距离与速度（归一化）：`distance, closing_speed, evader_speed, pursuer_speed`
- 航向角编码：`evader_yaw_sin, evader_yaw_cos, pursuer_yaw_sin, pursuer_yaw_cos`
- 俯仰角（归一化到 [-1, 1]）：`evader_pitch, pursuer_pitch`
- 几何关系：`los_cos`（逃跑方航向与视线方向夹角余弦）
- 边界风险（归一化）：`boundary_margin_x, boundary_margin_y, boundary_margin_z, min_boundary_margin`
- 边界方向（归一化到 [-1, 1]）：`evader_x_norm, evader_y_norm, evader_z_norm`
- 本机体坐标系威胁方向：`threat_forward, threat_right, threat_up`（单位方向分量，范围 [-1, 1]）
- 进度：`normalized_step`（`step_count / max_steps`）

---

## 4) 终止条件（重点）

> 当前默认训练空间：x/y 为 `[-50, 50]`，z 为 `[0, 50]`。


### 逃脱判定（必须连续 k 步）
需同时满足并持续 `k_escape` 步：
- `distance > d_safe`
- `closing_speed <= 0`
- `los_escape_ok == True`

### 捕获判定（也必须连续 k 步）
- `distance < d_capture` 持续 `k_capture` 步

---

## 5) 下一步（新手学习路线）

1. 先通过 `src/main.py` 看懂环境 step 输出。
2. 在 `src/training/train_lowlevel.py` 接入 SAC（每个场景一个策略）。
3. 先做 4x4 评估确认每个低层在主场景最优，再冻结低层训练上层 PPO 切换器。
4. 增加 4x4 评估脚本和鲁棒性扫描脚本。

---

## Linux/Mac 简版

```bash
cd /workspace/drone-hrl-v2
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m src.main --scenario rear_close_threat --steps 20
python -m pytest -q
```


## 6) 接入 SAC 训练（已可用）

> 默认已关闭 tensorboard 依赖（`--log-dir` 为空）。
> 如果你传了 `--log-dir` 但没安装 tensorboard，训练脚本会自动关闭该日志并继续训练。
> 若要启用 tensorboard，可执行：`python -m pip install tensorboard`，并传入 `--log-dir outputs/logs/...`。
> 当前默认难度已上调（更严格逃脱判定、更高追击速度比）以提升策略区分度。

### 6.1 单策略低层训练（推荐主次场景混合）
```powershell
python -m src.training.train_lowlevel --scenario rear_close_threat --timesteps 40000 --mix-ratio 0.2 --model-out outputs/checkpoints/sac_low_1_rear_close_threat.zip
```

可选场景：
- `rear_close_threat`（后方近距离威胁）
- `flank_threat`（侧翼威胁）
- `boundary_constrained`（边界受限）
- `vertical_z_threat`（垂直 z 轴威胁）
  - 设计意图：追击者初始具有明显垂直相对威胁（不仅是后向压迫），用于训练垂直机动策略；
  - reward 在该场景增加轻量垂直分离激励，并对接近 z 边界做额外惩罚，避免“无脑爬升/俯冲”越界。

全场景共享 reward 安全约束：
- 使用 shared soft boundary safety penalty（基于归一化边界余量），在接近边界前就开始惩罚，并在危险区间快速增大；
- 各场景仍通过 `w_boundary_risk` 控制惩罚强度（`boundary_constrained` 仍最高），用于保持边界控制难度分层。

`--mix-ratio` 说明：例如 `0.2` 表示该策略训练时有 80% 采样主场景，20% 采样其他三个次场景（均分）。

### 6.2 四个场景分别训练
```powershell
python -m src.training.train_lowlevel_all --timesteps 40000 --mix-ratio 0.2
```

训练日志默认在：`outputs/logs/sac`。



### 6.3 Full-episode 单策略评估（4x4）
```powershell
python -m src.evaluation.eval_lowlevel_matrix --episodes 100
python -m src.evaluation.eval_lowlevel_diagnostics --episodes 30
```

说明：这是 **full-episode single-policy evaluation**。它用于看“单个低层策略完整跑一局”的表现，
但不等同于 HRL 最终表现；也不能单独否定非 boundary 底层策略（因为最终会由高层 selector 切换策略）。

### 6.4 Skill-level 低层专属能力评估
```powershell
python -m src.evaluation.eval_lowlevel_skills --episodes 30 --skill-horizon 80
```

该脚本是 **option-level / skill-level evaluation**（短时域局部技能评估），不是完整 episode 逃生评估。

- 支持 early skill termination：在 `skill_horizon` 内一旦技能完成局部目标就提前结束，不强迫继续跑到 episode 末尾。
- 对非 boundary 技能（pi1/pi2/pi4）若完成技能后出现 x/y 边界风险，会记为 `handoff_to_boundary`（提示高层应切到 pi3），不直接否定该技能。
- pi4 使用“controlled vertical maneuver”定义：强调垂直分离落在目标区间/维持稳定 + z 边界可控，而不是无限增加 vertical separation。

主要输出字段：
`skill_success_rate`（option-level） 、`skill_completed_rate`、`avg_completion_step`、`distance_gain`、`closing_speed_reduction`、`threat_right_abs_reduction`、`min_boundary_margin_improvement`、`return_to_safe_region_rate`、`vertical_target_band_rate`、`vertical_separation_maintenance_rate`、`controlled_z_margin_rate`、`out_of_bounds_rate`、`z_out_of_bounds_rate`、`handoff_to_boundary_rate`。

说明：pi3 的成功判定以 `return_to_safe_region` 为核心；`handoff_to_boundary` 表示高层应切换到 pi3，不等于该底层技能直接失败。

CSV 输出：`outputs/evaluation/lowlevel_skill_diagnostics.csv`

### 6.5 单策略失败机制诊断（行为层）
```powershell
python -m src.evaluation.eval_policy_behavior --scenario rear_close_threat --model outputs/checkpoints/sac_low_1_rear_close_threat.zip --episodes 30
```

该脚本用于解释失败机制（越界轴向、动作激进程度、pitch/yaw 率、z 分离等），不直接用于调 reward。

### 6.6 pi3 边界恢复诊断（option-level）
```powershell
python -m src.evaluation.eval_boundary_skill_diagnostics --episodes 30 --skill-horizon 80
```

该脚本专门诊断 `pi3` 在 `boundary_constrained` 下的边界恢复失败机制（越界轴向、danger->controllable->safe 恢复分层、动作是否过保守/过激进等）。

`pi3` 的目标是把无人机从边界危险区带回可控机动区并交还高层 selector，不要求单策略完成完整 episode 逃生。

注意：它用于 **boundary recovery option** 诊断，不替代 full-episode evaluation。

### 6.7 High-level mixed threat selector 评估
```powershell
python -m src.evaluation.inspect_highlevel_mixed_scenarios --scenario-set composite
python -m src.evaluation.eval_highlevel_selector --mode fixed --fixed-policy 0 --episodes 20 --scenario-set composite
python -m src.evaluation.eval_highlevel_selector --mode random --episodes 20 --scenario-set composite
python -m src.evaluation.inspect_highlevel_mixed_scenarios --scenario-set sequential
python -m src.evaluation.eval_highlevel_selector --mode random --episodes 20 --scenario-set sequential
python -m src.evaluation.eval_option_sequence_search --episodes 2 --scenario-set sequential --max-seq-len 2 --option-durations 4,6
```

项目评估分三层：
- A. option-level skill evaluation：`eval_lowlevel_skills.py`
- B. full-episode single-policy stress test：`eval_lowlevel_diagnostics.py`
- C. high-level mixed threat selector evaluation：`eval_highlevel_selector.py`

高层 PPO 目标不是识别 S1/S2/S3/S4 标签，而是在复合威胁中学习 option 选择与切换。

`--scenario-set` 支持：`basic`（四基础场景）、`mixed`（加权抽样）、`composite`（真实复合威胁初始态）、`sequential`（单个 episode 内按阶段注入 rear/flank/boundary/vertical 威胁）、`continuous_pursuit`（连续动态追逃，不做 phase teleport，不把 regime label 加入 observation）。

`sequential` 用于验证真正的 option 切换：基础环境的中间 escape 不会自动完成 phase。每个 rear/flank/boundary/vertical phase 必须连续满足专属 geometry 条件后，环境才会注入下一阶段威胁；只有所有 phase 完成后才算最终 escaped。phase 名称、成功 streak 与失败统计仅写入 `info`，不会加入 observation。

`continuous_pursuit` 用于最终连续追逃压力测试：evader 状态保持连续，regime 仅控制 pursuer 的施压方式（rear/flank/vertical/boundary），不再用“完成固定 phase”作为成功条件。当前 regime 由 state-driven threat manager 根据 observation geometry 实时选择：边界风险优先，其次按 rear/flank/vertical 威胁分数判断；固定 schedule 只在威胁分数都较低时作为 fallback。episode 到达 `--episode-lowlevel-steps` 后，只有未被捕获/未越界且最近窗口保持安全距离与非持续闭合，才判为 escaped；否则为 timeout。默认起点已经校准到更靠近场地中心且初始航向不再沿 +x 轴直冲边界；默认 boundary priority 采用 `enter=0.24`、`exit=0.32`，避免在安全余量约 0.28 时过早锁定 boundary，同时仍允许真正边界风险打断 regime hold。

连续追逃诊断与评估示例：
```powershell
python -m src.evaluation.inspect_continuous_pursuit --episodes 1 --episode-lowlevel-steps 120 --print-every 10
python -m src.evaluation.eval_highlevel_selector --mode fixed --fixed-policy 2 --episodes 2 --scenario-set continuous_pursuit --episode-lowlevel-steps 120
python -m src.evaluation.eval_highlevel_selector --mode continuous_heuristic --episodes 20 --scenario-set continuous_pursuit --episode-lowlevel-steps 300
python -m src.evaluation.eval_highlevel_selector --mode regime_oracle --episodes 20 --scenario-set continuous_pursuit --episode-lowlevel-steps 300
python -m src.evaluation.eval_highlevel_selector --mode highlevel --episodes 2 --scenario-set continuous_pursuit --high-model outputs/checkpoints/ppo_highlevel_switch.zip --episode-lowlevel-steps 120
# 可用 --min-regime-hold-steps / --boundary-priority-enter / --boundary-priority-exit / --regime-schedule / --pursuer-speed-ratio 调整连续追逃诊断强度；默认 pursuer-speed-ratio=1.20，boundary enter/exit=0.24/0.32。
```

在训练高层 PPO 前，建议优先运行 one-shot phase × option 区分度诊断：
```powershell
python -m src.evaluation.eval_phase_option_discriminability --episodes 10 --phase-types all --option-duration 8 --eval-mode one_shot
```

`eval_phase_option_discriminability.py` 会使用现有 phase 注入函数和 phase-specific 成功条件，输出 rear/flank/boundary/vertical/rear_vertical × pi1/pi2/pi3/pi4 矩阵。诊断模式分为：
- `--eval-mode one_shot`：只执行一次 option 决策窗口，用于检查即时方向，但窗口可能过短；
- `--eval-mode fixed_window --window-lowlevel-steps 16`：执行固定数量的低层 step，用于判断 option 的短中期专属性，推荐在训练高层 PPO 前优先查看；
- `--eval-mode sustained`：重复执行同一 option 直到成功、失败或超时，适合检查最终能力，但可能高估错误 option；
- `--eval-mode both`：保持兼容，同时输出 `one_shot` 与 `sustained` 两个端点模式。

如果某个 option 在多数 phase 的 fixed-window 改善分数都是 top-1，或某个 phase 下所有 option 分数接近，则当前 sequential benchmark 缺少 option 区分度，不应直接开始训练 PPO。

如需判断异常来自底层策略还是 high-level phase 注入分布偏移，请运行：
```powershell
python -m src.evaluation.eval_phase_canonical_alignment --episodes 5 --option-duration 4
```

`eval_phase_canonical_alignment.py` 会先打印 canonical 低层主场景与 injected high-level phase 的初始 geometry，再在 checkpoint 可用时对同一组 option 输出 one-shot improvement matrix 和 `alignment_gap`。`rear_vertical` 是复合 phase，没有单一 canonical 低层场景，因此只打印 injected geometry。

基础 sequential phase（rear/flank/boundary/vertical）的注入 geometry 由四个 canonical 低层主场景派生，避免高层调用分布与底层训练分布维护两套手写参数。`rear_vertical` 仍是显式复合 phase。

### 6.8 高层轨迹可视化
```powershell
# 绘制训练好的高层 selector 成功轨迹（默认按 phase 分段，避免把 phase reset 画成真实飞行）
python -m src.evaluation.plot_highlevel_trajectories --mode highlevel --scenario-set sequential --scenario-name sequential_rear_vertical_to_boundary --episodes 20 --only-success --max-plots 5 --break-at-phase-transition

# 绘制 fixed pi3 的失败轨迹，便于与 selector 对比
python -m src.evaluation.plot_highlevel_trajectories --mode fixed --fixed-policy 2 --scenario-set sequential --scenario-name sequential_rear_vertical_to_boundary --episodes 20 --only-failure --max-plots 5 --break-at-phase-transition
```

`plot_highlevel_trajectories.py` 会保存逃跑方与追击方的 3D 轨迹、起终点、phase 起点、option 切换标记；在 `continuous_pursuit` 下还会标注 regime 切换点和 low-level step 数。PNG 与被绘图 episode 的 summary CSV 默认写入 `outputs/evaluation/highlevel_traj_plots/`。脚本优先保留 high-level 成功轨迹、fixed pi3 失败轨迹，并特别关注 `sequential_rear_vertical_to_boundary`。

注意：当前 sequential 图是 **phase-based benchmark rollout**，phase transition 会注入下一阶段状态，不是完全连续物理追逐。脚本默认 `--break-at-phase-transition`，每个 phase 单独画线，避免把 reset jump 误画成长直线；如需展示跳转，可加 `--show-phase-reset-jump`，它会用灰色虚线标注 `phase reset jump (not physical)`。`continuous_pursuit` 图不需要 phase 断线，因为 evader 不会在 regime 切换时 teleport。`--one-per-scenario` 与 `--one-per-option-sequence` 可减少重复图。`--showcase-mode continuous` 仍只是展示接口；不要通过插值或平滑伪造跨 phase 的连续轨迹。

连续追逃轨迹示例：
```powershell
python -m src.evaluation.plot_highlevel_trajectories --mode highlevel --scenario-set continuous_pursuit --episodes 2 --high-model outputs/checkpoints/ppo_highlevel_switch.zip --max-plots 1
```

### 6.9 冻结低层后训练上层 PPO 切换器
```powershell
python -m src.training.train_highlevel --timesteps 300000 --model-out outputs/checkpoints/ppo_highlevel_switch.zip
```

> 目标建议：
> - 四个低层策略在各自主场景先达到 60~70% 胜率；
> - 再训练上层切换器，整体目标 90% 左右。
