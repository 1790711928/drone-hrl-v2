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
python -m src.main --scenario s1_close_threat --steps 20 --save-plot --plot-path outputs/trajectory.png
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
python -m src.main --scenario s1_close_threat --steps 20
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
python -m src.main --scenario s1_close_threat --steps 20
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
python -m src.main --scenario s1_close_threat --steps 20
python -m pytest -q
```


## 6) 接入 SAC 训练（已可用）

> 默认已关闭 tensorboard 依赖（`--log-dir` 为空）。
> 如果你传了 `--log-dir` 但没安装 tensorboard，训练脚本会自动关闭该日志并继续训练。
> 若要启用 tensorboard，可执行：`python -m pip install tensorboard`，并传入 `--log-dir outputs/logs/...`。

### 6.1 单场景低层训练
```powershell
python -m src.training.train_lowlevel --scenario rear_close_threat --timesteps 120000 --model-out outputs/checkpoints/sac_low_1_rear_close_threat.zip
```

可选场景：
- `rear_close_threat`（后方近距离威胁）
- `flank_encirclement`（侧翼包围）
- `boundary_constrained`（边界受限）
- `vertical_z_threat`（垂直 z 轴威胁）

### 6.2 四个场景分别训练
```powershell
python -m src.training.train_lowlevel_all --timesteps 120000
```

训练日志默认在：`outputs/logs/sac`。



### 6.3 先做 4x4 低层评估（不同时训练）
```powershell
python -m src.evaluation.eval_lowlevel_matrix --episodes 30
```

判据：每个低层策略 `pi_i` 在自己的主场景 `S_i` 上应当最好（至少不差于其它策略）。

### 6.4 冻结低层后训练上层 PPO 切换器
```powershell
python -m src.training.train_highlevel --timesteps 300000 --model-out outputs/checkpoints/ppo_highlevel_switch.zip
```

> 目标建议：
> - 四个低层策略在各自主场景先达到 60~70% 胜率；
> - 再训练上层切换器，整体目标 90% 左右。
