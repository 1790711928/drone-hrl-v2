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

> 当前默认训练空间已放大：x/y 为 `[-200, 200]`，z 为 `[0, 80]`（比之前更大，便于逃逸策略学习）。


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
3. 冻结低层后，在 `src/training/train_highlevel.py` 接 PPO 选择器。
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
