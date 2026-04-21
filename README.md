# drone-hrl-v2

无人机 3D 追逃（分层强化学习）入门版工程。

> 你是新手也没关系：这个仓库现在按“搭积木”方式组织，先能跑通 `main`，再逐步接入 SAC/PPO 训练。

---

## 1. 项目结构（像搭积木）

```text
drone-hrl-v2/
├─ src/
│  ├─ main.py                      # 新手入口：先跑这个看环境一步步输出
│  ├─ env/
│  │  ├─ dynamics.py               # 3D动力学 + 规则追击
│  │  ├─ termination.py            # 终止条件（k步逃脱/k步捕获）
│  │  ├─ scenarios.py              # 4个场景初始条件
│  │  └─ pursuit_escape_env.py     # reset/step 环境主循环
│  ├─ training/
│  │  ├─ train_lowlevel.py         # 低层训练入口（先做 smoke run）
│  │  └─ train_highlevel.py        # 高层训练入口（PPO占位）
│  └─ evaluation/
│     └─ metrics.py                # 评估指标数据结构
├─ configs/
│  ├─ env.yaml
│  ├─ train_lowlevel.yaml
│  ├─ train_highlevel.yaml
│  └─ eval.yaml
└─ tests/
```

---

## 2. 本地运行（Windows 新手版）

### 2.1 进入目录
```bash
cd /d E:\python程序\drone_hrl_repo\-\drone_hrl_v2
```

> 如果你当前是在 WSL / Linux / Mac，请用：
```bash
cd /workspace/drone-hrl-v2
```

### 2.2 创建并激活虚拟环境
```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2.3 安装最小依赖
```bash
python -m pip install -U pip pytest
```

### 2.4 先跑主程序（推荐）
```bash
python -m src.main --scenario s1_close_threat --steps 20
```

你会看到每一步的：
- distance（追逃距离）
- closing_speed（闭合速度）
- outcome（运行/逃脱/捕获等）
- escape_streak（连续满足逃脱条件的步数）

---

## 3. 终止条件（重点）

为了防止“瞬时逃脱导致胜率虚高”，逃脱判定不是单步，而是连续 `k_escape` 步满足：
- `distance > d_safe`
- `closing_speed <= 0`
- `los_escape_ok == True`

捕获同理，要连续 `k_capture` 步小于捕获距离。

---

## 4. 测试

```bash
python -m pytest -q
```

---

## 5. 下一步（学习路线）

1. 先熟悉 `src/main.py` 和 `src/env/pursuit_escape_env.py` 的 reset/step。
2. 在 `train_lowlevel.py` 中接入 SAC（每个场景一个低层策略）。
3. 冻结低层后，在 `train_highlevel.py` 接入 PPO 切换器。
4. 增加 4x4 评估脚本和鲁棒性扫描脚本。
