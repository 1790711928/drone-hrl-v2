# drone-hrl-v2

无人机 3D 追逃（分层强化学习）MVP。

## 当前约束
- 逃跑方：智能体（低层使用 SAC）
- 追击方：规则引导（指向逃跑方），速度设置为略快于逃跑方
- 分层流程：先冻结高层训练低层，再冻结低层训练高层（PPO 切换器）

## 目录
- `src/env/dynamics.py`: 3D 动力学与规则追击控制
- `src/env/termination.py`: 终止条件（包含 `k` 步连续逃脱判定，避免瞬时逃脱导致胜率虚高）
- `src/training/train_lowlevel.py`: 低层 SAC 训练入口（占位）
- `src/training/train_highlevel.py`: 高层 PPO 训练入口（占位）
- `configs/*.yaml`: 环境、训练、评估配置
- `tests/`: 核心逻辑单测

## 终止条件（关键）
- 捕获：`distance < d_capture` 连续 `k_capture` 步
- 逃脱：同时满足以下条件并连续 `k_escape` 步
  - `distance > d_safe`
  - `closing_speed <= 0`
  - `los_escape_ok == True`
- 越界：超出 3D 边界
- 超时：`step_count >= max_steps`

## 快速测试
```bash
python -m pytest -q
```
