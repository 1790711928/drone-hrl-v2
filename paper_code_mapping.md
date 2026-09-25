# Drone HRL 代码—论文公式映射

本文档基于当前代码版本整理 low-level SAC、high-level PPO、三维动力学、reward 和 skill switching penalty 的论文公式映射。

## 1. Observation 空间：25 维

低层 SAC 与高层 PPO 共享同一个 25 维 observation vector。固定顺序由 `PursuitEscapeGymEnv.OBS_KEYS` 定义：

| 维度 | 字段 | 数学定义/来源 |
|---:|---|---|
| 1 | `dx` | `(x_e-x_p)/L_xy` |
| 2 | `dy` | `(y_e-y_p)/L_xy` |
| 3 | `dz` | `(z_e-z_p)/L_z` |
| 4 | `distance` | `d/sqrt(2L_xy^2+L_z^2)` |
| 5 | `closing_speed` | `(d_{t-1}-d_t)/dt / V_max` |
| 6 | `evader_speed` | `v_e/V_max` |
| 7 | `pursuer_speed` | `v_p/V_max` |
| 8 | `evader_yaw_sin` | `sin(psi_e)` |
| 9 | `evader_yaw_cos` | `cos(psi_e)` |
| 10 | `pursuer_yaw_sin` | `sin(psi_p)` |
| 11 | `pursuer_yaw_cos` | `cos(psi_p)` |
| 12 | `evader_pitch` | `clip(theta_e/(pi/2), -1, 1)` |
| 13 | `pursuer_pitch` | `clip(theta_p/(pi/2), -1, 1)` |
| 14 | `los_cos` | `f_e dot l_hat` |
| 15 | `boundary_margin_x` | `m_x/(L_xy/2)` |
| 16 | `boundary_margin_y` | `m_y/(L_xy/2)` |
| 17 | `boundary_margin_z` | `m_z/L_z` |
| 18 | `min_boundary_margin` | `min(m_x,m_y,m_z)/max(L_xy/2,L_z)` |
| 19 | `normalized_step` | `step_count/max_steps` |
| 20 | `evader_x_norm` | `2(x_e-x_min)/L_xy-1` |
| 21 | `evader_y_norm` | `2(y_e-y_min)/L_xy-1` |
| 22 | `evader_z_norm` | `2(z_e-z_min)/L_z-1` |
| 23 | `threat_forward` | `q_hat dot f_e` |
| 24 | `threat_right` | `q_hat dot r_e` |
| 25 | `threat_up` | `q_hat dot u_e` |

其中：

\[
\Delta x=x_e-x_p,\quad \Delta y=y_e-y_p,\quad \Delta z=z_e-z_p,
\]

\[
d=\sqrt{\Delta x^2+\Delta y^2+\Delta z^2},
\quad L_{xy}=x_{max}-x_{min},
\quad L_z=z_{max}-z_{min}.
\]

默认边界为 `x,y in [-50,50]`、`z in [0,50]`，默认最大步数为 600。速度归一化尺度为

\[
V_{max}=\max(v^e_{max},v^p_{max},1).
\]

### 1.1 LOS 与威胁方向

逃逸机 forward 方向：

\[
f_e=(\cos\theta_e\cos\psi_e,
\cos\theta_e\sin\psi_e,
\sin\theta_e).
\]

从追击机指向逃逸机的 LOS 单位向量：

\[
\hat l=\frac{p_e-p_p}{\|p_e-p_p\|},
\qquad los\_cos=f_e\cdot\hat l.
\]

逃逸机本体系的 right/up 轴为

\[
r_e=(-\sin\psi_e,\cos\psi_e,0),
\]

\[
u_e=(-\sin\theta_e\cos\psi_e,
-\sin\theta_e\sin\psi_e,
\cos\theta_e).
\]

威胁方向使用从逃逸机指向追击机的单位向量

\[
\hat q=\frac{p_p-p_e}{\|p_p-p_e\|},
\]

并计算 `threat_forward/right/up` 为其在逃逸机本体系三个轴上的投影。

## 2. Action 空间

### 2.1 Low-level SAC

低层 action 是连续三维向量：

```python
spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
```

\[
\mathcal A_L=[-1,1]^3,
\qquad a_L=(a_{acc},a_{yaw},a_{pitch}).
\]

物理控制量映射为：

\[
a=a_{acc},
\qquad \dot\psi=a_{yaw}\dot\psi_{max},
\qquad \dot\theta=a_{pitch}\dot\theta_{max}.
\]

默认 `yaw_rate_max=1.0`、`pitch_rate_max=0.7`。

### 2.2 High-level PPO

高层 action 是四分类离散变量：

```python
spaces.Discrete(4)
```

| action | option | 默认策略 |
|---:|---|---|
| 0 | Rear | `sac_low_1_rear_close_threat.zip` |
| 1 | Flank | `sac_low_2_flank_threat.zip` |
| 2 | Boundary | `sac_low_3_boundary_constrained.zip` |
| 3 | Vertical | `sac_low_4_vertical_z_threat.zip` |

每次高层决策默认执行 `option_duration=8` 个 low-level steps。

## 3. 环境动力学

### 3.1 单步运动学

速度：

\[
v_{t+1}=clip(v_t+a_t\Delta t,v_{min},v_{max}).
\]

姿态：

\[
\psi_{t+1}=\psi_t+clip(\dot\psi_t,-\dot\psi_{max},\dot\psi_{max})\Delta t,
\]

\[
\theta_{t+1}=clip(\theta_t+clip(\dot\theta_t,-\dot\theta_{max},\dot\theta_{max})\Delta t,-\pi/3,\pi/3).
\]

速度分量：

\[
v_x=v\cos\theta\cos\psi,
\quad v_y=v\cos\theta\sin\psi,
\quad v_z=v\sin\theta.
\]

位置：

\[
x_{t+1}=x_t+v_x\Delta t,
\quad y_{t+1}=y_t+v_y\Delta t,
\quad z_{t+1}=z_t+v_z\Delta t.
\]

默认 `dt=0.1`。

### 3.2 规则追击机

追击机目标角度：

\[
\psi_p^*=atan2(y_e-y_p,x_e-x_p),
\]

\[
\theta_p^*=atan2(z_e-z_p,\sqrt{(x_e-x_p)^2+(y_e-y_p)^2}).
\]

目标速度：

\[
v_p^*=\max(\rho v_e,v_p),
\]

其中默认追击速度比例 `rho=1.2`。控制量为

\[
a_p=v_p^*-v_p,
\quad \dot\psi_p=\psi_p^*-\psi_p,
\quad \dot\theta_p=\theta_p^*-\theta_p.
\]

## 4. 四个 low-level SAC reward

统一 reward 骨架为

\[
r_t=r_{distance}+r_{energy}+r_{survive}+r_{boundary}+r_{scenario}+r_{terminal}.
\]

\[
r_{distance}=w_d(d_{t+1}-d_t),
\]

\[
r_{energy}=-w_e(a_{acc}^2+\dot\psi^2+\dot\theta^2),
\qquad r_{survive}=w_s.
\]

soft boundary risk 定义为：

\[
B(p)=
\begin{cases}
0,&\tilde m\ge0.35,\\
((0.35-\tilde m)/0.35)^2,&\tilde m<0.35,
\end{cases}
\]

其中

\[
\tilde m=clip\left(
\min\left(\frac{m_x}{L_x/2},\frac{m_y}{L_y/2},\frac{m_z}{L_z}\right),0,1\right).
\]

\[
r_{boundary}=-w_bB(p_e).
\]

终止奖励：

\[
r_{terminal}=\begin{cases}
+50,& escaped,\\
-50,& captured,\\
-20,& out\_of\_bounds,\\
0,& running/timeout.
\end{cases}
\]

### 4.1 Rear option

权重：`w_distance=1.2`、`w_energy=0.0010`、`w_survive=0.02`、`w_boundary_risk=0.01`。

\[
 r^{rear}=1.2\Delta d-0.0010\|a\|^2+0.02-0.01B+r_{terminal}.
\]

### 4.2 Flank option

权重：`1.0`、`0.0012`、`0.02`、`0.03`。

\[
 r^{flank}=1.0\Delta d-0.0012\|a\|^2+0.02-0.03B+r_{terminal}.
\]

当前代码没有额外的 `threat_right` flank reward shaping；flank 的差异主要来自 reward profile 和初始场景几何。

### 4.3 Boundary option

基础权重：`0.9`、`0.0010`、`0.02`、`0.08`。

\[
r^{boundary}=0.9\Delta d-0.0010\|a\|^2+0.02-0.08B+r_{scenario}^{boundary}+r_{terminal}.
\]

其中

\[
r_{scenario}^{boundary}=
0.18\max(0,\Delta m)
+0.03I(\bar m\ge0.25)
-0.03I(\bar m<0.15)
\]

\[
\quad-0.08\max(0,-\Delta m_{critical})
-0.06[\max(0,0.25-\bar m_z)]^2
-0.004|\dot\theta|.
\]

### 4.4 Vertical option

基础权重：`1.25`、`0.0010`、`0.02`、`0.04`。

\[
r^{vertical}=1.25\Delta d-0.0010\|a\|^2+0.02-0.04B+r_{scenario}^{vertical}+r_{terminal}.
\]

归一化垂直分离：

\[
s_z=\min(|z_e-z_p|/L_z,0.5).
\]

\[
r_{scenario}^{vertical}=0.10s_z
-0.25[\max(0,0.30-\bar m_z)]^2.
\]

## 5. High-level PPO observation、action、reward

### Observation

高层直接复用 low-level 的 25 维 observation space：

```python
self.observation_space = self.inner.observation_space
```

当前没有显式的 phase one-hot、regime one-hot、previous-option 或 option ID 输入。

### Action

\[
\mathcal A_H=\{0,1,2,3\}
\]

分别选择 Rear、Flank、Boundary、Vertical 四个 frozen low-level policies。

### Basic / mixed / composite reward

若 option (o_t) 执行 (H_t\le8) 个 low-level steps，则

\[
R_t^H=\sum_{\tau=0}^{H_t-1}r_{t,\tau}^L
-\lambda_{switch}I(o_t\ne o_{t-1}).
\]

### Sequential reward

phase 完成需要连续 3 个 low-level steps 满足 phase condition。每个 phase 完成增加 5，全部 sequence 完成增加 20：

\[
R_t^{H,seq}
=\sum_\tau r_{t,\tau}^L
-50I(base\ escape)
+5I(phase\ complete)
+20I(sequence\ complete)
-\lambda_{switch}I(switch).
\]

### Continuous pursuit / showcase reward

每个 low-level step额外增加 0.01：

\[
R_t^{H,cont}
=\sum_\tau(r_{t,\tau}^L+0.01)
+25I(long\ horizon\ success)
-\lambda_{switch}I(switch).
\]

在 continuous 模式中，基础环境的普通 `escaped` 不直接作为长时 episode 结束信号。

## 6. Skill switching penalty

默认参数：

```python
switch_penalty = 0.02
```

数学形式：

\[
P_{switch,t}=\lambda_{switch}I(o_t\ne o_{t-1}).
\]

代码逻辑为：

```python
if self.prev_option is not None and idx != self.prev_option:
    self.switch_count += 1
    total_reward -= self.switch_penalty
self.prev_option = idx
```

该逻辑同时存在于普通/sequential 路径和 continuous/showcase 路径。惩罚按高层切换事件计算，而不是每个 low-level step 重复扣除。

## 7. 论文建模时需要注明的实现事实

1. 当前 observation 实际为 25 维，应以 `OBS_KEYS` 为准。
2. 高层与低层共享 observation schema；高层没有显式 phase/regime/previous-option 特征。
3. 四个 low-level reward 使用统一骨架，不是四个完全独立的 reward；boundary 和 vertical 有显式 scenario-specific shaping。
4. 当前 flank reward 没有直接使用 `threat_right` 作为 reward 项。
5. high-level reward 是 option 执行期间 low-level reward 的累计和，没有 duration normalization。
6. switching penalty 只在 option 变化时扣一次。
7. ESAS 属于 evaluation metric，不是训练 reward；如果论文同时介绍 reward 和行为专属性，应明确区分二者。

## 8. 代码定位索引

| 内容 | 文件 |
|---|---|
| Observation key 顺序与 Gym space | `src/training/sac_env.py` |
| Observation 数值构造、LOS、step | `src/env/pursuit_escape_env.py` |
| 三维运动学与追击控制 | `src/env/dynamics.py` |
| Reward profiles、boundary risk、scenario reward | `src/env/reward.py` |
| Termination conditions | `src/env/termination.py` |
| 四个 canonical scenarios | `src/env/scenarios.py` |
| HighLevelOptionEnv、option duration、switch penalty | `src/training/highlevel_env.py` |
| PPO training CLI 与 frozen SAC 路径 | `src/training/train_highlevel.py` |
| ESAS evaluation metric | `src/evaluation/escape_skill_alignment.py` |

