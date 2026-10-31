# wheel_legged_rl

巡天御风战队 2027 赛季串联腿步兵强化学习训练工程，基于 Isaac Sim 与 Isaac Lab 环境开发。

## 网络结构

训练采用非对称 actor-critic。actor 仅接收实机可测的本体感知 $\mathbf{o}_t \in \mathbb{R}^{25}$ 与其最近 5 帧构成的历史
$\mathbf{s}_t \in \mathbb{R}^{125}$；历史经 MLP encoder $f_\phi$ 压缩为隐变量 $\mathbf{z}_t \in \mathbb{R}^{3}$，与当前观测拼接后送入
actor $\pi_\theta$。critic $V_\psi$ 额外接收特权信息，仅用于训练，不参与导出。

$$
\mathbf{s}_t = \left[\mathbf{o}_{t-4};\,\dots;\,\mathbf{o}_t\right], \qquad
\mathbf{z}_t = f_\phi(\mathbf{s}_t), \qquad
\mathbf{a}_t = \pi_\theta\left(\left[\mathbf{o}_t;\,\mathbf{z}_t\right]\right)
$$

### 模块规模

| 模块 | 结构 | 参数量 | 部署 |
| --- | --- | --- | --- |
| encoder $f_\phi$ | $125 \to 128 \to 64 \to 3$ | 24,579 | 是 |
| actor $\pi_\theta$ | $28 \to 128 \to 64 \to 32 \to 6$ | 14,246 | 是 |
| critic $V_\psi$ | $144 \to 256 \to 128 \to 64 \to 1$ | 78,337 | 否 |

### 观测 $\mathbf{o}_t$

| 索引 | 分量 | 维 | 说明 |
| --- | --- | --- | --- |
| 0:3 | $\boldsymbol{\omega}_b$ | 3 | 机体角速度，机体系 |
| 3:6 | $\mathbf{g}_b$ | 3 | 重力方向在机体系的投影，单位向量 |
| 6:9 | $(v_x^{\mathrm{cmd}},\, \dot{\psi}^{\mathrm{cmd}},\, h^{\mathrm{cmd}})$ | 3 | 线速度、偏航角速度、高度指令 |
| 9:13 | $\mathbf{q}_{\mathrm{hip}}$ | 4 | 四个髋关节角 |
| 13:19 | $\dot{\mathbf{q}}$ | 6 | 全部关节速度，4 髋 + 2 轮 |
| 19:25 | $\mathbf{a}_{t-1}$ | 6 | 上一步动作 |

### 历史与隐变量

$\mathbf{s}_t$ 由 5 帧观测按时间顺序展平构成，采样周期 $T_s = 10\ \mathrm{ms}$（100 Hz），覆盖 50 ms。
其作用是估计基座线速度：该量实机不可测，故不直接进入观测。

### 动作与执行

动作 $\mathbf{a}_t \in \mathbb{R}^{6}$ 由 4 个髋关节位置目标与 2 个轮关节速度目标组成，由驱动器内环 PD 执行：

$$
\tau_i = k_{p,i}\left(q_i^{\ast} - q_i\right) + k_{d,i}\left(\dot{q}_i^{\ast} - \dot{q}_i\right)
$$

髋关节取 $q_i^{\ast} = q_i^{\mathrm{def}} + s_q a_i$、$\dot{q}_i^{\ast} = 0$；轮关节取 $\dot{q}_i^{\ast} = s_\omega a_i$ 且 $k_{p,i} = 0$，
否则轮会被锁死在目标角度上。增益 $k_{p,i}, k_{d,i}$ 与标度 $s_q, s_\omega$ 由实车标定给出。

### 部署约束

- encoder 与 actor 合计 38,825 参数：fp32 为 151.7 KB，int8 为 37.9 KB；单次推理 38,400 MAC，
  100 Hz 下约占 480 MHz 主频的 3%，fp32 无需量化。
- 导出为单输入单输出 ONNX，确定性输出（分布均值），观测归一化包含在图内。
- opset 默认 18；STM32 工具链不接受时降至 17。

## 奖励

只做三项跟踪：行走、转向、升降。速度与偏航用 Cauchy 核（正则化项之外唯一的正奖励，远处梯度不消失），
腿长用 Gaussian 核（指令超出机构行程时允许放弃）：

| 项 | 公式 | 权重 | $\sigma$ |
| --- | --- | --- | --- |
| `track_lin_vel_x` | $1/\left(1+(e_v/\sigma)^2\right)$ | 1.0 | 0.5 m/s |
| `track_ang_vel_z` | $1/\left(1+(e_\psi/\sigma)^2\right)$ | 1.0 | 0.5 rad/s |
| `track_leg_length` | $\exp\left(-(e_L/\sigma)^2\right)$ | 0.7 | 50 mm |
| `track_leg_length`（精项） | 同上 | 0.3 | 20 mm |

$e_v = v_x^{\mathrm{cmd}} - v_x^b\cos\theta$，$e_\psi = \dot\psi^{\mathrm{cmd}} - \omega_{b,z}$，$e_L = L_0^{\mathrm{cmd}} - L_0(\mathbf{q}_{\mathrm{hip}})$ 经
$L_0 = \sqrt{l_1^2+l_2^2-2l_1l_2\cos\theta}$ 由髋角算出。实现见 `mdp/rewards.py`，机构常数取自 `docs/robot_params.md`。
