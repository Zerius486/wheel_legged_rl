# 机器人参数 — `test-model v1.0`

> 资产路径：`src/wheel_legged_rl/assets/test_model_v1.0/robot.usda`（由 SolidWorks / USD 并联五连杆轮腿模型整理）。

## 1. 模型结构概览

- **模型名称**：`test-model v1.0`（并联五连杆双轮腿机器人）
- **刚体数量**：11 个（`body`、左右前后大腿 `*_thleg1` / `*_thleg2`、左右前后小腿 `*_llleg1` / `*_llleg2`、左右轮 `*_wheel`）
- **关节配置**：
  - 10 个 Articulation 树内旋转关节（全部绕 `+X` 轴旋转）：4 个主动髋关节、4 个被动膝关节、2 个主动轮关节
  - 2 个闭链约束旋转关节：`left_leg_closure`、`right_leg_closure`（标记 `physics:excludeFromArticulation = true`，由物理求解器闭合左右小腿末端）
- **碰撞配置**：每个连杆采用单凸包（`convexHull`）近似，关闭自碰撞（`physxArticulation:enabledSelfCollisions = 0`）

- **坐标系约定**：机体系 `+X` 指向右侧车轮轴方向（左右轮位于 `x = ±0.185 m`，所有关节绕 `+X` 旋转），`+Y` 为前后滚动纵向（前后髋间距沿 `Y` 轴分布），`+Z` 竖直向上。

## 2. 机构常数（`mdp/rewards.py` 对齐）

| 变量 | 值 | 单位 | 说明 |
| --- | --- | --- | --- |
| `C_HIP` | 0.0945 | m | 半髋间距（前后髋轴沿 `Y` 轴间距 `0.189 m`） |
| `L1` | 0.210 | m | 大腿（髋轴心 ↔ 膝轴心） |
| `L2` | 0.250 | m | 小腿（膝轴心 ↔ 轮电机轴心） |
| `THETA1_0` | 0.4856 | rad | CAD 零位前大腿与竖直夹角（`27.82°`） |
| `THETA2_0` | 0.4756 | rad | CAD 零位后大腿与竖直夹角（`27.25°`） |
| `L0_MIN` | 0.1631 | m | 腿长下限（对应 `+19.88°` 髋限位） |
| `L0_MAX` | 0.4410 | m | 腿长上限（对应 `-27.25°` 髋限位） |
| `L0_DEFAULT` | 0.3468 | m | CAD 零位（`q=0`）真实虚拟腿长 |

```python
# 并联五连杆正运动学（计入左腿 quat=(0, 0, 1, 0) 镜像符号与 CAD 零位偏置）
theta1 = THETA1_0 + sign1 * q1
theta2 = THETA2_0 + sign2 * q2
y_k1, z_k1 = -C_HIP - L1 * torch.sin(theta1), -L1 * torch.cos(theta1)
y_k2, z_k2 = C_HIP + L1 * torch.sin(theta2), -L1 * torch.cos(theta2)
dy, dz = y_k2 - y_k1, z_k2 - z_k1
d = torch.sqrt(torch.clamp(dy**2 + dz**2, min=1e-12))
h = torch.sqrt(torch.clamp(L2**2 - 0.25 * d**2, min=0.0))
y_w = 0.5 * (y_k1 + y_k2) + h * (dz / d)
z_w = 0.5 * (z_k1 + z_k2) - h * (dy / d)
L0 = torch.sqrt(y_w**2 + z_w**2)
theta0 = torch.atan2(y_w, -z_w)
```

## 3. 质量与惯量参数

> `body` 已更新为实机质量 `18.0 kg`，主转动惯量按质量比缩放；总整机质量约 `19.7428 kg`。

| 连杆 | 数量 | 单体质量 (kg) | 主对角惯量 `(Ixx, Iyy, Izz)` ($\mathrm{kg\cdot m^2}$) |
| --- | --- | --- | --- |
| `body` | 1 | 18.0000 | `(0.231027, 0.251845, 0.290882)` |
| `*_thleg1` / `*_thleg2`（大腿） | 4 | 0.1422 | `(1.389e-5, 6.860e-4, 6.976e-4)` |
| `*_llleg1` / `*_llleg2`（小腿） | 4 | 0.1589 | `(1.848e-5, 1.060e-3, 1.076e-3)` |
| `*_wheel`（轮子） | 2 | 0.2692 | `(2.389e-4, 2.389e-4, 4.374e-4)` |

## 4. 关节与执行器规格

| 类别 | 关节名称 | 电机 / 驱动 | 限位（deg / rad） | 约束与默认参数 |
| --- | --- | --- | --- | --- |
| 主动髋关节 | `left_thleg1_body` | DM8009P-2EC（位置 PD） | `[-27.82, 19.88]°` / `[-0.4856, 0.3469]` | `effort_limit=20 N·m`, `vel_limit=10.472 rad/s (100 rpm)`, 基准 `kp=30 N·m/rad, kd=1.0 N·m·s/rad`（reset 时按 `×[0.8, 1.2]` 均匀域随机化） |
| 主动髋关节 | `left_thleg2_body` | DM8009P-2EC（位置 PD） | `[-20.45, 27.25]°` / `[-0.3569, 0.4756]` | 同上 |
| 主动髋关节 | `right_thleg1_body` | DM8009P-2EC（位置 PD） | `[-19.88, 27.82]°` / `[-0.3469, 0.4856]` | 同上 |
| 主动髋关节 | `right_thleg2_body` | DM8009P-2EC（位置 PD） | `[-27.25, 20.45]°` / `[-0.4756, 0.3569]` | 同上 |
| 被动膝关节 | `left_llleg1_th` | 无驱动（被动） | `[-60.00, 55.50]°` / `[-1.0472, 0.9687]` | `kp=0, kd=0` |
| 被动膝关节 | `left_llleg2_thleg2` | 无驱动（被动） | `[-60.28, 55.22]°` / `[-1.0521, 0.9638]` | `kp=0, kd=0` |
| 被动膝关节 | `right_llleg1_thleg1` | 无驱动（被动） | `[-55.50, 60.00]°` / `[-0.9687, 1.0472]` | `kp=0, kd=0` |
| 被动膝关节 | `right_llleg2_thleg2` | 无驱动（被动） | `[-55.22, 60.28]°` / `[-0.9638, 1.0521]` | `kp=0, kd=0` |
| 主动轮关节 | `left_wheel_llleg2`, `right_wheel_llleg2` | 轮毂电机（速度控制） | 连续旋转（无限位） | 裸机最大连续转矩 `0.16 N·m`、空载转速 `9400 rpm`，减速比 `268/17 ≈ 15.7647`；轮端 `effort_limit ≈ 2.5224 N·m`, `vel_limit ≈ 62.4411 rad/s (596.27 rpm)`, 基准 `kp=0, kd=0.5 N·m·s/rad`（reset 时 `kd` 按 `×[0.8, 1.2]` 均匀域随机化） |
