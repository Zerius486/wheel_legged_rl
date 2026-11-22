# model3 机器人资产

资产位于 `src/wheel_legged_rl/assets/model3/`，沿用 `test_model_v1.0` 的 `robot.usda` + `assets/*.usdc` 布局。
`parameters.json` 保存加载和控制所需的初值、动作顺序及气弹簧参数；路径相对于该资产目录。
USD 和网格通过仓库已有的 Git LFS 规则管理，克隆仓库后需执行 `git lfs pull`。

## 结构与质量

- 23 个刚体、22 个树关节、6 个球铰闭链、4 个 1:1 链传动耦合。
- 每腿三处闭合：`llleg4 ↔ llleg`、`llleg2 ↔ thleg`、`spring_llleg ↔ llleg`。
- 球铰设为 `physics:excludeFromArticulation=true`，仍作为物理约束存在。
- 主动链轮驱动从动髋/曲柄。当前双方轴向相反，URDF 关系为 `q_follower = -q_reference`，
  USD 的 PhysX Mimic `gearing=+1`、`offset=0`。
- 五组最新 SW Link 质量、质心和惯量已换算到刚体局部坐标；其余 18 组沿用修复后的导出值，
  包括已确认的左大腿镜像。总质量 **14.1781438983 kg**。
- 与旧整机报告相比少 10.6399 g、质心相差 2.2090 mm、惯量矩阵相对差 0.87235%。
  采用最新 Link 报告，没有为匹配旧总表而补重或缩放。

## 加载与控制约定

独立资产默认 prim 为 `/Robot`，包含浮动基座，不包含地面、物理场景、灯光或世界固定关节。
树关节局部帧保留原 CAD 零位定义，刚体已经摆放在闭合初始姿态：大腿 60°、膝内角 50°。
加载后重置时，应使用 `parameters.json` 中**全部 22 个关节初值**，不能把所有关节清零或独立随机化。
平地初始根部高度为 0.1511447946 m，四元数采用 XYZW 顺序。

六个力矩输入按 `active_joints` 顺序排列：右大腿主动轮、右曲柄主动轮、左大腿主动轮、
左曲柄主动轮、右车轮、左车轮。只控制这些输入，其余关节保持被动。
沿机身 +X 前进时，右轮的速度/力矩方向为负，左轮为正。
角度及角速度在参数文件中使用 rad、rad/s；USD 关节限位与速度属性采用对应的度单位。

腿电机额定 20 N·m、峰值 40 N·m，默认连续工作按额定值限幅；轮电机额定 2.46 N·m、
堵转参考值 3.69 N·m。选定转速上限分别为 100 rpm 和 571 rpm。
这些均为减速后输出参数，外部链比为 1:1，不再乘内部减速比。
电机和链传动热特性、背隙及转子反射惯量尚未辨识。

## 气弹簧

`robot.usda` 不包含气弹簧 Drive；使用者需要根据参数显式施力，避免与额外 Drive 重复叠加。
`parameters.json` 的 `spring_joints.enabled` 是供控制器读取的配置，不会自动激活 USD 力。
两个滑动关节范围为 `[-0.07, 0]` m，正方向为压缩，`q=0` 为全压缩。
默认采用用户指定的胡克定律近似，零预载、刚度 2101.425 N/m、阻尼 0，最大伸展力 147.09975 N。

```text
compression = clamp(extension_sign * (q_at_full_extension_m - q), 0, stroke_m)
force = clamp(preload_N + stiffness_N_m * compression
              - damping_N_s_m * extension_sign * q_dot, 0, max_force_N)
joint_effort = extension_sign * force
```

当前 `extension_sign=-1`、`q_at_full_extension_m=-0.07`。此近似不代表实测气弹簧特性。

## 验证与接入范围

已用 OpenUSD 26.11 实际打开资产并验证包内相对引用、23 个刚体、关节连接、6 处球铰闭合、
4 处 Mimic、质量与主惯量重构；初始闭合误差约 4.10e-9 m。
自碰撞默认关闭，车轮使用圆柱碰撞，其余使用凸包近似。
尚未执行 Isaac/PhysX 动态仿真、GPU 多环境克隆或训练收敛验证。
本资产使用 PhysX Mimic 和闭链约束；现有任务默认的 Newton/MJWarp 后端没有针对 model3 验证，
接入时需明确选择支持这些约束的 PhysX 后端。

本次仅新增资产，未注册 model3 训练任务，也未改动 `test_model_v1.0` 的配置。
现有任务的关节名、腿长计算、复位逻辑和动作映射针对旧模型，不能仅替换 USD 路径后直接用于 model3。
后续接入需要按本参数文件配置初始姿态、驱动和弹簧，并在各环境内检查闭链及 Mimic 引用。
