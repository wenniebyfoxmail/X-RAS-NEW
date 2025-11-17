# Phase-Field VPINN/DRM Solver

相场断裂力学求解器，基于变分物理信息神经网络（VPINN）和深度Ritz方法（DRM）。

## 概述

### Phase-1: 单域求解器
基础 VPINN/DRM 实现，使用单一神经网络学习位移场 u(x) 和损伤场 d(x)。

### Phase-2: X-RAS-PINN 求解器
扩展 PINN 框架，结合：
- **域分解 (XPINN)**: 将几何划分为裂尖区域和远场区域
- **自适应采样 (RAS)**: 基于物理指标的自适应采样策略

---

## Phase-2 实现细节

### 1. 域分解 (Domain Decomposition)

#### 分区策略
将计算域 Ω 划分为两个子域：
- **Ω_sing**: 裂尖区域，半径 r_sing 内的圆形区域
- **Ω_far**: 远场区域，剩余部分

```python
def partition_domain(x, crack_tip, r_sing):
    """
    基于到裂尖的距离进行分区
    dist = ||x - x_tip|| 
    Ω_sing: dist <= r_sing
    Ω_far:  dist > r_sing
    """
```

#### 网络架构
- **裂尖域网络**: 高容量网络（4层，每层128神经元）
  - 捕捉裂尖附近的应力奇异性
  - `u_net_sing`, `d_net_sing`

- **远场域网络**: 标准容量网络（3层，每层64神经元）
  - 处理平滑的远场响应
  - `u_net_far`, `d_net_far`

### 2. XPINN 能量型损失函数

不再使用强形式残差（PDE残差），而是直接最小化能量泛函：

```
L_total = L_energy_sing + L_energy_far + L_bc + L_interface
```

其中：
- **L_energy_sing/far**: 各子域的 DRM 能量（弹性能 + 裂纹能）
- **L_bc**: 边界条件损失
- **L_interface**: 接口损失（见下文）

### 3. 接口损失

在两域交界面 Γ_I 上强制：

#### (a) 位移连续性
```
L_u = w_u · ||u_sing(x_I) - u_far(x_I)||²
```

#### (b) 牵引力平衡
```
L_σ = w_σ · ||σ_sing·n_1 + σ_far·n_2||²
```

其中 n_1, n_2 为接口两侧的法向量（n_1 = -n_2）。

**实现方法**: 
- 当前使用惩罚法（penalty method）
- 未来可改进为 Nitsche's method，避免对权重参数敏感

### 4. 自适应采样 (RAS)

#### 融合物理指标
结合两种物理量来指导采样：

```python
eta_fused = (1 - β) · sed_norm + β · grad_d_norm
```

其中：
- **sed** (应变能密度): `σ:ε = σ_xx·ε_xx + σ_yy·ε_yy + 2σ_xy·ε_xy`
  - 捕捉高应力区域
  
- **|∇d|** (损伤梯度): `||∇d||_2`
  - 捕捉裂纹扩展前沿

- **β** ∈ [0, 1]: 融合权重（默认 0.5，等权重）

#### 重采样策略
```python
def resample_points(x_old, indicator, N_add):
    """
    1. 构造概率分布: p_i ∝ indicator_i
    2. 有放回采样 N_add 个点
    3. 返回新采样点集
    """
```

**效果**: 采样点会自动聚集到裂尖和高应力梯度区域。

### 5. 三阶段训练流程

#### Phase 1: 远场预训练
```
目标: 预训练远场网络
操作: 
  - 冻结裂尖域网络 (u_sing, d_sing)
  - 仅更新远场网络 (u_far, d_far)
  - 最小化: L_energy_far + L_bc
持续: N_pre epochs (如 2000)
```

#### Phase 2: 裂尖聚焦 + 自适应采样
```
目标: 学习裂尖区域的奇异性，并动态增加采样点
操作:
  - 解冻裂尖域网络
  - 可选：冻结远场网络（提高效率）
  - 自适应循环 (N_adapt 次):
      for k in range(N_adapt):
          1. 训练 N_inner epochs
          2. 计算融合指标 eta
          3. 重采样添加 N_add 个新点
          4. 更新历史场 H_sing
持续: N_adapt × N_inner epochs (如 5 × 500)
```

#### Phase 3: 联合精化
```
目标: 联合优化所有网络，确保全局一致性
操作:
  - 解冻所有网络
  - 降低学习率（× 0.1）
  - 最小化完整的 L_total
持续: N_joint epochs (如 1000)
```

---

## 使用方法

### 基本使用

```python
from phase_field_vpinn import XRaSPINNSolver

# 配置问题
config = {
    'E': 210e3, 'nu': 0.3, 'G_c': 2.7, 'l': 0.015,
    'crack_tip': np.array([0.3, 0.5]),
    'r_sing': 0.15,
    'weights': {'lambda_bc': 100.0, 'lambda_int': 10.0}
}

# 创建求解器
solver = XRaSPINNSolver(config)

# 训练
results = solver.train(
    x_sing_init=x_sing,
    x_far=x_far,
    x_bc=x_bc,
    u_bc=u_bc,
    x_I=x_I,
    normal_I=normal_I,
    config=train_config
)

# 预测
u, d = solver.predict(x_test)

# 可视化采样分布
solver.visualize_sampling(
    x_sing=results['x_sing_final'],
    x_far=results['x_far'],
    save_path='figs/xras_sampling_scatter.png'
)
```

### 运行示例

```bash
python test_xras_pinn.py
```

这将：
1. 运行边缘裂纹拉伸问题
2. 生成采样分布可视化 (`figs/xras_sampling_scatter.png`)
3. 生成解场可视化 (`figs/xras_solution_fields.png`)
4. 打印训练历史摘要

---

## 输出可视化

### 采样分布图 (`xras_sampling_scatter.png`)
展示：
- 蓝色点: 远场采样点
- 红色点: 裂尖域采样点（密度更高）
- 绿色星: 裂尖位置
- 绿色虚线圆: 裂尖区域边界 (r = r_sing)

### 解场图 (`xras_solution_fields.png`)
展示：
- 位移 u, v 分量
- 损伤场 d
- 位移幅值 |u|

---

## 关键参数

### 域分解参数
- `crack_tip`: 裂尖坐标 (x, y)
- `r_sing`: 裂尖区域半径（建议 0.1 ~ 0.2 倍域尺寸）

### 训练参数
- `N_pre`: Phase 1 epochs (1000-2000)
- `N_adapt`: Phase 2 自适应循环次数 (3-5)
- `N_inner`: Phase 2 每次循环 epochs (500-1000)
- `N_joint`: Phase 3 epochs (1000-2000)
- `N_add`: 每次新增点数 (50-100)
- `beta`: 融合权重 (0.3-0.7，默认 0.5)

### 损失权重
- `lambda_bc`: 边界条件权重 (100)
- `lambda_int`: 接口损失权重 (10-50)
- `w_u`: 位移连续性权重 (1.0)
- `w_sigma`: 牵引力平衡权重 (1.0)

---

## 理论背景

### 相场断裂模型
总能量泛函：
```
E[u, d] = ∫_Ω [g(d)ψ⁺(ε) + ψ⁻(ε) + G_c/c₀(w(d)/l + l|∇d|²)] dΩ
```

其中：
- `g(d) = (1-d)² + k`: 退化函数
- `ψ⁺, ψ⁻`: 拉伸/压缩能量分解
- `w(d) = d²`: 裂纹密度函数

### XPINN 原理
在域 Ω = Ω_sing ∪ Ω_far 上：
```
E_total = E[u₁,d₁]_{Ω_sing} + E[u₂,d₂]_{Ω_far} + Penalty(Γ_I)
```

### RAS 原理
通过物理指标 η(x) 构造重要性采样分布：
```
p(x) ∝ η(x) = (1-β)·SED(x) + β·|∇d(x)|
```

---

## 验收标准 ✓

- [x] 域分解函数 `partition_domain`
- [x] 网络构建函数 `build_phase_field_network`
- [x] 应力计算函数 `compute_stress`
- [x] 接口损失函数 `compute_interface_loss`
- [x] 融合指标计算 `compute_indicator`
- [x] 重采样函数 `resample_points`
- [x] XRaSPINNSolver 类
  - [x] 三阶段训练流程
  - [x] 自适应采样循环
  - [x] 历史场更新
- [x] 可视化函数 `visualize_sampling`
- [x] 测试示例 `test_xras_pinn.py`
- [x] 文档说明

---

## 文件结构

```
.
├── phase_field_vpinn.py          # 主代码文件
│   ├── Phase-1: 单域求解器
│   └── Phase-2: X-RAS-PINN 求解器
├── test_xras_pinn.py             # X-RAS-PINN 测试示例
├── README.md                      # 本文档
└── figs/                          # 可视化输出目录
    ├── xras_sampling_scatter.png
    └── xras_solution_fields.png
```

---

## 未来改进方向

1. **Nitsche's method**: 替换接口惩罚法，提高鲁棒性
2. **hp-自适应**: 结合网络容量自适应（不仅是采样点）
3. **并行化**: 两域网络并行训练
4. **动态域分解**: 根据损伤场动态调整 Ω_sing
5. **多裂纹扩展**: 扩展到多裂纹问题

---

## 参考文献

- Amor, H., et al. (2009). "Regularized formulation of the variational brittle fracture with unilateral contact."
- Miehe, C., et al. (2010). "A phase field model for rate-independent crack propagation."
- Karniadakis, G. E., et al. (2021). "Physics-informed machine learning."
- XPINN: Jagtap, A. D., & Karniadakis, G. E. (2020). "Extended Physics-Informed Neural Networks."

---

## 联系与支持

如有问题或建议，请查阅代码注释或运行测试示例。

**最后更新**: 2025-11
