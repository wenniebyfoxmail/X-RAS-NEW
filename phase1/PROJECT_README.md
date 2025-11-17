# Phase-Field VPINN/DRM 断裂力学求解器

基于变分物理信息神经网络(VPINN)和深度Ritz方法(DRM)的相场断裂力学求解器。

## 📋 文件结构

```
phase_field_vpinn.py          # 主求解器（核心实现）
test_phase_field_vpinn.py     # 测试脚本（SENT基准问题）
PROJECT_README.md              # 本文件
```

## 🔬 理论背景

### 相场断裂模型

该求解器实现了基于变分原理的相场断裂模型，其中裂纹由损伤场 `d(x) ∈ [0,1]` 表征：
- `d = 0`: 完好材料
- `d = 1`: 完全破坏

### 总能量泛函

```
E[u,d] = ∫_Ω [g(d)ψ⁺(ε) + ψ⁻(ε) + (Gₓ/c₀)(w(d)/l + l|∇d|²)] dΩ
```

其中：
- `g(d) = (1-d)² + k`: 退化函数
- `ψ⁺, ψ⁻`: 拉伸/压缩能量分解
- `w(d) = d²`: 裂纹密度函数
- `Gₓ`: 断裂能
- `l`: 长度尺度参数

### 关键特性

1. **能量分解**：将应变能分解为拉伸和压缩部分，仅拉伸能驱动损伤
2. **历史场**：`H = max_{t≤s} ψ⁺(ε(t))` 保证加载路径依赖性
3. **不可逆性**：`d_n ≥ d_{n-1}` 确保裂纹只能增长不能愈合

## 🏗️ 代码架构

### 1. 神经网络 (`§1`)

```python
u_theta(x) → (u, v)    # 位移场网络
d_phi(x) → d ∈ [0,1]   # 损伤场网络（sigmoid激活）
```

### 2. 自动微分模块 (`§2`)

使用 `torch.autograd.grad` 计算：
- `compute_strain(u, x)`: 应变张量 ε(u)
- `compute_energy_split(ε)`: 能量分解 ψ⁺, ψ⁻
- `compute_d_gradient(d, x)`: 损伤梯度 ∇d

### 3. DRM损失函数 (`§3`)

```python
L_total = L_energy + L_BC + L_irrev
```

- **L_energy**: 能量泛函的蒙特卡洛积分
- **L_BC**: 边界条件罚函数
- **L_irrev**: 不可逆性罚函数

### 4. 准静态求解器 (`§4`)

实现 **Algorithm 1** 双循环结构：

**外循环** (加载步 n):
1. 更新边界条件 g_n
2. 保存 d_{n-1}
3. 更新并固定历史场 H
4. 进入内循环

**内循环** (优化epochs):
- 梯度下降优化 u 和 d 网络
- 最小化总损失函数

## 🚀 使用方法

### 快速开始

```python
# 导入
from phase_field_vpinn import DisplacementNetwork, DamageNetwork, PhaseFieldSolver

# 创建网络
u_net = DisplacementNetwork(layers=[2, 64, 64, 64, 2])
d_net = DamageNetwork(layers=[2, 64, 64, 64, 1])

# 配置问题
config = {
    'E': 210.0,      # 杨氏模量
    'nu': 0.3,       # 泊松比
    'G_c': 2.7e-3,   # 断裂能
    'l': 0.02,       # 长度尺度
    'lr_u': 1e-3,    # 位移网络学习率
    'lr_d': 1e-3,    # 损伤网络学习率
}

# 创建求解器
solver = PhaseFieldSolver(config, u_net, d_net)

# 准静态求解
history = solver.solve_quasi_static(
    loading_steps=[0.0, 0.005, 0.01],
    x_domain=domain_points,
    x_bc=boundary_points,
    get_bc_func=your_bc_function,
    n_epochs_per_step=1000
)
```

### 运行测试

```bash
python test_phase_field_vpinn.py
```

这将运行单边缺口拉伸(SENT)基准测试，输出：
- `sent_result.png`: 位移场和损伤场可视化
- `damage_evolution.png`: 损伤演化曲线

## 📊 SENT 基准问题

**几何**：矩形板 [0,1]×[0,1]，左侧中部有初始缺口

**边界条件**：
- 下边界 (y=0): 固定 u=v=0
- 上边界 (y=1): 拉伸 v=δ, u=0

**材料参数**：
- 杨氏模量: E = 210 GPa
- 泊松比: ν = 0.3
- 断裂能: Gₓ = 2.7×10⁻³ kN/mm
- 长度尺度: l = 0.02

**预期行为**：
损伤场应该从缺口尖端开始，沿着与拉伸方向垂直的路径传播，形成Mode-I裂纹。

## ⚙️ 关键参数调优

### 网络超参数
- **隐藏层宽度**: 64-128 (更复杂问题需要更宽)
- **隐藏层深度**: 3-5层
- **学习率**: 1e-3 到 1e-4

### 损失权重
- **weight_bc**: 100-1000 (边界条件越重要权重越大)
- **weight_irrev**: 100-1000 (防止损伤回退)

### 采样点
- **域内点**: 1000-5000 (更精细的场需要更多点)
- **边界点**: 100-500

### 训练策略
- **n_epochs_per_step**: 500-2000 (每个加载步)
- **n_loading_steps**: 5-20 (取决于所需分辨率)

## 🔧 扩展建议

### 1. 更复杂的几何
修改采样点生成函数，支持：
- 多个初始缺口
- 非矩形域
- 夹杂物

### 2. 更高级的BC
实现：
- 混合边界条件 (Dirichlet + Neumann)
- 接触边界条件
- 加载-卸载循环

### 3. 自适应采样
在损伤高梯度区域增加采样点密度。

### 4. 3D扩展
- 修改网络输入/输出维度
- 实现3D应变计算
- 体积单元积分

## 📚 参考文献

1. **相场模型**:
   - Miehe, C., Hofacker, M., & Welschinger, F. (2010). "A phase field model for rate-independent crack propagation"

2. **能量分解**:
   - Amor, H., Marigo, J. J., & Maurini, C. (2009). "Regularized formulation of the variational brittle fracture"

3. **PINN方法**:
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks"

4. **DRM方法**:
   - E, W., & Yu, B. (2018). "The deep Ritz method: A deep learning-based numerical algorithm"

## 🐛 故障排除

### 问题：损失不收敛
- **解决**: 降低学习率，增加epochs，检查边界条件权重

### 问题：损伤场全为0或1
- **解决**: 
  - 检查 Gₓ 和 l 参数是否合理
  - 增加 weight_irrev
  - 降低加载步长

### 问题：裂纹路径不合理
- **解决**:
  - 增加域内采样点数
  - 减小长度尺度 l
  - 在裂纹区域使用自适应采样

### 问题：训练速度慢
- **解决**:
  - 使用GPU: `config['device'] = 'cuda'`
  - 减少采样点数
  - 减小网络规模

## 📝 许可证

本代码仅供学习和研究使用。

## 👤 作者

Claude AI Assistant

---

**注意**: 这是一个教学实现，用于演示VPINN/DRM方法在断裂力学中的应用。对于生产级应用，建议进行更严格的验证和优化。
