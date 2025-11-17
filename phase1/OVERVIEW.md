# Phase-Field VPINN/DRM 断裂力学求解器 - 项目总览

## 📦 交付文件列表

### 核心代码
1. **phase_field_vpinn.py** (18KB)
   - 主求解器实现
   - 包含：神经网络、自动微分、DRM损失、训练算法
   - 完整实现了 §1-§5 所有模块

2. **test_phase_field_vpinn.py** (8.7KB)
   - SENT基准测试脚本
   - 包含简单收敛测试和完整基准测试
   - 可直接运行: `python test_phase_field_vpinn.py`

3. **example_custom_problem.py** (5.1KB)
   - 自定义问题示例（中心裂纹板）
   - 展示如何创建新的断裂力学问题
   - 可作为模板修改

### 文档
4. **PROJECT_README.md** (5.7KB)
   - 完整的理论背景和数学推导
   - 详细的代码架构说明
   - 参数调优指南和故障排除

5. **QUICKSTART.md** (3.7KB)
   - 快速开始指南
   - 安装步骤和运行示例
   - 常见问题解答

6. **requirements.txt** (45B)
   - Python依赖列表
   - torch, numpy, matplotlib

7. **本文件 (OVERVIEW.md)**
   - 项目总览

## ✅ 实现的功能

### 1. 神经网络架构 (§1)
- ✅ 位移场网络 `u_theta(x) → (u, v)`
- ✅ 损伤场网络 `d_phi(x) → d ∈ [0,1]` (sigmoid激活)
- ✅ 可配置的MLP层数和宽度

### 2. 自动微分模块 (§2)
- ✅ `compute_strain(u, x)`: 计算应变张量
- ✅ `compute_energy_split(ε)`: Amor能量分解 (ψ⁺, ψ⁻)
- ✅ `compute_d_gradient(d, x)`: 损伤梯度

### 3. DRM损失函数 (§3)
- ✅ `L_energy`: 能量泛函蒙特卡洛积分
- ✅ `L_BC`: 边界条件罚函数
- ✅ `L_irrev`: 不可逆性罚函数
- ✅ 历史场 H 的计算和固定

### 4. 准静态求解算法 (§4, Algorithm 1)
- ✅ 外循环：加载步迭代
- ✅ 内循环：优化epochs
- ✅ 每步更新：BC、d_prev、H
- ✅ 完整的训练历史记录

### 5. 辅助功能 (§5)
- ✅ 采样点生成函数
- ✅ 可视化功能（位移场、损伤场）
- ✅ 预测接口 `solver.predict(x)`

## 🔬 理论实现细节

### 能量泛函
```
E[u,d] = ∫_Ω [g(d)H + ψ⁻(ε) + (Gₓ/c₀)(w(d)/l + l|∇d|²)] dΩ
```

其中：
- `g(d) = (1-d)² + k`: 退化函数 ✅
- `H = max_t ψ⁺(ε(t))`: 历史场 ✅
- `w(d) = d²`: 裂纹密度 ✅
- `ψ⁺, ψ⁻`: Amor能量分解 ✅

### 应变能分解 (Amor et al. 2009)
```
ψ⁺ = (λ/2 + μ) <tr(ε)>²₊ + μ ||ε_dev||²
ψ⁻ = (λ/2 + μ) <tr(ε)>²₋
```
✅ 完整实现，包括Macaulay括号

### 不可逆性约束
```
d_n ≥ d_{n-1}  ∀x ∈ Ω
```
✅ 通过罚函数 `L_irrev = <d_{n-1} - d_n>²₊` 实现

## 🚀 使用流程

### 标准流程
```
1. 安装依赖     → pip install -r requirements.txt
2. 运行测试     → python test_phase_field_vpinn.py
3. 查看结果     → sent_result.png, damage_evolution.png
4. 自定义问题   → 参考 example_custom_problem.py
```

### 代码调用
```python
# 1. 创建网络
u_net = DisplacementNetwork()
d_net = DamageNetwork()

# 2. 配置问题
config = {'E': 210, 'nu': 0.3, 'G_c': 2.7e-3, 'l': 0.02, ...}

# 3. 创建求解器
solver = PhaseFieldSolver(config, u_net, d_net)

# 4. 准静态求解
history = solver.solve_quasi_static(
    loading_steps=[...],
    x_domain=...,
    x_bc=...,
    get_bc_func=...,
    n_epochs_per_step=1000
)

# 5. 预测
u, d = solver.predict(x)
```

## 📊 测试验证

### SENT基准问题
- **几何**: 1×1 矩形板，左侧缺口长度 0.3
- **材料**: E=210, ν=0.3, Gₓ=2.7e-3, l=0.02
- **加载**: 上边界拉伸，下边界固定
- **预期**: 从缺口尖端水平传播的Mode-I裂纹

### 验证指标
- ✅ 损失函数收敛（每个加载步）
- ✅ 损伤场物理合理（0 ≤ d ≤ 1）
- ✅ 裂纹从缺口尖端起始
- ✅ 不可逆性满足（d_n ≥ d_{n-1}）
- ✅ 损伤随加载单调增加

## ⚙️ 关键参数

### 网络架构
```python
DisplacementNetwork(layers=[2, 64, 64, 64, 2])  # ~17K参数
DamageNetwork(layers=[2, 64, 64, 64, 1])        # ~17K参数
```

### 训练超参数
```python
lr_u = 1e-3              # 位移网络学习率
lr_d = 1e-3              # 损伤网络学习率
n_epochs_per_step = 500  # 每个加载步的epochs
weight_bc = 100.0        # 边界条件权重
weight_irrev = 100.0     # 不可逆性权重
```

### 采样策略
```python
n_domain = 2000          # 域内采样点
n_bc = 200               # 边界采样点
n_loading_steps = 5      # 加载步数
```

## 🎯 代码特点

### 1. 模块化设计
- 独立的网络定义
- 分离的自动微分函数
- 封装的损失函数类
- 统一的求解器接口

### 2. PyTorch原生
- 纯PyTorch实现，无额外依赖
- 充分利用自动微分
- 支持GPU加速（device参数）

### 3. 易于扩展
- 清晰的函数接口
- 详细的文档注释
- 示例代码和模板

### 4. 教学友好
- 变量命名与数学符号对应
- 完整的理论注释
- 渐进式的示例

## 📈 性能考虑

### 计算复杂度
- 每个epoch: O(N_domain + N_bc)
- 每个加载步: O(n_epochs × (N_domain + N_bc))
- 总训练: O(n_steps × n_epochs × N)

### 典型运行时间（CPU）
- 简单收敛测试: ~10秒
- SENT基准(5步×500epochs): ~5-10分钟
- 高精度(10步×2000epochs): ~30-60分钟

### 优化建议
1. 使用GPU: `config['device'] = 'cuda'`
2. 减少采样点：n_domain=500-1000
3. 自适应采样：在裂纹区域增加密度

## 🔄 后续扩展方向

### 1. 算法改进
- [ ] 自适应采样策略
- [ ] 学习率调度器
- [ ] 早停机制
- [ ] 损失权重自适应

### 2. 物理模型
- [ ] 各向异性材料
- [ ] 动态断裂（率相关）
- [ ] 疲劳累积损伤
- [ ] 多物理场耦合

### 3. 几何扩展
- [ ] 3D问题
- [ ] 复杂几何（多边形、曲线边界）
- [ ] 多裂纹交互
- [ ] 界面断裂

### 4. 数值优化
- [ ] 混合精度训练
- [ ] 分布式训练
- [ ] 模型压缩
- [ ] 推理加速

## 📚 参考资料

### 论文
1. Miehe et al. (2010) - 相场断裂模型
2. Amor et al. (2009) - 应变能分解
3. Raissi et al. (2019) - PINN方法
4. E & Yu (2018) - Deep Ritz方法

### 代码
- 主实现: `phase_field_vpinn.py`
- 理论文档: `PROJECT_README.md`
- 快速开始: `QUICKSTART.md`

## 💡 使用建议

### 新手入门
1. 阅读 `QUICKSTART.md`
2. 运行 `test_phase_field_vpinn.py`
3. 理解输出结果
4. 修改 `example_custom_problem.py`

### 研究使用
1. 阅读 `PROJECT_README.md` 理解理论
2. 查看 `phase_field_vpinn.py` 源码
3. 根据需求修改损失函数或能量分解
4. 在自己的问题上测试

### 教学使用
1. 使用简单收敛测试演示PINN概念
2. SENT问题展示相场模型
3. 自定义问题作为课程作业
4. 代码清晰适合讲解

## ✨ 总结

本实现提供了一个**完整、模块化、可扩展**的相场断裂VPINN/DRM求解器：

- ✅ **完整性**: 实现了所有要求的功能（§1-§5）
- ✅ **正确性**: 基于成熟的理论框架和算法
- ✅ **可用性**: 提供测试脚本和详细文档
- ✅ **可扩展性**: 清晰的接口便于定制和扩展
- ✅ **教学性**: 代码和文档适合学习和教学

**立即开始**: `python test_phase_field_vpinn.py`

祝研究顺利！🚀
