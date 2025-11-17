# 快速开始指南

## 安装依赖

```bash
pip install torch numpy matplotlib
```

## 运行测试

```bash
python test_phase_field_vpinn.py
```

测试包含两个部分：
1. **简单收敛测试**：验证网络能够学习基本位移场（~100 epochs，快速）
2. **SENT基准测试**：完整的单边缺口拉伸问题（~2500 epochs，约5-10分钟）

## 预期输出

### 终端输出示例

```
======================================================================
  Phase-Field VPINN/DRM Solver - SENT Benchmark Test
======================================================================

[1/5] Creating problem configuration...
  Geometry: L=1.0, H=1.0
  Material: E=210.0, nu=0.3, G_c=0.0027, l=0.02

[2/5] Generating sampling points...
  Domain points: 2000
  Boundary points: 200

[3/5] Initializing neural networks...
  Displacement network: 17346 parameters
  Damage network: 17217 parameters

[4/5] Creating solver...

[5/5] Starting quasi-static loading...
Initializing fields...

============================================================
Loading Step 1/5 | Load = 0.000000
============================================================
  Epoch    0 | Loss: 1.234567e-03 | Energy: 5.678e-04 | BC: 4.567e-04 | Irrev: 2.345e-04
  Epoch  100 | Loss: 8.765432e-04 | Energy: 4.321e-04 | BC: 3.210e-04 | Irrev: 1.234e-04
  ...
Max damage: 0.001234

============================================================
Loading Step 5/5 | Load = 0.010000
============================================================
  ...
Max damage: 0.876543
```

### 生成的图片

**1. sent_result.png**
- 左图：水平位移 u
- 中图：垂直位移 v
- 右图：损伤场 d（显示裂纹路径）

**2. damage_evolution.png**
- 损伤演化曲线：最大损伤 vs 施加位移

## 自定义问题

### 最小示例

```python
from phase_field_vpinn import DisplacementNetwork, DamageNetwork, PhaseFieldSolver
import torch

# 1. 定义问题参数
config = {
    'E': 210.0,       # 杨氏模量
    'nu': 0.3,        # 泊松比
    'G_c': 2.7e-3,    # 断裂能
    'l': 0.02,        # 长度尺度
    'lr_u': 1e-3,
    'lr_d': 1e-3,
}

# 2. 创建采样点
x_domain = torch.rand(1000, 2)  # 域内随机点
x_bc = torch.tensor([[0.0, i/10] for i in range(11)], dtype=torch.float32)

# 3. 定义边界条件函数
def get_bc(load_value, x_bc):
    u_bc = torch.zeros(x_bc.shape[0], 2)
    u_bc[:, 1] = load_value  # 施加垂直位移
    return u_bc

# 4. 创建网络和求解器
u_net = DisplacementNetwork()
d_net = DamageNetwork()
solver = PhaseFieldSolver(config, u_net, d_net)

# 5. 求解
history = solver.solve_quasi_static(
    loading_steps=[0.0, 0.005, 0.01],
    x_domain=x_domain,
    x_bc=x_bc,
    get_bc_func=get_bc,
    n_epochs_per_step=500
)

# 6. 预测
u, d = solver.predict(x_domain)
```

## 参数调优建议

### 快速测试（~1分钟）
```python
n_loading_steps = 3
n_epochs_per_step = 200
n_domain = 500
```

### 标准测试（~5分钟）
```python
n_loading_steps = 5
n_epochs_per_step = 500
n_domain = 2000
```

### 高精度（~30分钟）
```python
n_loading_steps = 10
n_epochs_per_step = 2000
n_domain = 5000
```

## 常见问题

**Q: 损伤场全是0或1？**
A: 调整以下参数：
- 减小加载步长（更多的 loading_steps）
- 增加 weight_irrev（如1000）
- 调整 G_c 和 l 的比例

**Q: 训练很慢？**
A: 
- 减少采样点数（n_domain=500）
- 减少 epochs（n_epochs_per_step=200）
- 使用更小的网络（layers=[2, 32, 32, 2]）

**Q: 损失不收敛？**
A: 
- 降低学习率（lr=1e-4）
- 增加 weight_bc（如1000）
- 增加训练epochs

## 下一步

查看 `PROJECT_README.md` 了解：
- 完整的理论背景
- 详细的代码架构
- 扩展建议
- 参考文献

祝实验顺利！🚀
