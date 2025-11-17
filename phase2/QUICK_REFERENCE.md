# X-RAS-PINN 快速参考

## 🎯 核心功能速查

### 1. 域分解
```python
from phase_field_vpinn import partition_domain

mask_sing, mask_far = partition_domain(
    x=collocation_points,
    crack_tip=np.array([0.3, 0.5]),
    r_sing=0.15
)
```

### 2. 创建求解器
```python
from phase_field_vpinn import XRaSPINNSolver

config = {
    'E': 210e3, 'nu': 0.3, 'G_c': 2.7, 'l': 0.015,
    'crack_tip': np.array([0.3, 0.5]), 'r_sing': 0.15,
    'weights': {'lambda_bc': 100.0, 'lambda_int': 10.0}
}

solver = XRaSPINNSolver(config)
```

### 3. 训练
```python
results = solver.train(
    x_sing_init=x_sing,
    x_far=x_far,
    x_bc=x_bc,
    u_bc=u_bc,
    x_I=x_interface,
    normal_I=normals,
    config={
        'N_pre': 2000,
        'N_adapt': 5,
        'N_inner': 1000,
        'N_joint': 2000,
        'N_add': 100,
        'beta': 0.5
    }
)
```

### 4. 预测
```python
u_pred, d_pred = solver.predict(x_test)
```

### 5. 可视化
```python
solver.visualize_sampling(
    x_sing=results['x_sing_final'],
    x_far=results['x_far'],
    save_path='figs/sampling.png'
)
```

## 📊 关键输出

### 训练历史
```python
history = results['history']

# Phase 1
history['phase1']  # list of dicts: {'epoch', 'loss', 'energy_far', 'bc'}

# Phase 2  
history['phase2']  # list of dicts: {'cycle', 'epoch', 'loss', ...}
history['sampling']  # list of dicts: {'cycle', 'n_points'}

# Phase 3
history['phase3']  # list of dicts: {'epoch', 'loss', ...}
```

### 最终采样点
```python
x_sing_final = results['x_sing_final']  # 裂尖域最终点集
x_far = results['x_far']  # 远场点集（固定）
```

## ⚙️ 参数速查表

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **域分解** | | |
| `r_sing` | 0.1-0.2 × 域尺寸 | 裂尖区域半径 |
| **训练阶段** | | |
| `N_pre` | 1000-2000 | Phase 1 epochs |
| `N_adapt` | 3-5 | Phase 2 循环次数 |
| `N_inner` | 500-1000 | Phase 2 每循环 epochs |
| `N_joint` | 1000-2000 | Phase 3 epochs |
| **自适应采样** | | |
| `N_add` | 50-100 | 每次添加点数 |
| `beta` | 0.5 | SED vs 梯度权重 |
| **损失权重** | | |
| `lambda_bc` | 100-1000 | 边界条件 |
| `lambda_int` | 10-50 | 接口损失 |
| `w_u` | 1.0 | 位移连续 |
| `w_sigma` | 1.0 | 牵引力平衡 |

## 🔍 诊断速查

| 现象 | 可能原因 | 解决方案 |
|------|----------|----------|
| Loss 不下降 | 学习率太高 | 降低 lr_u, lr_d |
| 损伤全0 | G_c 太大 | 降低 G_c 或增加加载 |
| 损伤全1 | G_c 太小 | 增加 G_c |
| 采样不增加 | 指标计算错误 | 检查 x_cand 数量 |
| 接口不连续 | lambda_int 太小 | 增加 lambda_int |

## 📁 文件清单

1. ✅ **phase_field_vpinn.py** - 主代码（Phase-1 + Phase-2）
2. ✅ **test_xras_pinn.py** - 完整示例测试
3. ✅ **quick_test.py** - 快速验证测试
4. ✅ **README.md** - 详细技术文档
5. ✅ **USAGE_GUIDE.md** - 使用指南

## 🚀 快速测试命令

```bash
# 安装依赖
pip install torch numpy matplotlib

# 快速验证（2-3分钟）
python quick_test.py

# 完整示例（10-15分钟）
python test_xras_pinn.py
```

## 📈 期望结果

### 控制台输出
- ✓ Phase 1: 打印 "Far-field Pretraining"
- ✓ Phase 2: 打印 "Adaptation Cycle X/Y"
- ✓ Phase 2: 打印 "Added N points. New x_sing size: X"
- ✓ Phase 3: 打印 "Joint Refinement"
- ✓ 最终: 打印 "Training completed!"

### 生成文件
- ✓ `figs/xras_sampling_scatter.png` - 采样分布图
- ✓ `figs/xras_solution_fields.png` - 解场可视化（仅完整测试）

## 🎓 算法概览

```
X-RAS-PINN 工作流程：

1. 域分解
   Ω = Ω_sing ∪ Ω_far
   ↓
2. Phase 1: 预训练远场
   固定 u_sing, d_sing
   优化 u_far, d_far
   ↓
3. Phase 2: 裂尖聚焦 + RAS
   for k in range(N_adapt):
       训练 N_inner epochs
       计算指标 η = (1-β)·SED + β·|∇d|
       重采样添加 N_add 点
   ↓
4. Phase 3: 联合精化
   解冻所有网络
   降低学习率
   联合优化
   ↓
5. 输出结果
   u(x), d(x)
   采样分布可视化
```

## 💡 核心创新点

1. **域分解**: 高容量网络专注奇异性，标准网络处理平滑区
2. **接口耦合**: 位移连续 + 牵引力平衡
3. **自适应采样**: 物理指标引导，自动加密关键区域
4. **三阶段训练**: 从粗到精，从局部到全局

---

**版本**: Phase-2 Complete
**日期**: 2025-11-14
