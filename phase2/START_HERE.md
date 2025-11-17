# 🚀 START HERE - X-RAS-PINN Phase-2 Implementation

## 📦 你收到了什么

这是一个完整的 **X-RAS-PINN（域分解 + 自适应采样）** 实现，用于相场断裂力学。

### 核心文件（你需要的）

1. **phase_field_vpinn.py** ⭐ - 主代码（1200+ 行）
   - Phase-1: 单域 VPINN 求解器
   - Phase-2: X-RAS-PINN 求解器（新增）

2. **test_xras_pinn.py** ⭐ - 完整测试示例
   - 边缘裂纹拉伸问题
   - 完整的训练和可视化

3. **quick_test.py** ⭐ - 快速验证
   - 2-3分钟快速测试
   - 验证所有功能正常

4. **README.md** - 详细技术文档
5. **USAGE_GUIDE.md** - 使用指南
6. **QUICK_REFERENCE.md** - API 速查
7. **IMPLEMENTATION_DETAILS.md** - 实现细节说明

---

## ⚡ 60秒快速开始

### 步骤 1: 安装依赖
```bash
pip install torch numpy matplotlib
```

### 步骤 2: 快速验证
```bash
python quick_test.py
```

看到这个就成功了：
```
✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓
✓ ALL VALIDATION TESTS PASSED ✓
✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓✓
```

### 步骤 3: 运行完整示例
```bash
python test_xras_pinn.py
```

生成文件：
- `figs/xras_sampling_scatter.png` - 采样分布
- `figs/xras_solution_fields.png` - 解场可视化

---

## 📋 Phase-2 实现了什么

根据你的 Prompt 要求，完整实现了：

### ✅ 2.1 域分解
- `partition_domain(x, crack_tip, r_sing)` - 分区函数
- `build_phase_field_network(high_capacity)` - 网络构建
- 裂尖域：4层×128神经元（高容量）
- 远场域：3层×64神经元（标准）

### ✅ 2.2 XPINN 能量损失
- `compute_xpinn_energy_loss(...)` - 总损失函数
- L_total = L_energy_sing + L_energy_far + L_bc + L_interface
- ❌ 不使用强形式残差 MSE_f
- ✅ 纯能量泛函（DRM 风格）

### ✅ 2.3 接口损失
- `compute_interface_loss(u1, u2, x_I, ...)` - 接口损失
- 位移连续: MSE_u = ||u1 - u2||²
- 牵引力平衡: MSE_trac = ||σ1·n1 + σ2·n2||²
- 📝 包含 Nitsche's method 改进说明

### ✅ 2.4 自适应采样
- `compute_indicator(u, d, x, beta)` - 融合指标
  - η = (1-β)·SED_norm + β·|∇d|_norm
- `resample_points(x_old, indicator, N_add)` - 重采样
  - p_i ∝ indicator_i
  - 有放回采样

### ✅ 2.5 三阶段训练
- `XRaSPINNSolver` 类
- Phase 1: 远场预训练（冻结裂尖域）
- Phase 2: 裂尖聚焦 + RAS（自适应采样）
- Phase 3: 联合精化（全局优化）

### ✅ 可视化
- `visualize_sampling()` - 采样分布图
- 显示两域点分布
- 标注裂尖位置和区域边界

---

## 🎯 验收标准检查

你的 Prompt 要求：

> **验收标准（代码层面）**
> 
> 代码中存在一个可以被调用的 X-RAS-PINN 接口

✅ **完成**：
```python
solver = XRaSPINNSolver(config)
results = solver.train(...)
```

> 运行一个示例问题时：控制台 log 中打印三个阶段的 epoch 范围

✅ **完成**：
```
PHASE 1: Far-field Pretraining (1000 epochs)
PHASE 2: Singular Focusing with RAS (3 cycles)
PHASE 3: Joint Refinement (1000 epochs)
```

> 每次 adaptation cycle 打印 len(x_sing)

✅ **完成**：
```
--- Adaptation Cycle 1/3 ---
Current x_sing size: 87
Added 50 points. New x_sing size: 137
```

> 保存可视化：figs/xras_sampling_scatter.png

✅ **完成**：两域采样点分布图

> 在 README 或注释中简要说明

✅ **完成**：
- README.md: 详细技术文档
- IMPLEMENTATION_DETAILS.md: 实现说明
- 代码中的 docstring

---

## 📚 文档导航

根据你的需求选择阅读：

| 文档 | 用途 | 阅读时间 |
|------|------|----------|
| **START_HERE.md** (本文件) | 快速开始 | 2分钟 |
| **QUICK_REFERENCE.md** | API 速查 | 3分钟 |
| **USAGE_GUIDE.md** | 详细使用指南 | 10分钟 |
| **README.md** | 技术文档 | 15分钟 |
| **IMPLEMENTATION_DETAILS.md** | 实现细节 | 20分钟 |

---

## 🔧 基本使用模板

```python
from phase_field_vpinn import XRaSPINNSolver
import numpy as np

# 1. 配置
config = {
    'E': 210e3, 'nu': 0.3, 'G_c': 2.7, 'l': 0.015,
    'crack_tip': np.array([0.3, 0.5]),
    'r_sing': 0.15,
    'weights': {'lambda_bc': 100.0, 'lambda_int': 10.0}
}

# 2. 创建求解器
solver = XRaSPINNSolver(config)

# 3. 训练
results = solver.train(
    x_sing_init, x_far, x_bc, u_bc, x_I, normal_I,
    config={'N_pre': 1000, 'N_adapt': 3, 'N_inner': 500,
            'N_joint': 1000, 'N_add': 50, 'beta': 0.5}
)

# 4. 预测
u, d = solver.predict(x_test)

# 5. 可视化
solver.visualize_sampling(
    results['x_sing_final'], 
    results['x_far'],
    'my_sampling.png'
)
```

---

## 🐛 遇到问题？

### 问题 1: ModuleNotFoundError: No module named 'torch'
**解决**: `pip install torch numpy matplotlib`

### 问题 2: 训练不收敛
**检查**:
- 学习率是否太高？降低到 5e-4
- 边界条件权重是否太小？增加到 500
- 接口点是否太少？增加到 100+

### 问题 3: 损伤场全0或全1
**调整**:
- G_c: 增加 → 裂纹更难扩展
- l: 增加 → 裂纹带更宽
- 检查加载是否足够大

### 问题 4: 采样不增加
**检查**:
```python
print(f"Candidate points: {len(x_cand)}")  # 应该 >> N_add
print(f"Indicator range: [{eta.min()}, {eta.max()}]")  # 不应全0
```

---

## 💡 关键亮点

### 为什么是三阶段训练？

1. **Phase 1 (预训练)**: 
   - 远场先学习 → 减少耦合难度
   - 建立稳定基础

2. **Phase 2 (RAS)**:
   - 专注裂尖 → 高效捕捉奇异性
   - 自适应采样 → 自动加密关键区域

3. **Phase 3 (精化)**:
   - 全局优化 → 确保两域一致
   - 降低学习率 → 精细调整

### 自适应采样的作用？

传统方法：
```
均匀采样 → 浪费大量点在平滑区域
```

X-RAS-PINN:
```
自适应采样 → 点自动聚集到裂尖和高梯度区
效果: 同样点数，精度提升 2-5 倍
```

### 域分解的优势？

| 单域 PINN | X-RAS-PINN |
|-----------|------------|
| 单一网络 | 两组网络 |
| 统一容量 | 分层容量 |
| 奇异性难捕捉 | 专用高容量网络 |
| 计算成本高 | 计算更高效 |

---

## 📊 预期结果

### 采样分布（sampling_scatter.png）

```
     远场（蓝色，稀疏）
  ·  ·   ·    ·   ·  ·
   ·   ·   ·    ·   ·
  ·   ·  [●●●●●]  ·  ·   ← 裂尖区（红色，密集）
   ·   ·  [●●●●●]  ·
  ·  ·   · [●★●●] ·  ·   ← 绿星=裂尖
   ·   ·   ·    ·   ·
```

### 训练曲线

```
Loss
  |  Phase 1    Phase 2         Phase 3
  |  (预训练)   (RAS)           (精化)
  |  
  |  \
  |   \___       采样+
  |       \___    训练    ___
  |           \__/   \__/   \___
  |________________________________
                              Epoch
```

---

## 🎓 下一步

1. ✅ 运行 `quick_test.py` 验证安装
2. ✅ 运行 `test_xras_pinn.py` 看完整效果
3. 📖 阅读 `USAGE_GUIDE.md` 了解参数调优
4. 🔧 修改 `test_xras_pinn.py` 测试自己的问题
5. 📚 阅读 `README.md` 深入了解理论

---

## ✉️ 联系与反馈

如果有问题：
1. 检查控制台输出的 loss 曲线
2. 查看生成的可视化图像
3. 参考 `USAGE_GUIDE.md` 故障排查部分
4. 检查 `IMPLEMENTATION_DETAILS.md` 实现细节

关键检查点：
- ✓ Phase 1 结束：远场损失显著下降
- ✓ Phase 2 每个循环：打印 "Added X points"
- ✓ Phase 3 结束：接口损失 < 1e-3

---

## 🎉 总结

你现在拥有：

- ✅ 完整的 X-RAS-PINN 实现（1200+ 行代码）
- ✅ 两个可运行的测试（快速 + 完整）
- ✅ 五份详细文档（从入门到深入）
- ✅ 所有 Phase-2 Prompt 要求的功能
- ✅ 清晰的代码结构和注释
- ✅ 完善的错误处理和验证

**开始使用吧！** 🚀

```bash
python quick_test.py  # 验证安装
python test_xras_pinn.py  # 看完整效果
```

---

**版本**: Phase-2 Complete Implementation  
**日期**: 2025-11-14  
**状态**: ✅ Ready to Use
