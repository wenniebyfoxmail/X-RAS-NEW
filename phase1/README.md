# Phase-Field VPINN/DRM 断裂力学求解器

> 基于变分物理信息神经网络(VPINN)和深度Ritz方法(DRM)的相场断裂力学求解器
> 
> **完整实现 PyTorch + 自动微分 + 准静态加载算法**

---

## 🚀 快速开始 (3步)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
python test_phase_field_vpinn.py

# 3. 查看结果
# 生成的图片: sent_result.png, damage_evolution.png
```

**运行时间**: ~5-10分钟（CPU）

---

## 📚 文档导航

根据你的需求选择：

| 目标 | 文档 | 描述 |
|-----|------|------|
| 🎯 **快速上手** | [QUICKSTART.md](QUICKSTART.md) | 安装、运行、基础使用 |
| 📖 **理论学习** | [PROJECT_README.md](PROJECT_README.md) | 完整的数学理论和算法细节 |
| 📋 **项目总览** | [OVERVIEW.md](OVERVIEW.md) | 实现清单、性能指标、扩展方向 |
| 💻 **自定义问题** | [example_custom_problem.py](example_custom_problem.py) | 自定义问题模板代码 |

---

## 📦 核心文件

```
phase_field_vpinn.py          # 主求解器 (18KB)
├── §1 神经网络定义
├── §2 自动微分模块
├── §3 DRM损失函数
├── §4 准静态求解器
└── §5 辅助函数

test_phase_field_vpinn.py     # SENT基准测试 (8.7KB)
example_custom_problem.py     # 自定义问题示例 (5.1KB)
run_all_tests.py              # 一键测试脚本 (3.9KB)
```

---

## 🔬 实现的功能

### ✅ 核心算法
- [x] 双MLP网络架构（位移 + 损伤）
- [x] PyTorch自动微分计算应变和梯度
- [x] Amor能量分解（拉伸/压缩分离）
- [x] 历史场H的计算和固定
- [x] 不可逆性约束
- [x] 准静态加载双循环算法

### ✅ 测试验证
- [x] 简单收敛测试
- [x] SENT基准问题
- [x] 中心裂纹问题示例
- [x] 可视化输出

---

## 🎯 使用示例

### 最小代码示例

```python
from phase_field_vpinn import DisplacementNetwork, DamageNetwork, PhaseFieldSolver
import torch

# 配置
config = {
    'E': 210.0, 'nu': 0.3, 'G_c': 2.7e-3, 'l': 0.02,
    'lr_u': 1e-3, 'lr_d': 1e-3
}

# 网络
u_net = DisplacementNetwork()
d_net = DamageNetwork()

# 求解器
solver = PhaseFieldSolver(config, u_net, d_net)

# 求解
history = solver.solve_quasi_static(
    loading_steps=[0.0, 0.005, 0.01],
    x_domain=domain_points,
    x_bc=boundary_points,
    get_bc_func=bc_function,
    n_epochs_per_step=1000
)

# 预测
u, d = solver.predict(test_points)
```

完整示例见 `test_phase_field_vpinn.py`

---

## 📊 基准测试结果

**SENT问题** (单边缺口拉伸):
- 几何: 1×1 矩形板，左侧缺口0.3
- 材料: E=210 GPa, ν=0.3, Gₓ=2.7e-3 kN/mm
- 结果: Mode-I裂纹从缺口水平传播 ✓

**网络规模**:
- 位移网络: ~17K参数
- 损伤网络: ~17K参数

**训练时间** (CPU, 5步×500epochs):
- 总时间: ~5-10分钟
- 每步: ~1-2分钟

---

## 🛠️ 进阶使用

### 方案1: 快速测试 (~1分钟)
```bash
python run_all_tests.py --quick
```

### 方案2: 标准测试 (~5分钟)
```bash
python run_all_tests.py --standard
# 或直接: python test_phase_field_vpinn.py
```

### 方案3: 完整测试 (~15分钟)
```bash
python run_all_tests.py --full
# 运行SENT + 中心裂纹两个问题
```

### 方案4: 自定义问题
```bash
# 1. 复制模板
cp example_custom_problem.py my_problem.py

# 2. 修改几何和边界条件
# 编辑 my_problem.py

# 3. 运行
python my_problem.py
```

---

## 📈 参数调优指南

| 参数 | 快速 | 标准 | 高精度 |
|-----|-----|------|-------|
| loading_steps | 3 | 5 | 10 |
| epochs_per_step | 200 | 500 | 2000 |
| n_domain | 500 | 2000 | 5000 |
| 预计时间 | 1分钟 | 5分钟 | 30分钟 |

详细调优见 `PROJECT_README.md` § 参数调优

---

## 🔧 故障排除

### Q: 损失不收敛？
```python
# 降低学习率
config['lr_u'] = 1e-4
config['lr_d'] = 1e-4

# 增加边界条件权重
weight_bc = 1000
```

### Q: 损伤场不合理？
```python
# 增加不可逆性权重
weight_irrev = 1000

# 减小加载步长
loading_steps = np.linspace(0, 0.01, 10)  # 更多步
```

### Q: 训练太慢？
```python
# 使用GPU
config['device'] = 'cuda'

# 减少采样点
n_domain = 500
```

完整故障排除见 `PROJECT_README.md` § 故障排除

---

## 🌟 核心特性

### 1. 严格的物理模型
- 基于Amor能量分解（仅拉伸能驱动损伤）
- 历史场保证加载路径依赖性
- 不可逆性约束确保裂纹单向增长

### 2. 模块化设计
- 独立的网络、损失、求解器模块
- 清晰的函数接口
- 易于扩展和修改

### 3. PyTorch原生
- 充分利用自动微分
- 支持GPU加速
- 无额外依赖（仅torch, numpy, matplotlib）

### 4. 教学友好
- 变量命名与数学符号对应
- 详细的代码注释
- 完整的理论文档

---

## 📚 理论背景

### 相场模型

裂纹用连续损伤场 `d(x) ∈ [0,1]` 表征：

```
E[u,d] = ∫[g(d)H + ψ⁻(ε) + (Gₓ/c₀)(w(d)/l + l|∇d|²)]dΩ
```

其中:
- `g(d) = (1-d)² + k`: 退化函数
- `H = max_t ψ⁺(ε(t))`: 历史场
- `ψ⁺, ψ⁻`: 应变能分解
- `Gₓ`: 断裂能, `l`: 长度尺度

详细推导见 `PROJECT_README.md`

---

## 🔬 参考文献

1. **Miehe et al. (2010)** - Phase field model for rate-independent crack propagation
2. **Amor et al. (2009)** - Regularized formulation of the variational brittle fracture
3. **Raissi et al. (2019)** - Physics-informed neural networks
4. **E & Yu (2018)** - The deep Ritz method

---

## 📄 许可证

本代码仅供学习和研究使用。

---

## 👤 作者

Claude AI Assistant

---

## 🎓 引用

如果此代码对你的研究有帮助，请引用：

```bibtex
@software{phase_field_vpinn_2024,
  title={Phase-Field VPINN/DRM Solver for Fracture Mechanics},
  author={Claude AI Assistant},
  year={2024},
  note={PyTorch implementation}
}
```

---

## 🙏 致谢

感谢相场断裂力学和物理信息神经网络领域的前人工作。

---

**开始使用**: `python test_phase_field_vpinn.py` 🚀

**遇到问题?** 查看 [QUICKSTART.md](QUICKSTART.md) 或 [PROJECT_README.md](PROJECT_README.md)

**贡献欢迎!** 这是一个教学实现，欢迎改进和扩展。
