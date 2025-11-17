"""
测试脚本：单边缺口拉伸 (Single Edge Notched Tension, SENT) 基准问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from phase_field_vpinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver,
    generate_domain_points, visualize_solution
)


def create_sent_problem():
    """
    创建单边缺口拉伸(SENT)问题配置
    
    几何：矩形板 [0, L] × [0, H]，左侧中部有初始缺口
    加载：上边界施加位移，下边界固定
    """
    # 几何参数
    L = 1.0  # 长度
    H = 1.0  # 高度
    notch_length = 0.3  # 缺口长度
    
    # 材料参数
    E = 210.0        # 杨氏模量 (GPa)
    nu = 0.3         # 泊松比
    G_c = 8.1e-3     # 断裂能 (kN/mm)
    l = 0.01         # 长度尺度参数
    
    # 训练参数
    config = {
        'E': E,
        'nu': nu,
        'G_c': G_c,
        'l': l,
        'L': L,
        'H': H,
        'notch_length': notch_length,
        'lr_u': 1e-3,
        'lr_d': 1e-3,
        'device': 'cpu'
    }
    
    return config


def generate_sent_sampling_points(config, n_domain=2000, n_bc=200):
    """
    生成 SENT 问题的采样点
    
    Args:
        config: 问题配置
        n_domain: 域内采样点数
        n_bc: 边界采样点数
    
    Returns:
        x_domain: 域内点
        x_bc: 边界点（Dirichlet边界）
    """
    L = config['L']
    H = config['H']
    notch_length = config['notch_length']
    
    # 域内随机采样（避开缺口区域）
    x_domain = []
    while len(x_domain) < n_domain:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)
        
        # 避开缺口：x < notch_length 且 y ≈ H/2
        if x < notch_length and abs(y - H/2) < 0.01:
            continue
        
        x_domain.append([x, y])
    
    x_domain = torch.tensor(x_domain, dtype=torch.float32)
    
    # 边界点
    # 下边界 (y=0): 固定
    x_bottom = np.linspace(0, L, n_bc // 2)
    y_bottom = np.zeros_like(x_bottom)
    bc_bottom = np.stack([x_bottom, y_bottom], axis=1)
    
    # 上边界 (y=H): 施加位移
    x_top = np.linspace(0, L, n_bc // 2)
    y_top = np.ones_like(x_top) * H
    bc_top = np.stack([x_top, y_top], axis=1)
    
    x_bc = torch.tensor(np.vstack([bc_bottom, bc_top]), dtype=torch.float32)
    
    return x_domain, x_bc


def get_bc_function_sent(config):
    """
    返回边界条件函数
    
    边界条件：
    - 下边界 (y=0): u = 0, v = 0 (固定)
    - 上边界 (y=H): u = 0, v = δ (拉伸)
    """
    H = config['H']
    n_bc_half = 100  # 每条边界的点数
    
    def get_bc(load_value, x_bc):
        """
        Args:
            load_value: 上边界位移 δ
            x_bc: (N, 2) 边界点
        Returns:
            u_bc: (N, 2) 边界位移
        """
        n_bc = x_bc.shape[0]
        u_bc = torch.zeros(n_bc, 2)
        
        # 下半部分：固定
        u_bc[:n_bc//2, :] = 0.0
        
        # 上半部分：施加位移
        u_bc[n_bc//2:, 0] = 0.0          # u = 0
        u_bc[n_bc//2:, 1] = load_value   # v = δ
        
        return u_bc
    
    return get_bc


def test_sent_problem(n_loading_steps=5, n_epochs_per_step=500):
    """
    测试 SENT 问题
    
    Args:
        n_loading_steps: 加载步数
        n_epochs_per_step: 每步的训练epoch数
    """
    print("="*70)
    print("  Phase-Field VPINN/DRM Solver - SENT Benchmark Test")
    print("="*70)
    
    # 1. 创建问题
    print("\n[1/5] Creating problem configuration...")
    config = create_sent_problem()
    print(f"  Geometry: L={config['L']}, H={config['H']}")
    print(f"  Material: E={config['E']}, nu={config['nu']}, G_c={config['G_c']}, l={config['l']}")
    
    # 2. 生成采样点
    print("\n[2/5] Generating sampling points...")
    x_domain, x_bc = generate_sent_sampling_points(config, n_domain=2000, n_bc=200)
    print(f"  Domain points: {x_domain.shape[0]}")
    print(f"  Boundary points: {x_bc.shape[0]}")
    
    # 3. 创建神经网络
    print("\n[3/5] Initializing neural networks...")
    u_net = DisplacementNetwork(layers=[2, 64, 64, 64, 2])
    d_net = DamageNetwork(layers=[2, 64, 64, 64, 1])
    print(f"  Displacement network: {sum(p.numel() for p in u_net.parameters())} parameters")
    print(f"  Damage network: {sum(p.numel() for p in d_net.parameters())} parameters")
    
    # 4. 创建求解器
    print("\n[4/5] Creating solver...")
    solver = PhaseFieldSolver(config, u_net, d_net)
    
    # 5. 准静态加载
    print("\n[5/5] Starting quasi-static loading...")
    
    # 加载步：从 0 到最大位移
    max_displacement = 0.01  # 最大拉伸位移
    loading_steps = np.linspace(0.0, max_displacement, n_loading_steps)
    
    # 边界条件函数
    get_bc = get_bc_function_sent(config)
    
    # 求解
    history = solver.solve_quasi_static(
        loading_steps=loading_steps,
        x_domain=x_domain,
        x_bc=x_bc,
        get_bc_func=get_bc,
        n_epochs_per_step=n_epochs_per_step,
        weight_bc=100.0,
        weight_irrev=100.0
    )
    
    # 6. 可视化结果
    print("\n" + "="*70)
    print("  Training completed! Visualizing results...")
    print("="*70)
    
    # 创建规则网格用于可视化
    nx, ny = 100, 100
    x_grid = generate_domain_points(nx, ny, 
                                    x_range=(0, config['L']),
                                    y_range=(0, config['H']))
    
    visualize_solution(solver, x_grid, nx, ny, save_path='sent_result.png')

    # 绘制损伤演化曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    loads = [h['load'] for h in history]
    d_max = [h['d_max'] for h in history]
    ax.plot(loads, d_max, 'o-', linewidth=2, markersize=8)
    ax.set_xlabel('Applied Displacement δ', fontsize=12)
    ax.set_ylabel('Max Damage $d_{max}$', fontsize=12)
    ax.set_title('Damage Evolution', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('damage_evolution.png', dpi=150)
    print("Damage evolution curve saved!")
    plt.show()

    # 打印最终统计
    print("\n" + "="*70)
    print("  Final Statistics:")
    print("="*70)
    print(f"  Final displacement: {loads[-1]:.6f}")
    print(f"  Final max damage: {d_max[-1]:.6f}")
    print(f"  Damage initiation step: {next((i for i, d in enumerate(d_max) if d > 0.1), 'N/A')}")

    return solver, history


def test_simple_convergence():
    """简单的收敛性测试：验证网络能够学习基本的位移场"""
    print("\n" + "="*70)
    print("  Running Simple Convergence Test...")
    print("="*70)

    # 非常简单的配置
    config = {
        'E': 210.0,
        'nu': 0.3,
        'G_c': 2.7e-3,
        'l': 0.02,
        'L': 1.0,
        'H': 1.0,
        'notch_length': 0.3,
        'lr_u': 1e-3,
        'lr_d': 1e-3,
        'device': 'cpu'
    }

    # 生成少量采样点
    x_domain = torch.rand(500, 2)  # 随机域内点
    x_bc_bottom = torch.tensor([[i/50, 0.0] for i in range(51)], dtype=torch.float32)
    x_bc_top = torch.tensor([[i/50, 1.0] for i in range(51)], dtype=torch.float32)
    x_bc = torch.cat([x_bc_bottom, x_bc_top], dim=0)

    # 创建网络
    u_net = DisplacementNetwork(layers=[2, 32, 32, 2])
    d_net = DamageNetwork(layers=[2, 32, 32, 1])

    # 创建求解器
    solver = PhaseFieldSolver(config, u_net, d_net)

    # 单步训练
    def get_bc_simple(load, x_bc):
        n = x_bc.shape[0]
        u_bc = torch.zeros(n, 2)
        u_bc[n//2:, 1] = load  # 上边界拉伸
        return u_bc

    print("\nTraining for 100 epochs with small displacement...")
    solver.initialize_fields(x_domain)
    solver.train_step(x_domain, x_bc, get_bc_simple(0.001, x_bc),
                     n_epochs=100, verbose=False)

    # 检查预测
    u, d = solver.predict(x_bc_top[:10])
    print(f"\nSample predictions (top boundary):")
    print(f"  u_y (should be ~0.001): {u[:, 1].mean().item():.6f}")
    print(f"  d (should be ~0): {d.mean().item():.6f}")

    if abs(u[:, 1].mean().item() - 0.001) < 0.01:
        print("\n✓ Convergence test PASSED!")
    else:
        print("\n✗ Convergence test FAILED - check implementation")


if __name__ == "__main__":
    # 运行简单收敛测试
    test_simple_convergence()

    # 运行完整 SENT 基准测试
    print("\n" * 2)
    solver, history = test_sent_problem(
        n_loading_steps=5,      # 5个加载步（快速测试）
        n_epochs_per_step=500   # 每步500个epochs
    )

    print("\n" + "="*70)
    print("  All tests completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  - /mnt/user-data/outputs/sent_result.png")
    print("  - /mnt/user-data/outputs/damage_evolution.png")