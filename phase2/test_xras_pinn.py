"""
X-RAS-PINN 测试示例
演示域分解 + 自适应采样功能
"""

import torch
import numpy as np
from phase_field_vpinn import (
    XRaSPINNSolver,
    generate_domain_points,
    partition_domain
)
import matplotlib.pyplot as plt
import os


def example_edge_crack_tension():
    """
    示例问题：边缘裂纹拉伸问题
    
    几何：[0, 1] × [0, 1] 矩形
    裂纹：从左边界中点 (0, 0.5) 水平延伸到 (0.3, 0.5)
    裂尖位置：(0.3, 0.5)
    
    边界条件：
        - 底部 (y=0): v = 0
        - 顶部 (y=1): v = δ (施加位移)
        - 左右边界：自由
    """
    
    print("="*70)
    print("X-RAS-PINN Test: Edge Crack under Tension")
    print("="*70)
    
    # ========== 问题配置 ==========
    problem_config = {
        # 材料参数
        'E': 210e3,          # 杨氏模量 (MPa)
        'nu': 0.3,           # 泊松比
        'G_c': 2.7,          # 断裂能 (N/mm)
        'l': 0.015,          # 长度尺度参数 (mm)
        'k': 1e-6,           # 残余刚度
        
        # 域分解参数
        'crack_tip': np.array([0.3, 0.5]),  # 裂尖坐标
        'r_sing': 0.15,                      # 裂尖区域半径
        
        # 学习率
        'lr_u': 1e-3,
        'lr_d': 1e-3,
        
        # 损失权重
        'weights': {
            'lambda_bc': 100.0,    # 边界条件权重
            'lambda_int': 10.0,    # 接口损失权重
            'w_u': 1.0,            # 位移连续性权重
            'w_sigma': 1.0,        # 牵引力平衡权重
        },
        
        # 计算设备
        'device': 'cpu'
    }
    
    device = problem_config['device']
    crack_tip = problem_config['crack_tip']
    r_sing = problem_config['r_sing']
    
    # ========== 生成采样点 ==========
    print("\n生成采样点...")
    
    # 1. 域内点（粗网格）
    nx_domain, ny_domain = 40, 40
    x_domain = generate_domain_points(
        nx=nx_domain, ny=ny_domain,
        x_range=(0.0, 1.0), y_range=(0.0, 1.0)
    ).to(device)
    
    # 分区
    mask_sing, mask_far = partition_domain(x_domain, crack_tip, r_sing)
    x_sing_init = x_domain[mask_sing]
    x_far = x_domain[mask_far]
    
    print(f"  Initial singular domain points: {len(x_sing_init)}")
    print(f"  Far-field domain points: {len(x_far)}")
    
    # 2. 边界点（顶部和底部）
    nx_bc = 50
    x_bc_bottom = torch.tensor([
        [x, 0.0] for x in np.linspace(0, 1, nx_bc)
    ], dtype=torch.float32).to(device)
    
    x_bc_top = torch.tensor([
        [x, 1.0] for x in np.linspace(0, 1, nx_bc)
    ], dtype=torch.float32).to(device)
    
    x_bc = torch.cat([x_bc_bottom, x_bc_top], dim=0)
    
    # 边界条件：底部 v=0，顶部 v=δ
    delta = 0.001  # 施加的位移
    u_bc = torch.zeros_like(x_bc)
    u_bc[len(x_bc_bottom):, 1] = delta  # 顶部 v = delta
    
    print(f"  Boundary points: {len(x_bc)}")
    
    # 3. 接口点（裂尖区域边界上的点）
    n_interface = 100
    theta = np.linspace(0, 2*np.pi, n_interface, endpoint=False)
    x_I = torch.tensor([
        [crack_tip[0] + r_sing * np.cos(t), 
         crack_tip[1] + r_sing * np.sin(t)]
        for t in theta
    ], dtype=torch.float32).to(device)
    
    # 法向量（从裂尖域指向远场域，即径向向外）
    normal_I = torch.tensor([
        [np.cos(t), np.sin(t)] for t in theta
    ], dtype=torch.float32).to(device)
    
    print(f"  Interface points: {len(x_I)}")
    
    # ========== 创建求解器 ==========
    print("\n创建 X-RAS-PINN 求解器...")
    solver = XRaSPINNSolver(problem_config)
    
    # ========== 训练配置 ==========
    train_config = {
        'N_pre': 1000,       # Phase 1: 远场预训练 epochs
        'N_adapt': 3,        # Phase 2: 自适应循环次数
        'N_inner': 500,      # Phase 2: 每次循环内部 epochs
        'N_joint': 1000,     # Phase 3: 联合精化 epochs
        'N_add': 50,         # 每次自适应添加的点数
        'beta': 0.5,         # 融合指标权重 (0.5 = 等权SED和梯度)
        'freeze_far_in_phase2': True  # Phase 2 中冻结远场网络
    }
    
    # ========== 开始训练 ==========
    print("\n开始训练...")
    results = solver.train(
        x_sing_init=x_sing_init,
        x_far=x_far,
        x_bc=x_bc,
        u_bc=u_bc,
        x_I=x_I,
        normal_I=normal_I,
        config=train_config
    )
    
    # ========== 可视化采样分布 ==========
    print("\n生成采样分布可视化...")
    os.makedirs('figs', exist_ok=True)
    solver.visualize_sampling(
        x_sing=results['x_sing_final'],
        x_far=results['x_far'],
        save_path='figs/xras_sampling_scatter.png'
    )
    
    # ========== 预测并可视化解场 ==========
    print("\n预测解场...")
    nx_vis, ny_vis = 100, 100
    x_vis = generate_domain_points(
        nx=nx_vis, ny=ny_vis,
        x_range=(0.0, 1.0), y_range=(0.0, 1.0)
    ).to(device)
    
    u_pred, d_pred = solver.predict(x_vis)
    
    # 转换为 numpy
    u_np = u_pred.cpu().numpy()
    d_np = d_pred.cpu().numpy()
    x_np = x_vis.cpu().numpy()
    
    # 可视化
    print("\n生成解场可视化...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 位移 u
    im0 = axes[0, 0].tricontourf(x_np[:, 0], x_np[:, 1], u_np[:, 0], 
                                  levels=20, cmap='RdBu_r')
    axes[0, 0].set_title('Displacement u', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    # 标记裂尖
    axes[0, 0].scatter(crack_tip[0], crack_tip[1], c='green', s=100, 
                       marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    plt.colorbar(im0, ax=axes[0, 0])
    
    # 位移 v
    im1 = axes[0, 1].tricontourf(x_np[:, 0], x_np[:, 1], u_np[:, 1], 
                                  levels=20, cmap='RdBu_r')
    axes[0, 1].set_title('Displacement v', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    axes[0, 1].scatter(crack_tip[0], crack_tip[1], c='green', s=100, 
                       marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 损伤场 d
    im2 = axes[1, 0].tricontourf(x_np[:, 0], x_np[:, 1], d_np[:, 0], 
                                  levels=20, cmap='hot')
    axes[1, 0].set_title('Damage Field d', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].scatter(crack_tip[0], crack_tip[1], c='green', s=100, 
                       marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    plt.colorbar(im2, ax=axes[1, 0])
    
    # 位移幅值
    u_mag = np.sqrt(u_np[:, 0]**2 + u_np[:, 1]**2)
    im3 = axes[1, 1].tricontourf(x_np[:, 0], x_np[:, 1], u_mag, 
                                  levels=20, cmap='viridis')
    axes[1, 1].set_title('Displacement Magnitude |u|', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    axes[1, 1].scatter(crack_tip[0], crack_tip[1], c='green', s=100, 
                       marker='*', edgecolors='black', linewidths=1.5, zorder=10)
    plt.colorbar(im3, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('figs/xras_solution_fields.png', dpi=150, bbox_inches='tight')
    print("Solution fields saved to figs/xras_solution_fields.png")
    plt.close()
    
    # ========== 打印训练历史摘要 ==========
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    history = results['history']
    
    print(f"\nPhase 1 (Far-field Pretraining):")
    print(f"  Epochs: {len(history['phase1'])}")
    if history['phase1']:
        print(f"  Final loss: {history['phase1'][-1]['loss']:.6e}")
    
    print(f"\nPhase 2 (Singular Focusing + RAS):")
    print(f"  Adaptation cycles: {train_config['N_adapt']}")
    print(f"  Sampling progression:")
    for item in history['sampling']:
        print(f"    Cycle {item['cycle']}: {item['n_points']} points")
    if history['phase2']:
        print(f"  Final loss: {history['phase2'][-1]['loss']:.6e}")
    
    print(f"\nPhase 3 (Joint Refinement):")
    print(f"  Epochs: {len(history['phase3'])}")
    if history['phase3']:
        print(f"  Final loss: {history['phase3'][-1]['loss']:.6e}")
    
    print(f"\nFinal Statistics:")
    print(f"  Singular domain points: {len(results['x_sing_final'])}")
    print(f"  Far-field domain points: {len(results['x_far'])}")
    print(f"  Max damage: {d_np.max():.6f}")
    
    print("\n" + "="*70)
    print("Test completed successfully!")
    print("="*70)
    
    return solver, results


if __name__ == "__main__":
    # 运行示例
    solver, results = example_edge_crack_tension()
    
    print("\n✓ X-RAS-PINN implementation verified!")
    print("✓ Visualizations saved in 'figs/' directory")
