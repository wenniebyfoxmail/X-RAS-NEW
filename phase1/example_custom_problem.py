"""
自定义问题示例：中心裂纹板拉伸 (Center Crack Panel)

这个例子展示如何设置一个不同于SENT的问题。
"""

import torch
import numpy as np
from phase_field_vpinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver,
    visualize_solution, generate_domain_points
)


def create_center_crack_problem():
    """
    中心裂纹板拉伸问题
    
    几何：[0,1]×[0,1] 矩形板，中心有水平初始裂纹
    边界条件：上下边界施加拉伸位移
    """
    config = {
        'E': 210.0,
        'nu': 0.3,
        'G_c': 2.7e-3,
        'l': 0.015,  # 稍小的长度尺度
        'L': 1.0,
        'H': 1.0,
        'crack_half_length': 0.2,  # 中心裂纹半长
        'lr_u': 5e-4,  # 较小的学习率
        'lr_d': 5e-4,
    }
    return config


def generate_center_crack_points(config, n_domain=2000, n_bc=200):
    """生成中心裂纹问题的采样点"""
    L = config['L']
    H = config['H']
    a = config['crack_half_length']
    
    # 域内点（避开初始裂纹）
    x_domain = []
    while len(x_domain) < n_domain:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)
        
        # 避开中心裂纹：|x - L/2| < a 且 y ≈ H/2
        if abs(x - L/2) < a and abs(y - H/2) < 0.01:
            continue
        
        x_domain.append([x, y])
    
    x_domain = torch.tensor(x_domain, dtype=torch.float32)
    
    # 边界点：上下边界
    n_bc_half = n_bc // 2
    
    # 下边界 (y=0)
    x_bottom = np.linspace(0, L, n_bc_half)
    y_bottom = np.zeros_like(x_bottom)
    bc_bottom = np.stack([x_bottom, y_bottom], axis=1)
    
    # 上边界 (y=H)
    x_top = np.linspace(0, L, n_bc_half)
    y_top = np.ones_like(x_top) * H
    bc_top = np.stack([x_top, y_top], axis=1)
    
    x_bc = torch.tensor(np.vstack([bc_bottom, bc_top]), dtype=torch.float32)
    
    return x_domain, x_bc


def get_bc_center_crack(config):
    """边界条件：上下对称拉伸"""
    H = config['H']
    
    def get_bc(load_value, x_bc):
        n_bc = x_bc.shape[0]
        u_bc = torch.zeros(n_bc, 2)
        
        # 下边界：向下拉伸
        u_bc[:n_bc//2, 0] = 0.0
        u_bc[:n_bc//2, 1] = -load_value
        
        # 上边界：向上拉伸
        u_bc[n_bc//2:, 0] = 0.0
        u_bc[n_bc//2:, 1] = load_value
        
        return u_bc
    
    return get_bc


def solve_center_crack(n_loading_steps=5, n_epochs_per_step=500):
    """求解中心裂纹问题"""
    print("="*70)
    print("  Center Crack Panel - Custom Problem Example")
    print("="*70)
    
    # 1. 创建问题
    print("\n[1/5] Creating problem configuration...")
    config = create_center_crack_problem()
    print(f"  Center crack half-length: {config['crack_half_length']}")
    
    # 2. 生成采样点
    print("\n[2/5] Generating sampling points...")
    x_domain, x_bc = generate_center_crack_points(config, n_domain=2000, n_bc=200)
    print(f"  Domain points: {x_domain.shape[0]}")
    print(f"  Boundary points: {x_bc.shape[0]}")
    
    # 3. 创建网络
    print("\n[3/5] Initializing neural networks...")
    u_net = DisplacementNetwork(layers=[2, 64, 64, 64, 2])
    d_net = DamageNetwork(layers=[2, 64, 64, 64, 1])
    
    # 4. 创建求解器
    print("\n[4/5] Creating solver...")
    solver = PhaseFieldSolver(config, u_net, d_net)
    
    # 5. 准静态求解
    print("\n[5/5] Starting quasi-static loading...")
    loading_steps = np.linspace(0.0, 0.008, n_loading_steps)
    get_bc = get_bc_center_crack(config)
    
    history = solver.solve_quasi_static(
        loading_steps=loading_steps,
        x_domain=x_domain,
        x_bc=x_bc,
        get_bc_func=get_bc,
        n_epochs_per_step=n_epochs_per_step,
        weight_bc=100.0,
        weight_irrev=100.0
    )
    
    # 6. 可视化
    print("\n" + "="*70)
    print("  Visualizing results...")
    print("="*70)
    
    nx, ny = 100, 100
    x_grid = generate_domain_points(nx, ny, 
                                    x_range=(0, config['L']),
                                    y_range=(0, config['H']))
    
    visualize_solution(solver, x_grid, nx, ny, 
                      save_path='/mnt/user-data/outputs/center_crack_result.png')
    
    print("\n✓ Custom problem solved successfully!")
    print("  Results saved to: center_crack_result.png")
    
    return solver, history


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Running Custom Problem Example")
    print("="*70)
    print("\nThis example demonstrates how to set up a custom problem.")
    print("Problem: Center crack panel under symmetric tension\n")
    
    solver, history = solve_center_crack(
        n_loading_steps=5,
        n_epochs_per_step=500
    )
    
    print("\n" + "="*70)
    print("  Tips for creating your own problem:")
    print("="*70)
    print("  1. Define geometry and material parameters in config dict")
    print("  2. Create sampling point generation function")
    print("  3. Define boundary condition function")
    print("  4. Call solver.solve_quasi_static()")
    print("  5. Visualize results")
    print("\nSee this script as a template!")
