"""
X-RAS-PINN 快速验证测试
使用较小参数快速验证实现正确性
"""

import torch
import numpy as np
from phase_field_vpinn import (
    XRaSPINNSolver,
    generate_domain_points,
    partition_domain
)
import os


def quick_validation_test():
    """快速验证测试（使用小参数）"""
    
    print("="*70)
    print("X-RAS-PINN Quick Validation Test")
    print("="*70)
    
    # 问题配置
    problem_config = {
        'E': 210e3,
        'nu': 0.3,
        'G_c': 2.7,
        'l': 0.015,
        'k': 1e-6,
        'crack_tip': np.array([0.3, 0.5]),
        'r_sing': 0.15,
        'lr_u': 1e-3,
        'lr_d': 1e-3,
        'weights': {
            'lambda_bc': 100.0,
            'lambda_int': 10.0,
            'w_u': 1.0,
            'w_sigma': 1.0,
        },
        'device': 'cpu'
    }
    
    device = problem_config['device']
    crack_tip = problem_config['crack_tip']
    r_sing = problem_config['r_sing']
    
    # 生成采样点（较少的点）
    print("\n1. 生成采样点...")
    x_domain = generate_domain_points(nx=20, ny=20, 
                                      x_range=(0.0, 1.0), 
                                      y_range=(0.0, 1.0)).to(device)
    
    mask_sing, mask_far = partition_domain(x_domain, crack_tip, r_sing)
    x_sing_init = x_domain[mask_sing]
    x_far = x_domain[mask_far]
    
    print(f"   ✓ Singular domain: {len(x_sing_init)} points")
    print(f"   ✓ Far-field domain: {len(x_far)} points")
    
    # 边界点
    x_bc_bottom = torch.tensor([[x, 0.0] for x in np.linspace(0, 1, 20)], 
                               dtype=torch.float32).to(device)
    x_bc_top = torch.tensor([[x, 1.0] for x in np.linspace(0, 1, 20)], 
                            dtype=torch.float32).to(device)
    x_bc = torch.cat([x_bc_bottom, x_bc_top], dim=0)
    
    delta = 0.001
    u_bc = torch.zeros_like(x_bc)
    u_bc[len(x_bc_bottom):, 1] = delta
    
    print(f"   ✓ Boundary points: {len(x_bc)}")
    
    # 接口点
    n_interface = 40
    theta = np.linspace(0, 2*np.pi, n_interface, endpoint=False)
    x_I = torch.tensor([[crack_tip[0] + r_sing * np.cos(t), 
                        crack_tip[1] + r_sing * np.sin(t)] 
                       for t in theta], dtype=torch.float32).to(device)
    normal_I = torch.tensor([[np.cos(t), np.sin(t)] 
                            for t in theta], dtype=torch.float32).to(device)
    
    print(f"   ✓ Interface points: {len(x_I)}")
    
    # 创建求解器
    print("\n2. 创建 X-RAS-PINN 求解器...")
    solver = XRaSPINNSolver(problem_config)
    print("   ✓ Solver initialized")
    
    # 训练配置（快速测试）
    train_config = {
        'N_pre': 100,      # Phase 1
        'N_adapt': 2,       # Phase 2
        'N_inner': 50,      # Phase 2
        'N_joint': 100,     # Phase 3
        'N_add': 20,        # 每次添加点数
        'beta': 0.5,
        'freeze_far_in_phase2': True
    }
    
    # 训练
    print("\n3. 开始训练...")
    results = solver.train(
        x_sing_init=x_sing_init,
        x_far=x_far,
        x_bc=x_bc,
        u_bc=u_bc,
        x_I=x_I,
        normal_I=normal_I,
        config=train_config
    )
    print("   ✓ Training completed")
    
    # 验证结果
    print("\n4. 验证结果...")
    
    # 检查历史记录
    history = results['history']
    assert 'phase1' in history and len(history['phase1']) > 0, "Phase 1 history missing"
    assert 'phase2' in history and len(history['phase2']) > 0, "Phase 2 history missing"
    assert 'phase3' in history and len(history['phase3']) > 0, "Phase 3 history missing"
    assert 'sampling' in history and len(history['sampling']) > 0, "Sampling history missing"
    print("   ✓ History records validated")
    
    # 检查采样点增加
    initial_points = len(x_sing_init)
    final_points = len(results['x_sing_final'])
    assert final_points > initial_points, "Points should increase with adaptive sampling"
    print(f"   ✓ Adaptive sampling working: {initial_points} → {final_points} points")
    
    # 检查预测功能
    x_test = generate_domain_points(nx=10, ny=10, 
                                    x_range=(0.0, 1.0), 
                                    y_range=(0.0, 1.0)).to(device)
    u_pred, d_pred = solver.predict(x_test)
    assert u_pred.shape == (100, 2), "Displacement prediction shape incorrect"
    assert d_pred.shape == (100, 1), "Damage prediction shape incorrect"
    assert torch.all(d_pred >= 0) and torch.all(d_pred <= 1), "Damage values out of range"
    print("   ✓ Prediction function validated")
    
    # 生成可视化
    print("\n5. 生成可视化...")
    os.makedirs('figs', exist_ok=True)
    solver.visualize_sampling(
        x_sing=results['x_sing_final'],
        x_far=results['x_far'],
        save_path='figs/xras_sampling_scatter_test.png'
    )
    print("   ✓ Sampling visualization saved")
    
    # 打印摘要
    print("\n" + "="*70)
    print("Validation Summary")
    print("="*70)
    print(f"Phase 1 epochs: {len(history['phase1'])}")
    print(f"Phase 2 cycles: {train_config['N_adapt']}")
    print(f"Phase 3 epochs: {len(history['phase3'])}")
    print(f"Sampling progression:")
    for item in history['sampling']:
        print(f"  Cycle {item['cycle']}: {item['n_points']} points")
    print(f"Final statistics:")
    print(f"  Singular points: {final_points}")
    print(f"  Max damage: {d_pred.max().item():.6f}")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        success = quick_validation_test()
        print("\n" + "✓"*35)
        print("✓ ALL VALIDATION TESTS PASSED ✓")
        print("✓"*35)
        print("\n✓ X-RAS-PINN implementation verified!")
        print("✓ Ready for full-scale testing with test_xras_pinn.py")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
