"""
简单的梯度计算测试 - 用于验证修复是否成功
"""

import torch
from phase_field_vpinn import DisplacementNetwork, DamageNetwork, compute_strain

print("="*60)
print("  梯度计算测试")
print("="*60)

# 测试1: 位移网络和应变计算
print("\n[测试1] 位移网络梯度追踪")
try:
    u_net = DisplacementNetwork(layers=[2, 32, 32, 2])
    u_net.train()  # 训练模式
    
    x = torch.rand(10, 2)
    x_grad = x.clone().detach().requires_grad_(True)
    
    u = u_net(x_grad)
    print(f"  ✓ 位移输出形状: {u.shape}")
    print(f"  ✓ 位移 requires_grad: {u.requires_grad}")
    print(f"  ✓ 输入 requires_grad: {x_grad.requires_grad}")
    
    # 测试应变计算
    epsilon = compute_strain(u, x_grad)
    print(f"  ✓ 应变输出形状: {epsilon.shape}")
    print(f"  ✓ 应变 requires_grad: {epsilon.requires_grad}")
    
    print("\n  ✓ 测试1 通过!\n")
    
except Exception as e:
    print(f"\n  ✗ 测试1 失败: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试2: 损伤网络
print("[测试2] 损伤网络梯度追踪")
try:
    d_net = DamageNetwork(layers=[2, 32, 32, 1])
    d_net.train()
    
    x = torch.rand(10, 2)
    x_grad = x.clone().detach().requires_grad_(True)
    
    d = d_net(x_grad)
    print(f"  ✓ 损伤输出形状: {d.shape}")
    print(f"  ✓ 损伤范围: [{d.min().item():.4f}, {d.max().item():.4f}]")
    print(f"  ✓ 损伤 requires_grad: {d.requires_grad}")
    
    print("\n  ✓ 测试2 通过!\n")
    
except Exception as e:
    print(f"\n  ✗ 测试2 失败: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试3: 简单的反向传播
print("[测试3] 反向传播测试")
try:
    u_net = DisplacementNetwork(layers=[2, 16, 16, 2])
    u_net.train()
    
    x = torch.rand(5, 2).requires_grad_(True)
    u = u_net(x)
    
    loss = u.sum()
    loss.backward()
    
    print(f"  ✓ 损失值: {loss.item():.6f}")
    print(f"  ✓ 输入梯度形状: {x.grad.shape}")
    print(f"  ✓ 输入梯度范数: {x.grad.norm().item():.6f}")
    
    print("\n  ✓ 测试3 通过!\n")
    
except Exception as e:
    print(f"\n  ✗ 测试3 失败: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 完整流程（模拟initialize_fields）
print("[测试4] 完整初始化流程测试")
try:
    from phase_field_vpinn import compute_energy_split
    
    u_net = DisplacementNetwork()
    u_net.train()
    
    # 模拟 x_domain
    x_domain = torch.rand(100, 2)
    
    # 模拟 initialize_fields 中的操作
    x_domain_temp = x_domain.clone().detach().requires_grad_(True)
    u_temp = u_net(x_domain_temp)
    epsilon = compute_strain(u_temp, x_domain_temp)
    psi_plus, psi_minus = compute_energy_split(epsilon, E=210.0, nu=0.3)
    
    print(f"  ✓ ψ+ 形状: {psi_plus.shape}")
    print(f"  ✓ ψ+ 范围: [{psi_plus.min().item():.6e}, {psi_plus.max().item():.6e}]")
    print(f"  ✓ ψ- 形状: {psi_minus.shape}")
    print(f"  ✓ ψ- 范围: [{psi_minus.min().item():.6e}, {psi_minus.max().item():.6e}]")
    
    H = psi_plus.detach().clone()
    print(f"  ✓ 历史场 H 初始化成功")
    
    print("\n  ✓ 测试4 通过!\n")
    
except Exception as e:
    print(f"\n  ✗ 测试4 失败: {e}\n")
    import traceback
    traceback.print_exc()
    exit(1)

print("="*60)
print("  🎉 所有测试通过!")
print("="*60)
print("\n现在可以运行完整测试:")
print("  python test_phase_field_vpinn.py\n")
