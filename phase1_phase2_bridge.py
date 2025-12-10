"""
phase1_phase2_bridge.py
Phase-1 和 Phase-2 之间的桥接模块

功能：
1. 保存/加载 Phase-1 的模型和历史
2. 从已有损伤场初始化网络
3. 提供场的可视化和分析
"""

import os
import torch
import numpy as np
from pathlib import Path


def save_phase1_checkpoint(solver, history, config, save_path=None):
    """
    保存 Phase-1 的完整检查点
    
    Args:
        solver: PhaseFieldSolver 实例
        history: 训练历史列表
        config: 配置字典
        save_path: 保存路径（默认使用 config 中的路径）
    """
    if save_path is None:
        save_path = config.get("phase1_model_path", "outputs/phase1_checkpoint.pth")
    
    # 确保目录存在
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        # 模型权重
        'u_net_state_dict': solver.u_net.state_dict(),
        'd_net_state_dict': solver.d_net.state_dict(),
        
        # 训练历史
        'history': history,
        
        # 配置（用于验证一致性）
        'config': {
            'G_c': config['G_c'],
            'l': config['l'],
            'E': config['E'],
            'nu': config['nu'],
            'L': config['L'],
            'H': config['H'],
            'notch_length': config['notch_length'],
            'max_displacement': config['max_displacement'],
            'n_loading_steps': config['n_loading_steps'],
        },
        
        # 元信息
        'metadata': {
            'final_d_max': history[-1]['d_max'] if history else None,
            'final_d_mean': history[-1]['d_mean'] if history else None,
            'final_d_std': history[-1]['d_std'] if history else None,
            'final_loc_index': history[-1]['loc_index'] if history else None,
            'n_steps': len(history),
        }
    }
    
    torch.save(checkpoint, save_path)
    
    print(f"\n✓ Phase-1 检查点已保存:")
    print(f"  路径: {save_path}")
    print(f"  步数: {len(history)}")
    if history:
        print(f"  最终状态: d_max={history[-1]['d_max']:.4f}, "
              f"d_mean={history[-1]['d_mean']:.4f}, "
              f"loc_index={history[-1]['loc_index']:.1f}")
    
    return save_path


def load_phase1_checkpoint(checkpoint_path, verbose=True):
    """
    加载 Phase-1 的检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        verbose: 是否打印加载信息
    
    Returns:
        checkpoint: 包含所有保存信息的字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到 Phase-1 检查点: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if verbose:
        print(f"\n✓ Phase-1 检查点已加载:")
        print(f"  路径: {checkpoint_path}")
        print(f"  训练步数: {checkpoint['metadata']['n_steps']}")
        print(f"  最终状态:")
        print(f"    d_max:  {checkpoint['metadata']['final_d_max']:.4f}")
        print(f"    d_mean: {checkpoint['metadata']['final_d_mean']:.4f}")
        print(f"    d_std:  {checkpoint['metadata']['final_d_std']:.4f}")
        print(f"    loc_index: {checkpoint['metadata']['final_loc_index']:.1f}")
    
    return checkpoint


def initialize_network_from_field(network, x_domain, target_field, n_epochs=500, lr=5e-4, verbose=True):
    """
    从给定的损伤场初始化网络
    
    原理：通过 MSE 损失让网络学习已有的场分布
    
    Args:
        network: 待初始化的网络（DamageNetwork 或 DisplacementNetwork）
        x_domain: 域内采样点 [N, 2]
        target_field: 目标场 [N, 1] 或 [N, 2]
        n_epochs: 训练轮数
        lr: 学习率
        verbose: 是否打印训练信息
    
    Returns:
        network: 初始化后的网络
    """
    if verbose:
        print(f"\n  [场初始化] 用 {n_epochs} epochs 拟合目标场...")
        print(f"    目标场统计: mean={target_field.mean().item():.4f}, "
              f"max={target_field.max().item():.4f}, "
              f"std={target_field.std().item():.4f}")
    
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    # 确保数据类型一致
    x_domain = x_domain.detach()
    target_field = target_field.detach()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        pred = network(x_domain)
        loss = torch.mean((pred - target_field) ** 2)
        
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch % 100 == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                pred_final = network(x_domain)
                pred_mean = pred_final.mean().item()
                pred_max = pred_final.max().item()
                pred_std = pred_final.std().item()
            
            print(f"    Epoch {epoch:4d}: loss={loss.item():.3e}, "
                  f"pred: mean={pred_mean:.4f}, max={pred_max:.4f}, std={pred_std:.4f}")
    
    if verbose:
        print(f"  ✓ 场初始化完成")
    
    return network


def get_phase1_field_at_step(checkpoint, step_index=-1):
    """
    获取 Phase-1 在指定步的统计信息
    
    Args:
        checkpoint: 加载的检查点
        step_index: 步数索引 (-1 表示最后一步)
    
    Returns:
        step_info: 该步的统计信息字典
    """
    history = checkpoint['history']
    
    if step_index < 0:
        step_index = len(history) + step_index
    
    if step_index < 0 or step_index >= len(history):
        raise ValueError(f"步数索引 {step_index} 超出范围 [0, {len(history)})")
    
    step_info = history[step_index]
    
    print(f"\n  选择 Phase-1 的第 {step_index+1}/{len(history)} 步:")
    print(f"    载荷:      {step_info['load']:.6f}")
    print(f"    d_max:     {step_info['d_max']:.4f}")
    print(f"    d_mean:    {step_info['d_mean']:.4f}")
    print(f"    d_std:     {step_info['d_std']:.4f}")
    print(f"    loc_index: {step_info['loc_index']:.1f}")
    
    return step_info


def visualize_field_comparison(d_field_1, d_field_2, x_domain, config, save_path=None):
    """
    可视化两个损伤场的对比
    
    Args:
        d_field_1: 第一个场（如 Phase-1 原始）
        d_field_2: 第二个场（如网络预测）
        x_domain: 采样点
        config: 配置
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 转换为 numpy
    x = x_domain[:, 0].detach().cpu().numpy()
    y = x_domain[:, 1].detach().cpu().numpy()
    d1 = d_field_1.detach().cpu().numpy().flatten()
    d2 = d_field_2.detach().cpu().numpy().flatten()
    
    # 场 1
    sc1 = axes[0].scatter(x, y, c=d1, s=10, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("Field 1 (Original)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0])
    
    # 场 2
    sc2 = axes[1].scatter(x, y, c=d2, s=10, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title("Field 2 (Predicted)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[1])
    
    # 差异
    diff = np.abs(d1 - d2)
    sc3 = axes[2].scatter(x, y, c=diff, s=10, cmap='Reds', vmin=0, vmax=0.5)
    axes[2].set_title(f"Absolute Difference (L2={np.sqrt(np.mean(diff**2)):.4f})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 场对比图已保存: {save_path}")
    
    plt.close()


def verify_config_compatibility(phase1_config, phase2_config):
    """
    验证 Phase-1 和 Phase-2 的配置兼容性
    
    Args:
        phase1_config: Phase-1 的配置
        phase2_config: Phase-2 的配置
    
    Returns:
        is_compatible: 是否兼容
        warnings: 警告信息列表
    """
    warnings = []
    is_compatible = True
    
    # 检查关键物理参数
    critical_params = ['G_c', 'l', 'E', 'nu', 'L', 'H', 'notch_length']
    
    for param in critical_params:
        if param in phase1_config and param in phase2_config:
            v1 = phase1_config[param]
            v2 = phase2_config[param]
            
            if abs(v1 - v2) / max(abs(v1), 1e-10) > 0.01:  # 超过 1% 差异
                warnings.append(f"  ⚠️  参数 '{param}' 不一致: "
                              f"Phase-1={v1:.4e}, Phase-2={v2:.4e}")
                is_compatible = False
    
    if warnings:
        print("\n配置兼容性检查:")
        for w in warnings:
            print(w)
        print()
    else:
        print("\n✓ Phase-1 和 Phase-2 配置兼容")
    
    return is_compatible, warnings


# ============================================================================
# 便捷函数：完整的 Phase-1 → Phase-2 工作流
# ============================================================================

def setup_phase2_from_phase1(checkpoint_path, x_domain, config, 
                             u_net_phase2, d_net_phase2, 
                             initialize_u=False, initialize_d=True,
                             verbose=True):
    """
    从 Phase-1 检查点设置 Phase-2 的初始状态
    
    完整工作流：
    1. 加载 Phase-1 检查点
    2. 验证配置兼容性
    3. 获取指定步的模型
    4. 初始化 Phase-2 网络
    
    Args:
        checkpoint_path: Phase-1 检查点路径
        x_domain: Phase-2 的采样点
        config: Phase-2 的配置
        u_net_phase2: Phase-2 的位移网络
        d_net_phase2: Phase-2 的损伤网络
        initialize_u: 是否初始化位移场
        initialize_d: 是否初始化损伤场
        verbose: 是否打印详细信息
    
    Returns:
        load_value: 对应的载荷值
        checkpoint: Phase-1 检查点（供进一步使用）
    """
    if verbose:
        print("\n" + "="*70)
        print("  Phase-1 → Phase-2 桥接设置")
        print("="*70)
    
    # 1. 加载 Phase-1 检查点
    checkpoint = load_phase1_checkpoint(checkpoint_path, verbose=verbose)
    
    # 2. 验证配置
    verify_config_compatibility(checkpoint['config'], config)
    
    # 3. 获取指定步的信息
    step_index = config.get('phase1_load_step', -1)
    step_info = get_phase1_field_at_step(checkpoint, step_index)
    load_value = step_info['load']
    
    # 4. 加载 Phase-1 的网络权重
    from phase_field_vpinn import DisplacementNetwork, DamageNetwork
    
    u_net_phase1 = DisplacementNetwork()
    d_net_phase1 = DamageNetwork()
    
    u_net_phase1.load_state_dict(checkpoint['u_net_state_dict'])
    d_net_phase1.load_state_dict(checkpoint['d_net_state_dict'])
    
    u_net_phase1.eval()
    d_net_phase1.eval()
    
    # 5. 生成目标场
    with torch.no_grad():
        u_field_phase1 = u_net_phase1(x_domain)
        d_field_phase1 = d_net_phase1(x_domain)
    
    if verbose:
        print(f"\n  Phase-1 场统计:")
        print(f"    u: mean={u_field_phase1.mean().item():.4e}, "
              f"max={u_field_phase1.abs().max().item():.4e}")
        print(f"    d: mean={d_field_phase1.mean().item():.4f}, "
              f"max={d_field_phase1.max().item():.4f}, "
              f"std={d_field_phase1.std().item():.4f}")
    
    # 6. 初始化 Phase-2 网络
    field_init_epochs = config.get('field_init_epochs', 500)
    
    if initialize_u:
        if verbose:
            print("\n  [1/2] 初始化位移网络...")
        initialize_network_from_field(
            u_net_phase2, x_domain, u_field_phase1, 
            n_epochs=field_init_epochs, verbose=verbose
        )
    
    if initialize_d:
        if verbose:
            print("\n  [2/2] 初始化损伤网络...")
        initialize_network_from_field(
            d_net_phase2, x_domain, d_field_phase1,
            n_epochs=field_init_epochs, verbose=verbose
        )
    
    if verbose:
        print("\n" + "="*70)
        print("  ✓ Phase-2 初始化完成")
        print("="*70)
    
    return load_value, checkpoint


if __name__ == "__main__":
    print("Phase-1 ↔ Phase-2 桥接模块")
    print("用于在 Phase-2 中复用 Phase-1 的准静态演化结果")
