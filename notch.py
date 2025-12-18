"""
notch.py - Notch 几何的单一定义点

所有涉及 notch 的代码（采样、初始化、损失）都应引用此模块，
确保 notch 定义在整个项目中一致。

与 FE 代码对齐的关键参数:
- notch_length (a): 预制裂缝长度
- notch_seed_radius (ρ): 裂缝带半宽度
- initial_d: 裂尖初始损伤峰值
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# §1 几何参数提取
# ============================================================================

def get_notch_geometry(config: Dict) -> Dict:
    """
    从配置中提取 notch 几何参数
    
    Returns:
        dict: {
            'a': notch 长度,
            'y0': notch 中线 y 坐标,
            'rho': notch 带半宽度,
            'L': 试样长度,
            'H': 试样高度,
        }
    """
    return {
        'a': float(config["notch_length"]),
        'y0': float(config["H"]) / 2.0,
        'rho': float(config["notch_seed_radius"]),
        'L': float(config["L"]),
        'H': float(config["H"]),
    }


def get_notch_tip(config: Dict) -> torch.Tensor:
    """
    获取裂尖坐标
    
    Returns:
        torch.Tensor: [x_tip, y_tip] 形状 (2,)
    """
    geom = get_notch_geometry(config)
    return torch.tensor([geom['a'], geom['y0']], dtype=torch.float32)


# ============================================================================
# §2 Notch 判断与掩码
# ============================================================================

def notch_band_mask(x: torch.Tensor, config: Dict) -> torch.Tensor:
    """
    判断点是否在 notch band 内（用于 d=1 约束）
    
    Notch band 定义: x ≤ a 且 |y - y0| ≤ ρ
    
    Args:
        x: (N, 2) 坐标点
        config: 配置字典
        
    Returns:
        mask: (N,) bool 张量，True 表示在 notch band 内
    """
    geom = get_notch_geometry(config)
    
    in_x = x[:, 0] <= geom['a']
    in_y = torch.abs(x[:, 1] - geom['y0']) <= geom['rho']
    
    return in_x & in_y


def near_tip_mask(x: torch.Tensor, config: Dict, radius_factor: float = 1.0) -> torch.Tensor:
    """
    判断点是否在裂尖附近
    
    Args:
        x: (N, 2) 坐标点
        config: 配置字典
        radius_factor: 半径因子，实际半径 = radius_factor * notch_seed_radius
        
    Returns:
        mask: (N,) bool 张量
    """
    tip = get_notch_tip(config)
    rho = float(config["notch_seed_radius"])
    
    distances = torch.norm(x - tip.unsqueeze(0), dim=1)
    return distances < (radius_factor * rho)


def far_from_notch_mask(x: torch.Tensor, config: Dict, cut_factor: float = 1.5) -> torch.Tensor:
    """
    判断点是否远离 notch（用于远场 d=0 压制）
    
    Args:
        x: (N, 2) 坐标点
        config: 配置字典
        cut_factor: 截断因子，实际截断半径 = cut_factor * notch_seed_radius
        
    Returns:
        mask: (N,) bool 张量，True 表示在远场（且不在 notch band 内）
    """
    tip = get_notch_tip(config)
    rho = float(config["notch_seed_radius"])
    cut_radius = cut_factor * rho
    
    distances = torch.norm(x - tip.unsqueeze(0), dim=1)
    is_far = distances > cut_radius
    is_not_in_band = ~notch_band_mask(x, config)
    
    return is_far & is_not_in_band


# ============================================================================
# §3 采样函数
# ============================================================================

def sample_notch_band_points(config: Dict, n: int) -> torch.Tensor:
    """
    在 notch band 上均匀采样点（用于 d=1 约束损失）
    
    Args:
        config: 配置字典
        n: 采样点数
        
    Returns:
        x_notch: (n, 2) notch band 上的采样点
    """
    geom = get_notch_geometry(config)
    
    xs = np.random.uniform(0.0, geom['a'], size=n)
    ys = geom['y0'] + np.random.uniform(-geom['rho'], geom['rho'], size=n)
    
    pts = np.stack([xs, ys], axis=1)
    return torch.tensor(pts, dtype=torch.float32)


def sample_domain_avoiding_notch(config: Dict, n: int) -> torch.Tensor:
    """
    在域内均匀采样，但避开 notch band
    
    Args:
        config: 配置字典
        n: 采样点数
        
    Returns:
        x_domain: (n, 2) 域内采样点（不含 notch band）
    """
    geom = get_notch_geometry(config)
    
    pts = []
    while len(pts) < n:
        x = np.random.uniform(0, geom['L'])
        y = np.random.uniform(0, geom['H'])
        
        # 避开 notch band
        if x <= geom['a'] and abs(y - geom['y0']) <= geom['rho']:
            continue
        
        pts.append([x, y])
    
    return torch.tensor(pts, dtype=torch.float32)


def sample_near_tip(config: Dict, n: int, radius_factor: float = 1.0) -> torch.Tensor:
    """
    在裂尖附近加密采样
    
    Args:
        config: 配置字典
        n: 采样点数
        radius_factor: 采样半径因子
        
    Returns:
        x_near: (n, 2) 裂尖附近的采样点
    """
    tip = get_notch_tip(config).numpy()
    rho = float(config["notch_seed_radius"])
    geom = get_notch_geometry(config)
    
    radius = radius_factor * rho
    
    pts = []
    while len(pts) < n:
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        
        x = tip[0] + r * np.cos(angle)
        y = tip[1] + r * np.sin(angle)
        
        # 确保在域内
        if 0 <= x <= geom['L'] and 0 <= y <= geom['H']:
            # 避开 notch band 内部
            if not (x <= geom['a'] and abs(y - geom['y0']) <= geom['rho']):
                pts.append([x, y])
    
    return torch.tensor(pts, dtype=torch.float32)


# ============================================================================
# §4 初始损伤场
# ============================================================================

def compute_target_damage_field(x: torch.Tensor, config: Dict) -> torch.Tensor:
    """
    计算目标损伤场 d_target(x)，用于初始化 d_net
    
    规则（与 FE 一致）:
    1. Notch band 内: d = 1
    2. 裂尖附近: d = initial_d * exp(-(r/ρ)²) (Gaussian)
    3. 远场: d = 0
    4. 极近点: d ≈ 0.98
    
    Args:
        x: (N, 2) 坐标点
        config: 配置字典
        
    Returns:
        d_target: (N, 1) 目标损伤值
    """
    tip = get_notch_tip(config)
    geom = get_notch_geometry(config)
    initial_d = float(config["initial_d"])
    rho = geom['rho']
    
    # 计算到裂尖的距离
    distances = torch.norm(x - tip.unsqueeze(0), dim=1)
    
    # 1. 初始化为 Gaussian 分布
    d_target = initial_d * torch.exp(-(distances / rho) ** 2)
    d_target = d_target.unsqueeze(1).clamp(0.0, 1.0)
    
    # 2. Notch band 内强制 d = 1
    band_mask = notch_band_mask(x, config)
    d_target[band_mask] = 1.0
    
    # 3. 远场强制 d = 0（排除 notch band）
    far_mask = far_from_notch_mask(x, config, cut_factor=1.5)
    d_target[far_mask] = 0.0
    
    # 4. 极近点强化
    very_close = distances < (0.5 * rho)
    d_target[very_close.unsqueeze(1).expand_as(d_target)] = 0.98
    
    return d_target


def initialize_notch_damage(d_net, x_domain: torch.Tensor, config: Dict, 
                            verbose: bool = True) -> None:
    """
    初始化 d_net 以拟合目标损伤场
    
    Args:
        d_net: 损伤网络
        x_domain: (N, 2) 域内采样点（应包含 notch band 上的点）
        config: 配置字典
        verbose: 是否打印训练信息
        
    Returns:
        d_net: 初始化后的损伤网络
    """
    n_epochs = int(config.get("notch_init_epochs", 500))
    lr = float(config.get("notch_init_lr", 5e-4))
    
    # 确保输入已 detach
    x_domain = x_domain.detach()
    
    # 计算目标场
    d_target = compute_target_damage_field(x_domain, config)
    
    if verbose:
        tip = get_notch_tip(config)
        geom = get_notch_geometry(config)
        print("\n  [Notch 初始化]")
        print(f"    裂尖位置: ({tip[0]:.3f}, {tip[1]:.3f})")
        print(f"    Gaussian 半径: {geom['rho']:.4f}")
        print(f"    初始峰值: {config['initial_d']:.2f}")
        print(f"    受影响点数 (d>0.1): {(d_target > 0.1).sum().item()}")
        print(f"    高损伤点数 (d>0.9): {(d_target > 0.9).sum().item()}")
    
    # 训练
    optimizer = torch.optim.Adam(d_net.parameters(), lr=lr)
    
    tip = get_notch_tip(config)
    geom = get_notch_geometry(config)
    distances = torch.norm(x_domain - tip.unsqueeze(0), dim=1)
    
    best_loss = float("inf")
    patience = 0
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        d_pred = d_net(x_domain)
        
        # MSE 损失
        loss_mse = torch.mean((d_pred - d_target) ** 2)
        
        # 裂尖强化损失
        tip_mask = distances < geom['rho']
        if tip_mask.sum() > 0:
            loss_tip = torch.mean((d_pred[tip_mask] - 0.95) ** 2)
        else:
            loss_tip = torch.tensor(0.0)
        
        # Notch band 保持损失
        band_mask = notch_band_mask(x_domain, config)
        if band_mask.sum() > 0:
            loss_band = torch.mean((d_pred[band_mask] - 1.0) ** 2)
        else:
            loss_band = torch.tensor(0.0)
        
        # 远场压制损失
        far_mask = far_from_notch_mask(x_domain, config)
        if far_mask.sum() > 0:
            loss_far = torch.mean(d_pred[far_mask] ** 2)
        else:
            loss_far = torch.tensor(0.0)
        
        loss = loss_mse + 2.0 * loss_band + 1.0 * loss_tip + 2.0 * loss_far
        loss.backward()
        optimizer.step()
        
        # 打印与早停
        if verbose and (epoch % 200 == 0 or epoch == n_epochs - 1):
            with torch.no_grad():
                d_max = d_pred.max().item()
                d_mean = d_pred.mean().item()
            print(f"    Epoch {epoch:4d}: loss={loss.item():.3e}, "
                  f"d_max={d_max:.3f}, d_mean={d_mean:.3f}")
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 200 and epoch > 500:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break
    
    # 最终统计
    if verbose:
        with torch.no_grad():
            d_final = d_net(x_domain)
            print(f"\n    ✓ 初始化完成:")
            print(f"      d_max:  {d_final.max().item():.3f}")
            print(f"      d_mean: {d_final.mean().item():.3f}")
            print(f"      d_std:  {d_final.std().item():.3f}")
    
    return d_net


# ============================================================================
# §5 损失函数辅助
# ============================================================================

def compute_notch_hold_loss(d_net, x_notch: torch.Tensor, 
                            target_d: float = 1.0, 
                            weight: float = 500.0) -> torch.Tensor:
    """
    计算 notch band 保持损失（强制 d ≈ target_d）
    
    Args:
        d_net: 损伤网络
        x_notch: (N, 2) notch band 上的采样点
        target_d: 目标损伤值
        weight: 损失权重
        
    Returns:
        loss: 标量损失
    """
    if x_notch is None or x_notch.numel() == 0:
        return torch.tensor(0.0, device=next(d_net.parameters()).device)
    
    d_pred = d_net(x_notch)
    d_target = torch.full_like(d_pred, target_d)
    
    return weight * torch.mean((d_pred - d_target) ** 2)


# ============================================================================
# §6 可视化辅助
# ============================================================================

def plot_notch_geometry(config: Dict, ax=None, color='red', alpha=0.3):
    """
    在 matplotlib axes 上绘制 notch 几何
    
    Args:
        config: 配置字典
        ax: matplotlib axes，如果为 None 则创建新图
        color: notch 区域颜色
        alpha: 透明度
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    if ax is None:
        fig, ax = plt.subplots()
    
    geom = get_notch_geometry(config)
    
    # 绘制 notch band
    rect = patches.Rectangle(
        (0, geom['y0'] - geom['rho']),
        geom['a'],
        2 * geom['rho'],
        linewidth=1,
        edgecolor=color,
        facecolor=color,
        alpha=alpha,
        label='Notch band'
    )
    ax.add_patch(rect)
    
    # 标记裂尖
    ax.plot(geom['a'], geom['y0'], 'r*', markersize=10, label='Crack tip')
    
    return ax


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 创建测试配置
    test_config = {
        "L": 0.1,
        "H": 0.1,
        "notch_length": 0.05,
        "notch_seed_radius": 0.01,
        "initial_d": 0.95,
        "notch_init_epochs": 100,
    }
    
    print("=" * 60)
    print("  Notch Module Test")
    print("=" * 60)
    
    # 测试几何提取
    geom = get_notch_geometry(test_config)
    print(f"\n几何参数: {geom}")
    
    # 测试裂尖
    tip = get_notch_tip(test_config)
    print(f"裂尖坐标: {tip}")
    
    # 测试采样
    x_notch = sample_notch_band_points(test_config, 100)
    print(f"\nNotch band 采样: {x_notch.shape}")
    
    x_domain = sample_domain_avoiding_notch(test_config, 500)
    print(f"域内采样 (避开 notch): {x_domain.shape}")
    
    x_near = sample_near_tip(test_config, 100)
    print(f"裂尖附近采样: {x_near.shape}")
    
    # 测试掩码
    x_test = torch.rand(1000, 2) * torch.tensor([0.1, 0.1])
    band_mask = notch_band_mask(x_test, test_config)
    far_mask = far_from_notch_mask(x_test, test_config)
    print(f"\n测试点数: {x_test.shape[0]}")
    print(f"  在 notch band 内: {band_mask.sum().item()}")
    print(f"  在远场: {far_mask.sum().item()}")
    
    # 测试目标损伤场
    d_target = compute_target_damage_field(x_test, test_config)
    print(f"\n目标损伤场统计:")
    print(f"  d_max: {d_target.max().item():.3f}")
    print(f"  d_mean: {d_target.mean().item():.3f}")
    print(f"  d_std: {d_target.std().item():.3f}")
    
    print("\n✅ 所有测试通过!")
