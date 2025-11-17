"""
test_sent_phase2_unified.py

SENT + Notch 场景下的完整对比实验：
单域 VPINN (baseline) vs X-RAS-PINN (域分解 + 自适应采样)

改进点:
1. ✅ 使用统一配置 (config_sent.py)
2. ✅ DEBUG/FULL模式切换
3. ✅ 相同物理参数、采样策略、notch初始化
4. ✅ 论文级可视化输出
5. ✅ 量化对比指标
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# §1 路径与导入
# ============================================================================

def get_output_dir():
    """统一输出目录"""
    out_dir = Path(os.getcwd()) / "outputs"
    out_dir.mkdir(exist_ok=True)
    return out_dir

# 确保可以导入本地模块
sys.path.insert(0, str(Path(__file__).parent))

# 导入核心模块
from phase_field_vpinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver,
    generate_domain_points
)

from xras_pinn_solver import (
    XRASPINNSolver, SubdomainModels
)

from config import create_config, print_config

# ============================================================================
# §2 统一的采样点生成（与Phase-1一致）
# ============================================================================

def generate_sent_with_notch_points(config):
    """
    生成SENT采样点（与Phase-1的test_sent_fixed.py完全一致）

    策略:
    - 70% 全域均匀（避开notch凹槽）
    - 30% notch尖端附近加密
    """
    L = config["L"]
    H = config["H"]
    notch_length = config["notch_length"]
    n_domain = config["n_domain"]
    n_bc = config["n_bc"]

    notch_tip = np.array([notch_length, H / 2.0])

    # 70% 均匀 + 30% 集中
    n_uniform = int(n_domain * 0.7)
    n_concentrated = n_domain - n_uniform

    x_domain_list = []

    # 1) 均匀采样（略避开notch凹槽）
    while len(x_domain_list) < n_uniform:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)

        # 避开notch凹槽区域
        if x < notch_length and abs(y - H / 2) < 0.02:
            continue

        x_domain_list.append([x, y])

    # 2) notch尖端附近加密
    radius_local = 0.02
    for _ in range(n_concentrated):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius_local)

        x = notch_tip[0] + r * np.cos(angle)
        y = notch_tip[1] + r * np.sin(angle)

        if 0 <= x <= L and 0 <= y <= H:
            if not (x < notch_length and abs(y - H / 2) < 0.02):
                x_domain_list.append([x, y])

    x_domain = torch.tensor(x_domain_list, dtype=torch.float32)

    # 边界点（上下边）
    n_bc_half = n_bc // 2
    x_bottom = np.linspace(0, L, n_bc_half)
    y_bottom = np.zeros_like(x_bottom)
    bc_bottom = np.stack([x_bottom, y_bottom], axis=1)

    x_top = np.linspace(0, L, n_bc_half)
    y_top = np.ones_like(x_top) * H
    bc_top = np.stack([x_top, y_top], axis=1)

    x_bc = torch.tensor(np.vstack([bc_bottom, bc_top]), dtype=torch.float32)

    return x_domain, x_bc


def initialize_notch_damage(d_net, x_domain, config):
    """
    在notch尖端初始化损伤种子（与Phase-1一致）

    策略:
    - 高斯型目标场: d_target = initial_d * exp(-(r/radius)^2)
    - 极近点强制高损伤（d > 0.9）
    - 远场强制为0
    """
    notch_length = config["notch_length"]
    H = config["H"]
    initial_d = config["initial_d"]
    seed_radius = config["notch_seed_radius"]
    n_epochs = config["notch_init_epochs"]

    # ✅ 关键修正：确保x_domain已经detach，避免重复backward
    x_domain = x_domain.detach()

    notch_tip = torch.tensor([notch_length, H / 2.0])

    # 距离场
    with torch.no_grad():
        distances = torch.norm(x_domain - notch_tip, dim=1)

        # 高斯目标场
        d_target = initial_d * torch.exp(-(distances / seed_radius) ** 2)
        d_target = d_target.unsqueeze(1)
        d_target = torch.clamp(d_target, 0.0, 1.0)

        # 远场强制为0（防止高斯尾巴）
        cut_radius = 1.5 * seed_radius
        far_mask = distances > cut_radius
        d_target[far_mask] = 0.0

        # 极近点强制高损伤
        very_close = distances < (0.5 * seed_radius)
        d_target[very_close] = 0.98

    print("\n  [Notch初始化]")
    print(f"    尖端位置: ({notch_length:.2f}, {H/2:.2f})")
    print(f"    高斯半径: {seed_radius:.3f}, 峰值: {initial_d:.2f}")
    print(f"    受影响点数(d>0.1): {(d_target > 0.1).sum().item()}")
    print(f"    极近点数(d>0.9):   {(d_target > 0.9).sum().item()}")

    # 训练d_net拟合目标
    optimizer = torch.optim.Adam(d_net.parameters(), lr=5e-4)

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        d_pred = d_net(x_domain)

        loss_mse = torch.mean((d_pred - d_target) ** 2)

        # 尖端强化损失
        tip_points = distances < seed_radius
        if tip_points.sum() > 0:
            loss_tip = torch.mean((d_pred[tip_points] - 0.95) ** 2)
        else:
            loss_tip = torch.tensor(0.0)

        loss = loss_mse + 0.5 * loss_tip
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                d_max = d_pred.max().item()
                d_mean = d_pred.mean().item()
            print(f"    Epoch {epoch:4d}: loss={loss.item():.3e}, "
                  f"d_max={d_max:.3f}, d_mean={d_mean:.3f}")

    return d_net


def get_bc_function_sent(config):
    """生成边界条件函数（上拉伸，下固定）"""
    def get_bc(load_value, x_bc):
        n_bc = x_bc.shape[0]
        u_bc = torch.zeros(n_bc, 2)

        # 底边固定
        u_bc[:n_bc // 2, :] = 0.0

        # 顶边拉伸
        u_bc[n_bc // 2:, 0] = 0.0
        u_bc[n_bc // 2:, 1] = load_value

        return u_bc

    return get_bc


# ============================================================================
# §3 训练函数：VPINN baseline
# ============================================================================

def run_vpinn_baseline(config, x_domain, x_bc, u_bc):
    """
    训练单域VPINN作为baseline

    Returns:
        solver: 训练好的求解器
        time_elapsed: 训练时间（秒）
    """
    print("\n" + "="*70)
    print("  [VPINN Baseline] 开始训练")
    print("="*70)

    # 1. 初始化网络
    u_net = DisplacementNetwork()
    d_net = DamageNetwork()

    # 2. Notch初始化
    print("\n  [1/3] Notch损伤种子初始化...")
    d_net = initialize_notch_damage(d_net, x_domain, config)

    # 3. 构建求解器
    solver = PhaseFieldSolver(config, u_net, d_net)

    # 4. 训练
    print(f"\n  [2/3] 训练VPINN ({config['n_epochs_vpinn']} epochs)...")
    t0 = time.time()

    solver.train_step(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=config["n_epochs_vpinn"],
        weight_bc=config["weight_bc"],
        weight_irrev=config["weight_irrev"],
        verbose=False  # 减少日志输出
    )

    t1 = time.time()
    time_elapsed = t1 - t0

    # 5. 统计
    with torch.no_grad():
        d_vals = d_net(x_domain)
        d_mean = d_vals.mean().item()
        d_max = d_vals.max().item()
        d_std = d_vals.std().item()

    print(f"\n  [3/3] VPINN训练完成!")
    print(f"    时间: {time_elapsed:.2f}s")
    print(f"    损伤统计: d_max={d_max:.4f}, d_mean={d_mean:.4f}, d_std={d_std:.4f}")
    print("="*70)

    return solver, time_elapsed


# ============================================================================
# §4 训练函数：X-RAS-PINN
# ============================================================================

def run_xras_solver(config, x_domain, x_bc, u_bc):
    """
    训练X-RAS-PINN（域分解 + 自适应采样）

    Returns:
        solver: 训练好的求解器
        x_domain_new: 自适应采样后的点集
        time_elapsed: 训练时间（秒）
    """
    print("\n" + "="*70)
    print("  [X-RAS-PINN] 开始训练")
    print("="*70)

    # 1. 初始化子域网络
    u_net_1 = DisplacementNetwork()
    d_net_1 = DamageNetwork()
    u_net_2 = DisplacementNetwork()
    d_net_2 = DamageNetwork()

    # 2. 只对Ω_sing的d_net_1做notch初始化
    print("\n  [1/4] Notch初始化（仅Ω_sing网络）...")
    d_net_1 = initialize_notch_damage(d_net_1, x_domain, config)

    models = SubdomainModels(
        u_net_1=u_net_1,
        d_net_1=d_net_1,
        u_net_2=u_net_2,
        d_net_2=d_net_2,
    )

    # 3. 构建X-RAS求解器
    crack_center = torch.tensor(config["notch_tip"])
    r_sing = config["r_sing"]

    solver = XRASPINNSolver(config, models, crack_center, r_sing=r_sing)

    # 4. 三阶段训练
    t0 = time.time()

    # Phase 1: 预训练
    print(f"\n  [2/4] Phase 1: 预训练 ({config['n_epochs_phase1']} epochs)...")
    solver.train_phase1_pretrain(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=config["n_epochs_phase1"],
        weight_bc=config["weight_bc"],
        verbose=False
    )

    # Phase 2: 聚焦 + 自适应采样
    print(f"\n  [3/4] Phase 2: 聚焦训练 + RAS ({config['n_epochs_phase2']} epochs)...")
    x_domain_new = solver.train_phase2_focused(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=config["n_epochs_phase2"],
        weight_bc=config["weight_bc"],
        weight_interface=config["weight_interface"],
        N_add=config["N_add_ras"],
        verbose=False
    )

    # Phase 3: 联合微调
    print(f"\n  [4/4] Phase 3: 联合微调 ({config['n_epochs_phase3']} epochs)...")
    solver.train_phase3_joint_finetune(
        x_domain_new,
        x_bc,
        u_bc,
        n_epochs=config["n_epochs_phase3"],
        n_interface=config["n_interface"],
        weight_bc=config["weight_bc"],
        weight_interface=config["weight_interface"],
        verbose=False
    )

    t1 = time.time()
    time_elapsed = t1 - t0

    # 5. 统计
    with torch.no_grad():
        _, d_vals = solver.predict(x_domain_new)
        d_mean = d_vals.mean().item()
        d_max = d_vals.max().item()
        d_std = d_vals.std().item()

    print(f"\n  X-RAS训练完成!")
    print(f"    时间: {time_elapsed:.2f}s")
    print(f"    采样点: {x_domain.shape[0]} → {x_domain_new.shape[0]}")
    print(f"    损伤统计: d_max={d_max:.4f}, d_mean={d_mean:.4f}, d_std={d_std:.4f}")
    print("="*70)

    return solver, x_domain_new, time_elapsed


# ============================================================================
# §5 论文级可视化与对比
# ============================================================================

def visualize_comparison(
    config,
    vpinn_solver,
    xras_solver,
    x_domain_vpinn,
    x_domain_xras,
    time_vpinn,
    time_xras,
    output_dir
):
    """
    生成论文级对比图像

    包含:
    1. Damage场对比（VPINN, X-RAS, 差异）
    2. 采样点分布对比
    3. 训练时间对比
    """
    print("\n  [可视化] 生成对比图像...")

    # 1. 生成统一评估网格
    nx, ny = 150, 150
    x_grid = generate_domain_points(
        nx, ny,
        x_range=(0.0, config["L"]),
        y_range=(0.0, config["H"])
    )

    X = x_grid[:, 0].cpu().numpy().reshape(nx, ny)
    Y = x_grid[:, 1].cpu().numpy().reshape(nx, ny)

    # 2. 评估两个求解器的damage场
    _, d_vpinn = vpinn_solver.predict(x_grid)
    d_vpinn = d_vpinn.cpu().numpy().reshape(nx, ny)

    _, d_xras = xras_solver.predict(x_grid)
    d_xras = d_xras.cpu().numpy().reshape(nx, ny)

    # 3. 计算差异
    diff = np.abs(d_vpinn - d_xras)

    # 4. 计算裂尖附近的局部L2误差
    x_flat = x_grid[:, 0].cpu().numpy()
    y_flat = x_grid[:, 1].cpu().numpy()
    pts = np.stack([x_flat, y_flat], axis=1)
    crack_center = np.array(config["notch_tip"])
    dist = np.linalg.norm(pts - crack_center, axis=1)

    r_near = 0.1
    mask_near = dist < r_near

    d_v_flat = d_vpinn.reshape(-1)
    d_x_flat = d_xras.reshape(-1)

    l2_all = np.sqrt(np.mean((d_v_flat - d_x_flat) ** 2))
    l2_near = np.sqrt(np.mean((d_v_flat[mask_near] - d_x_flat[mask_near]) ** 2))

    print(f"    L2差异(全域): {l2_all:.4e}")
    print(f"    L2差异(裂尖r<{r_near}): {l2_near:.4e}")

    # 5. 绘图
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # -------------------------
    # 第一行：Damage场对比
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, d_vpinn, levels=60, vmin=0, vmax=1, cmap='viridis')
    ax1.plot(config["notch_length"], config["H"]/2, 'r*', markersize=12)
    ax1.set_title("VPINN - Damage Field", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, d_xras, levels=60, vmin=0, vmax=1, cmap='viridis')
    ax2.plot(config["notch_length"], config["H"]/2, 'r*', markersize=12)
    ax2.set_title("X-RAS-PINN - Damage Field", fontsize=12, fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(X, Y, diff, levels=60, cmap='magma')
    ax3.plot(config["notch_length"], config["H"]/2, 'r*', markersize=12)
    ax3.set_title(f"Absolute Difference\n(L2={l2_all:.3e})", fontsize=12, fontweight='bold')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)

    # -------------------------
    # 第二行：采样点分布对比
    # -------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    x_v = x_domain_vpinn.detach().cpu().numpy()
    ax4.scatter(x_v[:, 0], x_v[:, 1], s=2, alpha=0.4, c='blue')
    ax4.plot(config["notch_length"], config["H"]/2, 'r*', markersize=12, label='Notch tip')
    ax4.set_title(f"VPINN Sampling\n({x_domain_vpinn.shape[0]} points)",
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_aspect('equal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    x_x = x_domain_xras.detach().cpu().numpy()
    ax5.scatter(x_x[:, 0], x_x[:, 1], s=2, alpha=0.3, c='green')
    ax5.plot(config["notch_length"], config["H"]/2, 'r*', markersize=12, label='Notch tip')
    # 标记奇异区边界
    circle = plt.Circle(
        (config["notch_length"], config["H"]/2),
        config["r_sing"],
        color='orange',
        fill=False,
        linestyle='--',
        linewidth=2,
        label=f'Ω_sing (r={config["r_sing"]})'
    )
    ax5.add_patch(circle)
    ax5.set_title(f"X-RAS Sampling\n({x_domain_xras.shape[0]} points, +{x_domain_xras.shape[0]-x_domain_vpinn.shape[0]} RAS)",
                  fontsize=12, fontweight='bold')
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_aspect('equal')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 点密度热力图（X-RAS）
    ax6 = fig.add_subplot(gs[1, 2])
    H_density, xedges, yedges = np.histogram2d(
        x_x[:, 0], x_x[:, 1],
        bins=[30, 30],
        range=[[0, config["L"]], [0, config["H"]]]
    )
    im6 = ax6.imshow(H_density.T, origin='lower', aspect='auto',
                     extent=[0, config["L"], 0, config["H"]],
                     cmap='hot', interpolation='bilinear')
    ax6.plot(config["notch_length"], config["H"]/2, 'c*', markersize=12)
    ax6.set_title("X-RAS Point Density\n(Heatmap)", fontsize=12, fontweight='bold')
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    plt.colorbar(im6, ax=ax6, label='Points per bin')

    # -------------------------
    # 第三行：性能对比
    # -------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    methods = ["VPINN", "X-RAS"]
    times = [time_vpinn, time_xras]
    colors = ['skyblue', 'lightgreen']
    bars = ax7.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax7.set_ylabel("Training Time (s)", fontsize=11)
    ax7.set_title("Training Time Comparison", fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Damage分布直方图对比
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(d_v_flat, bins=50, alpha=0.6, label='VPINN', color='blue', edgecolor='black')
    ax8.hist(d_x_flat, bins=50, alpha=0.6, label='X-RAS', color='green', edgecolor='black')
    ax8.axvline(d_v_flat.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'VPINN mean={d_v_flat.mean():.3f}')
    ax8.axvline(d_x_flat.mean(), color='green', linestyle='--', linewidth=2,
                label=f'X-RAS mean={d_x_flat.mean():.3f}')
    ax8.set_xlabel("Damage d")
    ax8.set_ylabel("Count")
    ax8.set_title("Damage Distribution Comparison", fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # 量化指标表格
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    metrics_data = [
        ["Metric", "VPINN", "X-RAS"],
        ["d_max", f"{d_v_flat.max():.4f}", f"{d_x_flat.max():.4f}"],
        ["d_mean", f"{d_v_flat.mean():.4f}", f"{d_x_flat.mean():.4f}"],
        ["d_std", f"{d_v_flat.std():.4f}", f"{d_x_flat.std():.4f}"],
        ["Points", f"{x_domain_vpinn.shape[0]}", f"{x_domain_xras.shape[0]}"],
        ["Time (s)", f"{time_vpinn:.1f}", f"{time_xras:.1f}"],
        ["L2 diff (all)", "-", f"{l2_all:.3e}"],
        ["L2 diff (near)", "-", f"{l2_near:.3e}"],
    ]

    table = ax9.table(cellText=metrics_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 表头加粗
    for i in range(3):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax9.set_title("Quantitative Metrics", fontsize=12, fontweight='bold', pad=20)

    # 保存
    save_path = output_dir / "sent_phase2_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ 对比图已保存: {save_path}")

    return {
        "l2_all": l2_all,
        "l2_near": l2_near,
        "d_max_vpinn": d_v_flat.max(),
        "d_max_xras": d_x_flat.max(),
        "d_mean_vpinn": d_v_flat.mean(),
        "d_mean_xras": d_x_flat.mean(),
    }


# ============================================================================
# §6 主测试函数
# ============================================================================

def test_sent_phase2(debug=True):
    """
    主测试：SENT + notch 上的 VPINN vs X-RAS-PINN 完整对比

    Args:
        debug: True=快速测试, False=精细实验
    """
    output_dir = get_output_dir()

    print("\n" + "="*70)
    print("  SENT Phase-2: VPINN vs X-RAS-PINN 对比实验")
    print("="*70)
    print(f"  输出目录: {output_dir}\n")

    # 1. 加载统一配置
    print("[1/6] 加载统一配置...")
    config = create_config(debug=debug)
    print_config(config)

    # 2. 生成采样点
    print("[2/6] 生成SENT采样点...")
    torch.manual_seed(42)  # 固定随机种子
    np.random.seed(42)

    x_domain, x_bc = generate_sent_with_notch_points(config)
    print(f"  域内点: {x_domain.shape[0]}, 边界点: {x_bc.shape[0]}")

    # 可视化初始采样
    plt.figure(figsize=(7, 6))
    plt.scatter(x_domain[:, 0].numpy(), x_domain[:, 1].numpy(), s=2, alpha=0.4)
    plt.plot(config["notch_length"], config["H"]/2, 'r*', markersize=15, label='Notch tip')
    plt.title("Initial Sampling Points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_dir / "initial_sampling.png", dpi=150)
    plt.close()
    print(f"  ✓ 初始采样点图已保存")

    # 3. 边界条件（取最大载荷）
    print("\n[3/6] 构建边界条件...")
    max_displacement = config["max_displacement"]
    get_bc = get_bc_function_sent(config)
    u_bc = get_bc(max_displacement, x_bc)
    print(f"  最大位移: {max_displacement}")

    # 4. 训练VPINN baseline
    print("\n[4/6] 训练VPINN baseline...")
    vpinn_solver, time_vpinn = run_vpinn_baseline(config, x_domain, x_bc, u_bc)

    # 5. 训练X-RAS-PINN
    print("\n[5/6] 训练X-RAS-PINN...")
    xras_solver, x_domain_xras, time_xras = run_xras_solver(config, x_domain, x_bc, u_bc)

    # 6. 可视化与对比
    print("\n[6/6] 生成对比可视化...")
    metrics = visualize_comparison(
        config,
        vpinn_solver,
        xras_solver,
        x_domain,
        x_domain_xras,
        time_vpinn,
        time_xras,
        output_dir
    )

    # 7. 打印总结
    print("\n" + "="*70)
    print("  实验总结")
    print("="*70)
    print(f"  VPINN时间: {time_vpinn:.2f}s | X-RAS时间: {time_xras:.2f}s")
    print(f"  加速比: {time_vpinn/time_xras:.2f}x" if time_xras < time_vpinn
          else f"  X-RAS耗时: {time_xras/time_vpinn:.2f}x VPINN")
    print(f"  采样点: VPINN={x_domain.shape[0]}, X-RAS={x_domain_xras.shape[0]}")
    print(f"  L2误差(全域): {metrics['l2_all']:.4e}")
    print(f"  L2误差(裂尖): {metrics['l2_near']:.4e}")
    print("="*70)
    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print("  - initial_sampling.png")
    print("  - sent_phase2_comparison.png")

    return True


# ============================================================================
# §7 主入口
# ============================================================================

if __name__ == "__main__":
    # 可以通过命令行参数控制模式
    import argparse

    parser = argparse.ArgumentParser(description="SENT Phase-2 对比实验")
    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "full"],
        help="运行模式: debug=快速测试, full=精细实验"
    )

    args = parser.parse_args()

    debug_mode = (args.mode == "debug")

    print(f"\n启动模式: {'DEBUG (快速测试)' if debug_mode else 'FULL (精细实验)'}")

    try:
        success = test_sent_phase2(debug=debug_mode)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)