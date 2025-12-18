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
from fe_baseline_utils import (
    compare_fe_pinn_midline,
    plot_fe_vs_pinn_midline,
)

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
from solver_pinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver,
    generate_domain_points
)

from solver_xras import (
    XRASPINNSolver, SubdomainModels
)

from config import create_config, print_config

# 导入 Phase-1 桥接模块
try:
    from phase1_phase2_bridge import (
        load_phase1_checkpoint,
        setup_phase2_from_phase1,
        initialize_network_from_field
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    print("⚠️  警告: phase1_phase2_bridge.py 未找到，Phase-1 集成功能不可用")
    BRIDGE_AVAILABLE = False

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

def run_vpinn_baseline(config, x_domain, x_bc, u_bc, 
                       from_phase1=False, phase1_checkpoint=None,
                       d_prev_field=None): # ✅ 接收 d_prev_field):
    """
    训练单域VPINN作为baseline
    
    Args:
        config: 配置字典
        x_domain: 域内采样点
        x_bc: 边界采样点  
        u_bc: 边界条件
        from_phase1: 是否从 Phase-1 初始化
        phase1_checkpoint: Phase-1 检查点（如果 from_phase1=True）

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

    # 2. 根据模式选择初始化策略
    if from_phase1 and phase1_checkpoint is not None:
        print("\n  [1/3] 从 Phase-1 初始化...")
        
        # 从 Phase-1 加载网络并初始化
        u_net_phase1 = DisplacementNetwork()
        d_net_phase1 = DamageNetwork()
        
        u_net_phase1.load_state_dict(phase1_checkpoint['u_net_state_dict'])
        d_net_phase1.load_state_dict(phase1_checkpoint['d_net_state_dict'])
        
        u_net_phase1.eval()
        d_net_phase1.eval()
        
        # 生成目标场
        with torch.no_grad():
            u_field = u_net_phase1(x_domain)
            d_field = d_net_phase1(x_domain)
        
        print(f"    Phase-1 场统计:")
        print(f"      u: mean={u_field.mean().item():.4e}, max={u_field.abs().max().item():.4e}")
        print(f"      d: mean={d_field.mean().item():.4f}, max={d_field.max().item():.4f}")
        
        # 初始化 Phase-2 网络
        field_init_epochs = config.get('field_init_epochs', 500)
        
        print(f"    初始化位移网络...")
        initialize_network_from_field(u_net, x_domain, u_field, 
                                      n_epochs=field_init_epochs, verbose=False)
        
        print(f"    初始化损伤网络...")
        initialize_network_from_field(d_net, x_domain, d_field,
                                      n_epochs=field_init_epochs, verbose=False)
        
        # 验证初始化质量
        with torch.no_grad():
            u_init = u_net(x_domain)
            d_init = d_net(x_domain)
            u_error = torch.mean((u_init - u_field)**2).sqrt().item()
            d_error = torch.mean((d_init - d_field)**2).sqrt().item()
        
        print(f"    初始化质量: u_RMSE={u_error:.4e}, d_RMSE={d_error:.4e}")
        
    else:
        print("\n  [1/3] Notch损伤种子初始化...")
        d_net = initialize_notch_damage(d_net, x_domain, config)

    # 3. 构建求解器
    solver = PhaseFieldSolver(config, u_net, d_net)

    # ✅ 关键：设置 solver 的 d_prev，启用不可逆约束
    if from_phase1 and d_prev_field is not None:
        solver.d_prev = d_prev_field.to(config['device'])
        print(f"  [不可逆约束] 已设置 d_prev (max={d_prev_field.max():.3f})")

    # 4. 训练
    if from_phase1:
        # 从 Phase-1 继续训练，用较少的 epoch 精细化
        n_epochs_train = config.get('phase2_refine_epochs', 500)
        print(f"\n  [2/3] 精细化训练 VPINN ({n_epochs_train} epochs)...")
        # ✅ 强制启用强不可逆约束 (覆盖 config 中的 0.0)
        weight_irrev = 1000.0  # 强约束
    else:
        # 从头训练
        weight_irrev = 0.0
        n_epochs_train = config['n_epochs_vpinn']
        print(f"\n  [2/3] 训练 VPINN ({n_epochs_train} epochs)...")
    
    t0 = time.time()

    solver.train_step(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=n_epochs_train,
        weight_bc=config.get("weight_bc", 200.0),
        weight_irrev = weight_irrev,
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

def run_xras_solver(config, x_domain, x_bc, u_bc,
                   from_phase1=False, phase1_checkpoint=None,
                   d_prev_field=None):
    """
    训练X-RAS-PINN（域分解 + 自适应采样）
    
    支持消融实验:
    - use_partition: 是否使用域分解
    - use_ras: 是否使用自适应采样
    - use_interface_loss: 是否使用接口损失
    
    Args:
        config: 配置字典
        x_domain: 域内采样点
        x_bc: 边界采样点
        u_bc: 边界条件
        from_phase1: 是否从 Phase-1 初始化
        phase1_checkpoint: Phase-1 检查点

    Returns:
        solver: 训练好的求解器
        x_domain_new: 自适应采样后的点集
        time_elapsed: 训练时间（秒）
    """
    print("\n" + "="*70)
    print("  [X-RAS-PINN] 开始训练")
    print("="*70)
    
    # 读取实验控制开关
    use_partition = config.get("use_partition", True)
    use_ras = config.get("use_ras", True)
    use_interface_loss = config.get("use_interface_loss", True)

    # ✅ 读取 indicator_beta
    indicator_beta = config.get("indicator_beta", 0.6)

    print(f"  实验配置:")
    print(f"    域分解: {'ON' if use_partition else 'OFF'}")
    print(f"    自适应采样: {'ON' if use_ras else 'OFF'}")
    print(f"    Indicator Beta (β): {indicator_beta:.2f}")  # ✅ 打印 Beta 值
    print(f"    从 Phase-1 初始化: {'ON' if from_phase1 else 'OFF'}")

    # 1. 初始化子域网络
    u_net_1 = DisplacementNetwork()
    d_net_1 = DamageNetwork()
    u_net_2 = DisplacementNetwork()
    d_net_2 = DamageNetwork()

    # 2. 初始化策略
    if from_phase1 and phase1_checkpoint is not None:
        print("\n  [1/4] 从 Phase-1 初始化...")
        
        # 从 Phase-1 加载网络
        u_net_phase1 = DisplacementNetwork()
        d_net_phase1 = DamageNetwork()
        
        u_net_phase1.load_state_dict(phase1_checkpoint['u_net_state_dict'])
        d_net_phase1.load_state_dict(phase1_checkpoint['d_net_state_dict'])
        
        u_net_phase1.eval()
        d_net_phase1.eval()
        
        # 生成目标场
        with torch.no_grad():
            u_field = u_net_phase1(x_domain)
            d_field = d_net_phase1(x_domain)
        
        print(f"    Phase-1 场统计:")
        print(f"      d: mean={d_field.mean().item():.4f}, max={d_field.max().item():.4f}")
        
        # 初始化所有子域网络
        field_init_epochs = config.get('field_init_epochs', 300)
        
        print(f"    初始化 Ω_sing 网络...")
        initialize_network_from_field(u_net_1, x_domain, u_field, 
                                      n_epochs=field_init_epochs, verbose=False)
        initialize_network_from_field(d_net_1, x_domain, d_field,
                                      n_epochs=field_init_epochs, verbose=False)
        
        print(f"    初始化 Ω_far 网络...")
        initialize_network_from_field(u_net_2, x_domain, u_field,
                                      n_epochs=field_init_epochs, verbose=False)
        initialize_network_from_field(d_net_2, x_domain, d_field,
                                      n_epochs=field_init_epochs, verbose=False)
        
        # 验证初始化质量
        with torch.no_grad():
            d_init_1 = d_net_1(x_domain)
            d_init_2 = d_net_2(x_domain)
            d_error_1 = torch.mean((d_init_1 - d_field)**2).sqrt().item()
            d_error_2 = torch.mean((d_init_2 - d_field)**2).sqrt().item()
        
        print(f"    初始化质量: d_RMSE_1={d_error_1:.4e}, d_RMSE_2={d_error_2:.4e}")
        
    else:
        print("\n  [1/4] Notch初始化（仅Ω_sing网络）...")
        d_net_1 = initialize_notch_damage(d_net_1, x_domain, config)

    models = SubdomainModels(
        u_net_1=u_net_1,
        d_net_1=d_net_1,
        u_net_2=u_net_2,
        d_net_2=d_net_2,
    )

    # 3. 构建X-RAS求解器
    crack_center = torch.tensor([config["notch_length"], config["H"] / 2.0])
    r_sing = config["r_sing"]

    solver = XRASPINNSolver(config, models, crack_center, r_sing=r_sing)

    # ✅✅✅ 关键修改：将历史场传递给 solver，启用不可逆约束
    if from_phase1 and d_prev_field is not None:
        # 这里我们需要再次提取 d_prev_field，或者作为参数传进来
        # 既然 run_xras_solver 没有接收 d_prev_field 参数，我们得加上
        solver.set_history_field(x_domain, d_prev_field)

    # ✅ 根据消融开关调整接口损失权重
    if not use_interface_loss:
        print("  ⚠️  接口损失已禁用")
        original_weight = config["weight_interface"]
        config["weight_interface"] = 0.0

    # 4. 三阶段训练
    t0 = time.time()
    
    # 根据是否从 Phase-1 初始化调整训练策略
    if from_phase1:
        # 从 Phase-1 继续：减少 Phase 1 的 epoch，增加 Phase 2/3
        phase1_epochs = config["n_epochs_phase1"] // 2
        phase2_epochs = config["n_epochs_phase2"]
        phase3_epochs = config["n_epochs_phase3"]
        print(f"\n  训练策略（从 Phase-1 继续）:")
        print(f"    Phase 1: {phase1_epochs} epochs (减半)")
        print(f"    Phase 2: {phase2_epochs} epochs")
        print(f"    Phase 3: {phase3_epochs} epochs")
    else:
        # 从头训练：使用标准配置
        phase1_epochs = config["n_epochs_phase1"]
        phase2_epochs = config["n_epochs_phase2"]
        phase3_epochs = config["n_epochs_phase3"]

    # Phase 1: 预训练
    print(f"\n  [2/4] Phase 1: 预训练 ({phase1_epochs} epochs)...")
    solver.train_phase1_pretrain(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=phase1_epochs,
        weight_bc=config.get("weight_bc", 200.0),
        verbose=False
    )

    # Phase 2: 聚焦 + 自适应采样
    print(f"\n  [3/4] Phase 2: 聚焦训练 + RAS ({phase2_epochs} epochs)...")
    
    # ✅ 根据 use_ras 开关决定是否添加点
    N_add_actual = config["N_add_ras"] if use_ras else 0
    if not use_ras:
        print("  ⚠️  自适应采样已禁用，不增加新点")
    
    x_domain_new = solver.train_phase2_focused(
        x_domain,
        x_bc,
        u_bc,
        n_epochs=phase2_epochs,
        weight_bc=config.get("weight_bc", 200.0),
        weight_interface=config["weight_interface"],
        N_add=N_add_actual,
        beta=indicator_beta,  # ✅ 传入 indicator_beta 参数
        verbose=False
    )

    # 1. 切断计算图
    x_domain_new = x_domain_new.detach()
    # 2. 如果启用了Phase - 1约束，必须为新网格计算对应的d_prev
    if from_phase1 and phase1_checkpoint is not None:
        print(f"\n  [Info] 网格已更新 ({x_domain.shape[0]} -> {x_domain_new.shape[0]})，正在更新历史场...")

        # 临时重新加载 Phase-1 网络
        temp_d_net = DamageNetwork()
        temp_d_net.load_state_dict(phase1_checkpoint['d_net_state_dict'])
        temp_d_net.eval()

        with torch.no_grad():
            # 对新网格 (x_domain_new) 进行预测
            # 这确保了新加的点也有正确的历史约束值，而不是 0
            d_prev_new = temp_d_net(x_domain_new).detach()

        # 调用 Solver 的方法更新内部状态
        solver.set_history_field(x_domain_new, d_prev_new)

    # Phase 3: 联合微调
    print(f"\n  [4/4] Phase 3: 联合微调 ({phase3_epochs} epochs)...")
    solver.train_phase3_joint_finetune(
        x_domain_new,
        x_bc,
        u_bc,
        n_epochs=phase3_epochs,
        n_interface=config.get("n_interface", 100),
        weight_bc=config.get("weight_bc", 200.0),
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
    output_dir,
    exp_name: str = "Baseline"
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
    crack_center = np.array([config["notch_length"], config["H"] / 2.0])
    dist = np.linalg.norm(pts - crack_center, axis=1)

    r_near = config.get("r_near", 0.1)  # 从 config 读取近场半径
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
    safe_name = exp_name.replace('-', '_').replace(' ', '')  # ✅ 清理名称
    save_path = output_dir / f"sent_phase2_comparison_{safe_name}.png" # ✅ 使用 exp_name 构造路径
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

def test_sent_phase2(debug=True, input_config=None, exp_name="Baseline", fe_path=None):
    """
    主测试：SENT + notch 上的 VPINN vs X-RAS-PINN 完整对比

    Args:
        debug: True=快速测试, False=精细实验
        input_config: 外部传入的配置字典 (用于消融/参数扫描)
    """
    output_dir = get_output_dir()

    if fe_path is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        fe_path = os.path.join(BASE_DIR, "data", "fe_sent_phasefield.npz")

        print("[FE] Using baseline file:", fe_path)

    print("\n" + "="*70)
    print("  SENT Phase-2: VPINN vs X-RAS-PINN 对比实验")
    print("="*70)
    print(f"  输出目录: {output_dir}\n")

    # 1. 加载统一配置
    print("[1/6] 加载统一配置...")
    # ✅ 优先使用外部传入的配置
    if input_config is not None:
        config = input_config
        # 确保 debug 模式标志与传入的配置匹配
        print(f"  ✓ 外部配置已加载 ({'DEBUG' if debug else 'FULL'})")
    else:
        # 如果没有外部配置，则根据 debug 标志创建新配置
        config = create_config(debug=debug)

    print_config(config)

    # 2. 生成采样点
    print("[2/6] 生成SENT采样点...")
    torch.manual_seed(config["seed"])  # 固定随机种子
    np.random.seed(config["seed"])

    # ✅ 预算模式逻辑：统一控制 VPINN 和 X-RAS 的基础点数
    N_base = config["n_domain"]

    config_vpinn = config.copy()
    config_xras = config.copy()

    # ✅ 预算模式逻辑：如果开启，调整基础点数
    if config["budget_mode"]:
        # --- 预算模式（固定总点数）---
        N_total_budget = N_base
        N_add = config["N_add_ras"]

        # 1. X-RAS 基础点数：总预算 - RAS增加点数
        N_xras_base = max(N_total_budget - N_add, 500)  # 确保至少 500 个基础点

        # 2. VPINN 点数：使用总预算点数
        N_vpinn_base = N_total_budget

        print(f"  ✓ 预算模式开启: 总点数预算 = {N_total_budget} 点")
        print(f"  VPINN Baseline 使用: {N_vpinn_base} 点")
        print(f"  X-RAS 使用: {N_xras_base} 基础点 + {N_add} RAS点")

        # 更新配置中的点数，用于采样函数
        config_vpinn["n_domain"] = N_vpinn_base
        config_xras["n_domain"] = N_xras_base

        # ⚠️ 注意：config_vpinn 和 config_xras 仅在 budget_mode 下 n_domain 可能不同
        # 在 standard_mode 下，它们都使用 config["n_domain"]

    else:
        print(f"  ✓ 标准模式: 基础点数 = {config['n_domain']}, RAS增加 = {config['N_add_ras']}")
        config_vpinn = config
        config_xras = config

    # --- 生成采样点 (使用 VPINN 的 n_domain 设定) ---
    # 我们先生成 VPINN 所需的 N_vpinn_base 点集，X-RAS 将在其上进行划分和 RAS
    x_domain, x_bc = generate_sent_with_notch_points(config_vpinn)
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
    
    # ✅ 检查是否使用 Phase-1 结果
    use_phase1 = config.get("use_phase1_result", False)
    phase1_checkpoint = None
    actual_load_value = config["max_displacement"]
    
    if use_phase1 and BRIDGE_AVAILABLE:
        checkpoint_path = config.get("phase1_model_path", "outputs/phase1_checkpoint.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"\n  ✓ 检测到 Phase-1 检查点: {checkpoint_path}")
            print(f"  ✓ 将从 Phase-1 加载结果作为初始状态")
            
            try:
                # 加载 Phase-1 检查点
                phase1_checkpoint = load_phase1_checkpoint(checkpoint_path, verbose=True)
                
                # 获取指定步的载荷
                from phase1_phase2_bridge import get_phase1_field_at_step
                step_index = config.get("phase1_load_step", -1)
                step_info = get_phase1_field_at_step(phase1_checkpoint, step_index)
                actual_load_value = step_info["load"]
                
                print(f"\n  使用 Phase-1 的载荷: {actual_load_value:.6f}")
                
            except Exception as e:
                print(f"\n  ⚠️  加载 Phase-1 失败: {e}")
                print(f"  将退回到标准模式（从 notch 初始化）")
                use_phase1 = False
                phase1_checkpoint = None
                actual_load_value = config["max_displacement"]
        else:
            print(f"\n  ⚠️  Phase-1 检查点不存在: {checkpoint_path}")
            print(f"  将退回到标准模式（从 notch 初始化）")
            print(f"  提示: 请先运行 test_sent_fixed.py 生成 Phase-1 结果")
            use_phase1 = False
    elif use_phase1 and not BRIDGE_AVAILABLE:
        print(f"\n  ⚠️  phase1_phase2_bridge.py 不可用，无法使用 Phase-1 结果")
        print(f"  将退回到标准模式")
        use_phase1 = False
    
    # 根据实际载荷生成边界条件
    get_bc = get_bc_function_sent(config)
    u_bc = get_bc(actual_load_value, x_bc)
    print(f"  载荷值: {actual_load_value:.6f}")

    # 提取 Phase-1 损伤场 (作为不可逆约束)
    d_prev_field = None
    if use_phase1 and phase1_checkpoint is not None:
        # 重新构建一个临时网络来获取场值
        temp_d_net = DamageNetwork()
        temp_d_net.load_state_dict(phase1_checkpoint['d_net_state_dict'])
        temp_d_net.eval()
        with torch.no_grad():
            d_prev_field = temp_d_net(x_domain).detach()  # 获取基准场
            print(f"  [Info] 提取 Phase-1 损伤场用于约束: mean={d_prev_field.mean():.4f}")

    # 4. 训练VPINN baseline
    print("\n[4/6] 训练VPINN baseline...")
    # VPINN 使用全量初始采样点
    vpinn_solver, time_vpinn = run_vpinn_baseline(
        config_vpinn, x_domain, x_bc, u_bc,
        from_phase1=use_phase1,
        phase1_checkpoint=phase1_checkpoint,
        d_prev_field=d_prev_field  # ✅ 新增参数：传入历史场
    )

    # 5. 训练X-RAS-PINN
    print("\n[5/6] 训练X-RAS-PINN...")
    # X-RAS 同样使用 x_domain，但在内部根据 config_xras["n_domain"] 划分/初始化
    # 如果是非预算模式，config_xras["n_domain"] == config_vpinn["n_domain"]
    # 如果是预算模式，config_xras["n_domain"] < config_vpinn["n_domain"]


    # --- 为 X-RAS 提取基础点集 ---
    if config["budget_mode"]:
        N_xras_base = config_xras["n_domain"]  # N_xras_base < N_vpinn_base

        # 随机从 x_domain 中抽取 N_xras_base 个点
        indices = torch.randperm(x_domain.shape[0])[:N_xras_base]
        x_domain_xras_base = x_domain[indices].clone().detach()
        print(f"  X-RAS 初始点集从 VPINN 点集中随机抽取 {x_domain_xras_base.shape[0]} 点")
    else:
        x_domain_xras_base = x_domain  # 标准模式下使用全部初始点

    xras_solver, x_domain_xras, time_xras = run_xras_solver(
        config_xras, x_domain_xras_base, x_bc, u_bc,  # ✅ X-RAS 使用 config_xras 和其基础点集
        from_phase1=use_phase1,
        phase1_checkpoint=phase1_checkpoint,
        d_prev_field=d_prev_field
    )

    # 6. 可视化与对比
    print("\n[6/6] 生成对比可视化...")
    metrics = visualize_comparison(
        config,
        vpinn_solver,
        xras_solver,
        x_domain,  # VPINN 初始采样点
        x_domain_xras,  # X-RAS 最终采样点
        time_vpinn,
        time_xras,
        output_dir,
        exp_name
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

    safe_name = exp_name.replace('-', '_').replace(' ', '')

    # 8. 保存 Phase-2 原始结果（供之后离线 FE summary 使用）
    try:
        phase2_ckpt = {
            "config": config,
            "exp_name": exp_name,
            "metrics": metrics,
            "time_vpinn": float(time_vpinn),
            "time_xras": float(time_xras),
            "x_domain_vpinn": x_domain.detach().cpu().numpy(),
            "x_domain_xras": x_domain_xras.detach().cpu().numpy(),
            # VPINN: 单域网络
            "vpinn_state": {
                "u_net": vpinn_solver.u_net.state_dict(),
                "d_net": vpinn_solver.d_net.state_dict(),
            },
            # X-RAS: 四个子域网络
            "xras_state": {
                "u_net_1": xras_solver.models.u_net_1.state_dict(),
                "d_net_1": xras_solver.models.d_net_1.state_dict(),
                "u_net_2": xras_solver.models.u_net_2.state_dict(),
                "d_net_2": xras_solver.models.d_net_2.state_dict(),
            },
        }

        ckpt_path = output_dir / f"phase2_raw_{safe_name}.pth"
        torch.save(phase2_ckpt, ckpt_path)
        print(f"  - Phase-2 原始 checkpoint: {ckpt_path}")
    except Exception as e:
        print(f"  ⚠️ 保存 Phase-2 checkpoint 失败: {e}")

    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print("  - initial_sampling.png")
    print(f"  - sent_phase2_comparison_{safe_name}.png")
    print(f"  - phase2_raw_{safe_name}.pth  （供后续 FE summary 使用）")

    return True

    # # 7+. FE 基准对比 (可选)
    # if fe_path is not None and os.path.exists(fe_path):
    #     print("\n[附加] FE 基准对比 (midline)...")
    #
    #     # 注意：component 要和 FE 载荷方向一致
    #     # 如果 FE 是右端 x 向拉伸 → component="ux"
    #     # 如果 FE 是上/下 y 向拉伸 → component="uy"
    #     fe_component = "uy"  # 或 "uy"，视你的 FE 脚本而定
    #
    #     # VPINN vs FE
    #     stats_v = plot_fe_vs_pinn_midline(
    #         vpinn_solver,
    #         fe_path=fe_path,
    #         orientation="horizontal",  # 或 "vertical"，看你想取哪条中线
    #         component=fe_component,
    #         save_path=output_dir / f"fe_vs_vpinn_midline_{safe_name}.png",
    #     )
    #
    #     # X-RAS vs FE
    #     stats_x = plot_fe_vs_pinn_midline(
    #         xras_solver,
    #         fe_path=fe_path,
    #         orientation="horizontal",
    #         component=fe_component,
    #         save_path=output_dir / f"fe_vs_xras_midline_{safe_name}.png",
    #     )
    #
    #     print(f"  [FE对比] VPINN Rel L2 = {stats_v['rel_l2']:.3e}")
    #     print(f"  [FE对比] X-RAS  Rel L2 = {stats_x['rel_l2']:.3e}")
    #     print("  已保存: ")
    #     print(f"   - fe_vs_vpinn_midline_{safe_name}.png")
    #     print(f"   - fe_vs_xras_midline_{safe_name}.png")
    # else:
    #     print("\n[附加] 未找到 FE baseline npz，跳过 FE 对比")


    print(f"\n✓ 所有结果已保存到: {output_dir}")
    print("  - initial_sampling.png")
    print(f"  - sent_phase2_comparison_{safe_name}.png")

    # # 8. 可选：生成 Phase-2 vs FE 的 summary npz（便于后处理/画表）
    # try:
    #     # 局部导入，避免在没有 fe_baseline_utils.py 时直接崩溃
    #     from fe_baseline_utils import (
    #         load_fe_phasefield_npz,
    #         compute_global_L2_for_u_and_d,
    #         compare_damage_midline,
    #         compare_load_reaction_curve,   # 目前没在这里用，但保留接口
    #     )
    # except ImportError:
    #     print("\n[FE Summary] 找不到 fe_baseline_utils.py，跳过 FE 对比与 summary 保存。")
    # else:
    #     try:
    #         # 1) 推断 FE 相场 npz 路径
    #         #    优先使用函数参数 fe_path，如果不是 npz，则回退到默认路径
    #         if fe_path is not None and fe_path.endswith(".npz"):
    #             fe_npz_path = fe_path
    #         else:
    #             base_dir = os.path.dirname(os.path.abspath(__file__))
    #             fe_npz_path = os.path.join(base_dir, "data", "fe_sent_phasefield.npz")
    #
    #         print(f"\n[FE Summary] 使用 FE 相场基准: {fe_npz_path}")
    #
    #         # 2) 全局 L2 误差 (u, d)：分别对 VPINN 和 X-RAS 计算
    #         stats_v = compute_global_L2_for_u_and_d(
    #             vpinn_solver, fe_phase_path=fe_npz_path, verbose=False
    #         )
    #         stats_x = compute_global_L2_for_u_and_d(
    #             xras_solver, fe_phase_path=fe_npz_path, verbose=False
    #         )
    #
    #         # 3) 水平中线 (y ≈ H/2) 的损伤误差，作为代表性局部指标
    #         mid_v = compare_damage_midline(
    #             vpinn_solver,
    #             fe_phase_path=fe_npz_path,
    #             orientation="horizontal",
    #             verbose=False,
    #         )
    #         mid_x = compare_damage_midline(
    #             xras_solver,
    #             fe_phase_path=fe_npz_path,
    #             orientation="horizontal",
    #             verbose=False,
    #         )
    #
    #         # 4) （占位）载荷–反力曲线误差
    #         #    如果你后面在训练里统计了 PINN 的 reactions，可以在这里用
    #         #    compare_load_reaction_curve(...) 得到 l2_curve_*，暂时先填 nan
    #         l2_curve_v = float("nan")
    #         rel_l2_curve_v = float("nan")
    #         l2_curve_x = float("nan")
    #         rel_l2_curve_x = float("nan")
    #
    #         # 5) 写入 summary npz
    #         summary_path = output_dir / f"phase2_vs_fe_summary_{safe_name}.npz"
    #         np.savez(
    #             summary_path,
    #             exp_name=exp_name,
    #             fe_path=fe_npz_path,
    #             # --- 全域 L2 (u, d) ---
    #             l2_u_vpinn=stats_v["l2_u"],
    #             rel_l2_u_vpinn=stats_v["rel_l2_u"],
    #             l2_d_vpinn=stats_v["l2_d"],
    #             rel_l2_d_vpinn=stats_v["rel_l2_d"],
    #             l2_u_xras=stats_x["l2_u"],
    #             rel_l2_u_xras=stats_x["rel_l2_u"],
    #             l2_d_xras=stats_x["l2_d"],
    #             rel_l2_d_xras=stats_x["rel_l2_d"],
    #             # --- midline (horizontal) 损伤 L2 ---
    #             l2_d_mid_vpinn=mid_v["l2"],
    #             rel_l2_d_mid_vpinn=mid_v["rel_l2"],
    #             l2_d_mid_xras=mid_x["l2"],
    #             rel_l2_d_mid_xras=mid_x["rel_l2"],
    #             # --- 载荷–反力曲线误差（占位，后面可替换为真实数值） ---
    #             l2_curve_vpinn=l2_curve_v,
    #             rel_l2_curve_vpinn=rel_l2_curve_v,
    #             l2_curve_xras=l2_curve_x,
    #             rel_l2_curve_xras=rel_l2_curve_x,
    #         )
    #         print(f"[FE Summary] Phase-2 vs FE summary 已保存到: {summary_path}")
    #     except Exception as e:
    #         print(f"[FE Summary] 生成 summary 失败: {e}")

    return True



# ============================================================================
# §7 主入口
# ============================================================================
#
# if __name__ == "__main__":
#     # 可以通过命令行参数控制模式
#     import argparse
#
#     parser = argparse.ArgumentParser(description="SENT Phase-2 对比实验")
#     parser.add_argument(
#         "--mode",
#         type=str,
#         default="debug",
#         choices=["debug", "full"],
#         help="运行模式: debug=快速测试, full=精细实验"
#     )
#
#     args = parser.parse_args()
#
#     debug_mode = (args.mode == "debug")
#
#     print(f"\n启动模式: {'DEBUG (快速测试)' if debug_mode else 'FULL (精细实验)'}")
#
#     try:
#         # ✅ 调用时不再传递 config，只传递 debug 标志
#         success = test_sent_phase2(debug=debug_mode, input_config=None)
#         sys.exit(0 if success else 1)
#     except Exception as e:
#         print(f"\n❌ 错误: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)