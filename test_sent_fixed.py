"""
SENT测试 - 带Notch初始损伤版本 (MacBook适配, 带 DEBUG_MODE)

功能：
1. 在 notch 尖端初始化损伤种子 d(x)
2. notch 附近加密采样点
3. Phase-field 相场断裂 + VPINN/DRM 求解
4. DEBUG_MODE 切换：快速调参 / 精细论文级结果
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===========================
# 全局开关：调试模式
# ===========================
DEBUG_MODE = True   # True=快速测试；False=精细实验


# ===========================
# 输出路径
# ===========================
def get_output_dir():
    """获取输出目录: ./outputs"""
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ===========================
# 导入相场 VPINN 核心模块
# ===========================
try:
    from phase_field_vpinn import (
        DisplacementNetwork,
        DamageNetwork,
        PhaseFieldSolver,
        generate_domain_points,
        visualize_solution,
        compute_strain,
        compute_energy_split,
    )
except ImportError:
    print("错误: 找不到 phase_field_vpinn.py")
    print("请确保 phase_field_vpinn.py 在当前目录下。")
    sys.exit(1)


# ===========================
# 统一配置管理
# ===========================
def create_config(debug: bool = DEBUG_MODE):
    """
    统一管理所有物理 & 数值参数。
    debug=True: 快速调参配置
    debug=False: 论文级精细配置
    """
    # 几何 & 材料常数
    L = 1.0
    H = 1.0
    notch_length = 0.3
    E = 210.0
    nu = 0.3
    k = 1e-6
    device = "cpu"
    lr_u = 2e-4
    lr_d = 2e-4

    base = {
        "L": L,
        "H": H,
        "notch_length": notch_length,
        "E": E,
        "nu": nu,
        "k": k,
        "device": device,
        "lr_u": lr_u,
        "lr_d": lr_d,
    }

    if debug:
        print(">> Running in DEBUG_MODE (快速调参配置)")
        base.update(
            {
                # 相场材料
                "G_c": 10e-3,
                "l": 0.003,
                # 采样 & 载荷
                "n_domain": 1500,
                "n_bc": 100,
                "max_displacement": 0.008,
                "n_loading_steps": 8,
                # notch 初始化
                "notch_seed_radius": 0.02,    # 初始化种子半径（高斯核）
                "initial_d": 0.5,             # 初始化 d 峰值
                "notch_init_epochs": 300,     # notch 预训练 epoch
                # 准静态训练
                "n_epochs_initial": 700,      # 前几个 load step
                "n_epochs_later": 350,        # 后面 load step
                "n_epochs_switch": 3,         # n < 3 用 initial
                "weight_irrev": 800.0,        # 不可逆约束权重
                # notch 保持项（第1步）
                "notch_region_radius": 0.02,
                "notch_hold_weight": 10.0,
                "notch_hold_target": 0.8,
                # 远场区域半径（用于 d_far 统计）
                "far_region_radius": 0.25,
            }
        )
    else:
        print(">> Running in FULL MODE (精细实验配置)")
        base.update(
            {
                # 相场材料
                "G_c": 6e-3,
                "l": 0.004,
                # 采样 & 载荷
                "n_domain": 4000,
                "n_bc": 200,
                "max_displacement": 0.010,
                "n_loading_steps": 20,
                # notch 初始化
                "notch_seed_radius": 0.04,
                "initial_d": 0.65,
                "notch_init_epochs": 1200,
                # 准静态训练
                "n_epochs_initial": 1200,
                "n_epochs_later": 800,
                "n_epochs_switch": 3,
                "weight_irrev": 1200.0,
                # notch 保持项
                "notch_region_radius": 0.03,
                "notch_hold_weight": 20.0,
                "notch_hold_target": 0.8,
                "far_region_radius": 0.2,
            }
        )

    return base


# ===========================
# 采样点生成（SENT + notch 加密）
# ===========================
def generate_sent_with_notch_points(config):
    """
    生成 SENT 采样点，在 notch 尖端附近加密。
    """
    L = config["L"]
    H = config["H"]
    notch_length = config["notch_length"]
    n_domain = config["n_domain"]
    n_bc = config["n_bc"]

    notch_tip = np.array([notch_length, H / 2])

    # 70% 均匀 + 30% notch 附近
    n_uniform = int(n_domain * 0.7)
    n_concentrated = n_domain - n_uniform

    x_domain_list = []

    # 1) 均匀采样（略避开 notch 凹槽）
    while len(x_domain_list) < n_uniform:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)

        if x < notch_length and abs(y - H / 2) < 0.02:
            continue

        x_domain_list.append([x, y])

    # 2) notch 尖端附近加密
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

    # 边界点：下边固定，上边施加位移
    n_bc_half = n_bc // 2
    x_bottom = np.linspace(0, L, n_bc_half)
    y_bottom = np.zeros_like(x_bottom)
    bc_bottom = np.stack([x_bottom, y_bottom], axis=1)

    x_top = np.linspace(0, L, n_bc_half)
    y_top = np.ones_like(x_top) * H
    bc_top = np.stack([x_top, y_top], axis=1)

    x_bc = torch.tensor(np.vstack([bc_bottom, bc_top]), dtype=torch.float32)

    return x_domain, x_bc


# ===========================
# notch 初始损伤种子
# ===========================
def initialize_notch_damage(d_net, x_domain, config):
    """
    在 notch 尖端初始化损伤种子 d(x)。

    思路：
    - 在 notch 尖端附近设置一个高斯型目标 d_target
    - 用 MSE + 尖端强化损失训练 d_net 若干 epoch
    """
    notch_length = config["notch_length"]
    H = config["H"]
    initial_d = config["initial_d"]
    seed_radius = config["notch_seed_radius"]
    n_epochs = config["notch_init_epochs"]

    notch_tip = torch.tensor([notch_length, H / 2.0])

    # 距离
    distances = torch.norm(x_domain - notch_tip, dim=1)

    # 高斯目标场
    d_target = initial_d * torch.exp(-(distances / seed_radius) ** 2)
    d_target = d_target.unsqueeze(1)
    d_target = torch.clamp(d_target, 0.0, 1.0)

    # ✅ 远场强制为 0（防止高斯尾巴太厚）
    cut_radius = 1.5 * seed_radius
    far_mask = distances > cut_radius
    d_target[far_mask] = 0.0

    # 极近点强制高损伤
    very_close = distances < (0.5 * seed_radius)
    d_target[very_close] = 0.98

    print("\n  初始化 notch 损伤种子:")
    print(f"    尖端位置: ({notch_length:.2f}, {H/2:.2f})")
    print(f"    高斯半径: {seed_radius:.3f}")
    print(f"    初始峰值: {initial_d:.2f}")
    print(f"    受影响点数(d>0.1): {(d_target > 0.1).sum().item()}")
    print(f"    极近点数(d>0.9):   {(d_target > 0.9).sum().item()}")

    optimizer = torch.optim.Adam(d_net.parameters(), lr=5e-4)

    best_loss = float("inf")
    patience = 0

    print(f"    训练 d_net 拟合 d_target（{n_epochs} epochs）...")
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        d_pred = d_net(x_domain)

        loss_mse = torch.mean((d_pred - d_target) ** 2)

        tip_points = distances < seed_radius
        if tip_points.sum() > 0:
            loss_tip = torch.mean((d_pred[tip_points] - 0.95) ** 2)
        else:
            loss_tip = 0.0

        far_points = distances > cut_radius
        if far_points.sum() > 0:
            loss_far = torch.mean(d_pred[far_points] ** 2)
        else:
            loss_far = 0.0

        loss = loss_mse + 1.5 * loss_tip + 2.0 * loss_far

        # loss = loss_mse + 2.0 * loss_tip
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                d_max_now = d_pred.max().item()
                d_mean_now = d_pred.mean().item()
            print(
                f"      Epoch {epoch:4d}: loss={loss.item():.6e} | "
                f"d_max={d_max_now:.3f}, d_mean={d_mean_now:.3f}"
            )

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 200 and epoch > 500:
                print(f"      Early stopping at epoch {epoch}")
                break

    with torch.no_grad():
        d_final = d_net(x_domain)
        d_max = d_final.max().item()
        d_mean = d_final.mean().item()
        d_std = d_final.std().item()
        d_at_tip = (
            d_final[distances < seed_radius].mean().item()
            if (distances < seed_radius).sum() > 0
            else 0.0
        )

    print("\n    ✓ 初始化完成:")
    print(f"      d_max:    {d_max:.3f}")
    print(f"      d_mean:   {d_mean:.3f}")
    print(f"      d_std:    {d_std:.3f}")
    print(f"      d_at_tip: {d_at_tip:.3f}")

    return d_net


# ===========================
# 边界条件
# ===========================
def get_bc_function_sent(config):
    """拉伸：下边固定，上边 y 向位移 = load_value"""

    H = config["H"]

    def get_bc(load_value, x_bc):
        n_bc = x_bc.shape[0]
        u_bc = torch.zeros(n_bc, 2)
        # 下边：全零
        u_bc[: n_bc // 2, :] = 0.0
        # 上边：x 方向 0, y 方向 = load_value
        u_bc[n_bc // 2 :, 0] = 0.0
        u_bc[n_bc // 2 :, 1] = load_value
        return u_bc

    return get_bc


# ===========================
# 主测试函数
# ===========================
def test_sent_with_notch():
    """运行带 notch 的 SENT 相场测试"""

    output_dir = get_output_dir()
    print(f"输出目录: {output_dir}")

    print("=" * 70)
    print("  SENT Test with Notch Initialization")
    print("=" * 70)

    # 1. 配置
    print("\n[1/7] Creating configuration...")
    config = create_config(DEBUG_MODE)
    print(f"  Gc = {config['G_c']:.2e}")
    print(f"  l  = {config['l']}")
    print(f"  Gc/l = {config['G_c'] / config['l']:.2f}")
    print(f"  n_domain = {config['n_domain']}, n_bc = {config['n_bc']}")
    print(f"  max_displacement = {config['max_displacement']}, "
          f"n_loading_steps = {config['n_loading_steps']}")

    # 2. 采样点
    print("\n[2/7] Generating sampling points (concentrated near notch)...")
    x_domain, x_bc = generate_sent_with_notch_points(config)
    print(f"  Domain points: {x_domain.shape[0]}")

    # 保存采样点图
    plt.figure(figsize=(6, 4))
    plt.scatter(x_domain[:, 0].numpy(), x_domain[:, 1].numpy(), s=1, alpha=0.5)
    plt.scatter(
        config["notch_length"],
        config["H"] / 2,
        s=80,
        c="red",
        marker="*",
        label="Notch tip",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sampling Points Distribution")
    plt.legend()
    sampling_path = os.path.join(output_dir, "sampling_points.png")
    plt.savefig(sampling_path, dpi=150)
    plt.close()
    print(f"  Sampling visualization saved to: {sampling_path}")

    # 3. 网络
    print("\n[3/7] Initializing networks...")
    u_net = DisplacementNetwork(layers=[2, 64, 64, 64, 2])
    d_net = DamageNetwork(layers=[2, 64, 64, 64, 1])

    # 4. 初始化 notch 损伤
    print("\n[4/7] Initializing notch damage seed...")
    d_net = initialize_notch_damage(d_net, x_domain, config)

    # 5. 求解器
    print("\n[5/7] Creating solver...")
    solver = PhaseFieldSolver(config, u_net, d_net)

    # 6. 准静态加载
    print("\n[6/7] Quasi-static loading...")
    n_loading_steps = config["n_loading_steps"]
    max_displacement = config["max_displacement"]
    loading_steps = np.linspace(0.0, max_displacement, n_loading_steps)

    get_bc = get_bc_function_sent(config)
    history = []

    print("\nInitializing fields...")
    solver.initialize_fields(x_domain)

    # notch 区域掩码（用于统计 & notch 保持损失）
    notch_tip = torch.tensor([config["notch_length"], config["H"] / 2])
    distances_to_tip = torch.norm(x_domain - notch_tip, dim=1)
    notch_region = distances_to_tip < config["notch_region_radius"]
    far_region = distances_to_tip > config["far_region_radius"]

    for n, load_value in enumerate(loading_steps):
        print("\n" + "=" * 60)
        print(f"Step {n + 1}/{len(loading_steps)} | Load = {load_value:.6f}")
        print("=" * 60)

        u_bc = get_bc(load_value, x_bc)

        with torch.no_grad():
            d_prev = solver.d_net(x_domain).detach().clone()

        solver.u_net.train()
        solver.d_net.train()

        # 每步 epoch 数
        if n < config["n_epochs_switch"]:
            n_epochs = config["n_epochs_initial"]
        else:
            n_epochs = config["n_epochs_later"]

        for epoch in range(n_epochs):
            solver.optimizer_u.zero_grad()
            solver.optimizer_d.zero_grad()

            L_energy = solver.drm_loss.compute_energy_loss(
                x_domain, solver.u_net, solver.d_net
            )
            L_bc = solver.drm_loss.compute_bc_loss(
                x_bc, u_bc, solver.u_net, 200.0
            )
            L_irrev = solver.drm_loss.compute_irreversibility_loss(
                x_domain, solver.d_net, d_prev, config["weight_irrev"]
            )

            # notch 区域保持高损伤（主要在第1步）
            d_current = solver.d_net(x_domain)
            if notch_region.sum() > 0:
                notch_weight = (
                    config["notch_hold_weight"] if n == 0 else 0.0
                )
                target_notch_d = config["notch_hold_target"]
                L_notch = notch_weight * torch.mean(
                    (d_current[notch_region] - target_notch_d) ** 2
                )
            else:
                L_notch = torch.tensor(0.0)

            loss = L_energy + L_bc + L_irrev + L_notch
            loss.backward()

            torch.nn.utils.clip_grad_norm_(solver.u_net.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(solver.d_net.parameters(), 0.5)

            solver.optimizer_u.step()
            solver.optimizer_d.step()

            if epoch % 300 == 0 or epoch == n_epochs - 1:
                with torch.no_grad():
                    d_check = solver.d_net(x_domain)
                    d_max = d_check.max().item()
                    d_mean = d_check.mean().item()
                    d_std = d_check.std().item()
                    d_notch = (
                        d_check[notch_region].mean().item()
                        if notch_region.sum() > 0
                        else 0.0
                    )
                    if far_region.sum() > 0:
                        d_far = d_check[far_region].mean().item()
                    else:
                        d_far = 0.0
                    loc_index = d_notch / (d_far + 1e-6) if d_far > 0 else 0.0

                loss_str = (
                    f"Loss={loss.item():.3e} | "
                    f"E={L_energy.item():.2e}, BC={L_bc.item():.2e}, "
                    f"Irr={L_irrev.item():.2e}, N={float(L_notch):.2e}"
                )
                print(
                    f"  Epoch {epoch:4d} | {loss_str} | "
                    f"d_mean={d_mean:.3f}, d_max={d_max:.3f}, "
                    f"std={d_std:.3f}, notch={d_notch:.3f}, "
                    f"far={d_far:.3f}, loc={loc_index:.1f}"
                )

        # 记录每步统计
        with torch.no_grad():
            d_final_step = solver.d_net(x_domain)
            d_max_f = d_final_step.max().item()
            d_mean_f = d_final_step.mean().item()
            d_std_f = d_final_step.std().item()
            d_notch_f = (
                d_final_step[notch_region].mean().item()
                if notch_region.sum() > 0
                else 0.0
            )
            d_far_f = (
                d_final_step[far_region].mean().item()
                if far_region.sum() > 0
                else 0.0
            )
            loc_index_f = (
                d_notch_f / (d_far_f + 1e-6) if d_far_f > 0 else 0.0
            )

        history.append(
            {
                "step": n,
                "load": load_value,
                "d_max": d_max_f,
                "d_mean": d_mean_f,
                "d_std": d_std_f,
                "d_notch": d_notch_f,
                "d_far": d_far_f,
                "loc_index": loc_index_f,
            }
        )

        print(
            f"Step summary: d_max={d_max_f:.4f}, "
            f"d_mean={d_mean_f:.4f}, d_std={d_std_f:.4f}, "
            f"loc_index={loc_index_f:.1f}"
        )

    # 7. 可视化
    print("\n[7/7] Visualization...")
    nx, ny = 150, 150
    x_grid = generate_domain_points(
        nx, ny, x_range=(0, config["L"]), y_range=(0, config["H"])
    )

    result_path = os.path.join(output_dir, "sent_with_notch.png")
    visualize_solution(solver, x_grid, nx, ny, save_path=result_path)
    print(f"  Damage field saved to: {result_path}")

    # 统计图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    loads = [h["load"] for h in history]
    d_max_list = [h["d_max"] for h in history]
    d_mean_list = [h["d_mean"] for h in history]
    d_std_list = [h["d_std"] for h in history]
    loc_list = [h["loc_index"] for h in history]

    # (1) d_max & d_mean 演化
    axes[0].plot(loads, d_max_list, "o-", linewidth=2, label="d_max", markersize=4)
    axes[0].plot(loads, d_mean_list, "s-", linewidth=2, label="d_mean", markersize=4)
    axes[0].axhline(0.7, color="r", linestyle="--", alpha=0.3, label="target d_max")
    axes[0].axhline(0.3, color="orange", linestyle="--", alpha=0.3, label="target d_mean")
    axes[0].set_xlabel("Load δ")
    axes[0].set_ylabel("Damage")
    axes[0].set_title("Evolution of d_max & d_mean")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (2) 标准差 + 定位指标
    axes[1].plot(loads, d_std_list, "d-", linewidth=2, color="purple", label="std(d)")
    axes[1].set_xlabel("Load δ")
    axes[1].set_ylabel("Std(d)")
    axes[1].set_title("Std(d) (crack contrast)")
    axes[1].grid(True, alpha=0.3)

    ax2 = axes[1].twinx()
    ax2.plot(loads, loc_list, "g--", linewidth=2, label="loc_index")
    ax2.set_ylabel("loc_index (d_notch / d_far)")
    axes[1].legend(loc="upper left")

    # (3) 最后一步 d 分布
    with torch.no_grad():
        d_final_all = solver.d_net(x_domain).numpy().flatten()
    axes[2].hist(d_final_all, bins=50, edgecolor="black", alpha=0.7)
    axes[2].axvline(d_final_all.mean(), color="r", linestyle="--",
                    label=f"mean={d_final_all.mean():.3f}")
    axes[2].axvline(d_final_all.max(), color="orange", linestyle="--",
                    label=f"max={d_final_all.max():.3f}")
    axes[2].set_xlabel("Damage d")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Final Damage Distribution")
    axes[2].legend()

    plt.tight_layout()
    stats_path = os.path.join(output_dir, "stats_with_notch.png")
    plt.savefig(stats_path, dpi=150)
    plt.close()
    print(f"  Statistics saved to: {stats_path}")

    # 诊断
    print("\n" + "=" * 70)
    print("  Final Diagnosis")
    print("=" * 70)
    final = history[-1]

    criterion_1 = final["d_max"] > 0.7
    criterion_2 = final["d_mean"] < 0.3
    criterion_3 = final["d_std"] > 0.2

    print(
        f"  d_max:   {final['d_max']:.4f}  "
        f"{'✓' if criterion_1 else '✗'} (target > 0.7)"
    )
    print(
        f"  d_mean:  {final['d_mean']:.4f}  "
        f"{'✓' if criterion_2 else '✗'} (target < 0.3)"
    )
    print(
        f"  d_std:   {final['d_std']:.4f}  "
        f"{'✓' if criterion_3 else '✗'} (target > 0.2)"
    )
    print(f"  loc_idx: {final['loc_index']:.2f}  (越大裂纹越局部化)")

    success = criterion_1 and criterion_2 and criterion_3

    if success:
        print("\n  🎉 SUCCESS! Crack localized!")
        print("     Phase 1 OK，可以进入 Phase 2 (X-RAS-PINN)")
    else:
        print("\n  ⚠️  需要继续调参：")
        if not criterion_1:
            print(
                f"     → d_max 偏低: 可以尝试增大 max_displacement "
                f"或减小 G_c"
            )
        if not criterion_2:
            print(
                "     → d_mean 偏高: 可以尝试增大 G_c 或减小 l，"
                "让裂纹更集中"
            )
        if not criterion_3:
            print("     → d_std 偏低: 裂纹过于弥散，尝试减小 l 或调整 notch 初始化")

    print("\n生成的文件:")
    print(f"  - {sampling_path}")
    print(f"  - {result_path}")
    print(f"  - {stats_path}")

    return solver, history


# ===========================
# main
# ===========================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SENT with Notch - Phase-field VPINN")
    print("=" * 70)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"DEBUG_MODE = {DEBUG_MODE}")
    print(f"输出将保存到: {os.path.join(os.getcwd(), 'outputs')}")

    try:
        input("\n按 Enter 开始 (或在无交互环境下自动继续)...")
    except EOFError:
        print("\n自动开始...")

    try:
        solver, history = test_sent_with_notch()
        print("\n" + "=" * 70)
        print("  测试完成!")
        print("=" * 70)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
