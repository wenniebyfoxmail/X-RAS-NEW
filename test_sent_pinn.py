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
from pathlib import Path # 引入 Path 对象方便操作路径
import json
import os
import datetime


def save_experiment_log(output_dir, config, history=None):
    """
    将实验配置和最终结果保存为 JSON 和 TXT 文件
    """
    # 1. 保存完整 Config 为 JSON (方便程序读取)
    json_path = os.path.join(output_dir, "experiment_config.json")
    # 把 Tensor 或特殊对象转为字符串，防止 json 报错
    config_serializable = {k: (str(v) if isinstance(v, torch.Tensor) else v) for k, v in config.items()}

    with open(json_path, 'w') as f:
        json.dump(config_serializable, f, indent=4)

    # 2. 保存易读的 Summary TXT (方便人看)
    txt_path = os.path.join(output_dir, "experiment_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("========================================\n")
        f.write(f"Experiment Summary\n")
        f.write("========================================\n")
        f.write(f"Domain Points: {config['n_domain']}\n")
        f.write(f"G_c:           {config['G_c']}\n")
        f.write(f"length scale:  {config['l']}\n")
        f.write(f"Max Load:      {config['max_displacement']}\n")
        f.write(f"Notch Radius:  {config['notch_seed_radius']}\n")
        if history:
            f.write("----------------------------------------\n")
            f.write(f"Final Step:    {len(history)}\n")
            f.write(f"Final d_mean:  {history[-1]['d_mean']:.4f}\n")
            f.write(f"Final d_max:   {history[-1]['d_max']:.4f}\n")

    print(f"  [Log] Experiment config saved to: {output_dir}")
# ===========================
# 确保模块可导入
# ===========================
# 尝试将当前脚本的父目录加入路径，以导入其他模块
sys.path.insert(0, str(Path(__file__).parent))
# ===========================


# 导入统一配置
# ===========================
try:
    from config import create_config, print_config
except ImportError:
    print("错误: 找不到 config.py")
    print("请确保 config.py 在当前目录下。")
    sys.exit(1)


# ===========================
# 全局开关：调试模式
# ===========================
DEBUG_MODE = False   # True=快速测试；False=精细实验


# ===========================
# 输出路径
# ===========================
def get_output_dir():
    """获取输出目录: ./outputs"""
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ===========================
    # 导入 Phase-1 / Phase-2 桥接模块
# ===========================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import phase1_phase2_bridge
    from phase1_phase2_bridge import save_phase1_checkpoint, load_phase1_checkpoint
    print(f"  [Info] 成功导入桥接模块: {phase1_phase2_bridge.__file__}")
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"  ⚠️  ImportError: {e}")
    print(f"  ⚠️  当前 sys.path: {sys.path}")
    print("  ⚠️  phase1_phase2_bridge.py 不存在或无法导入，无法保存/加载检查点")
    BRIDGE_AVAILABLE = False
except Exception as e:
    print(f"  ⚠️  导入时发生意外错误: {e}")
    BRIDGE_AVAILABLE = False


# ===========================
# 导入相场 VPINN 核心模块
# ===========================
try:
    from solver_pinn import (
        DisplacementNetwork,
        DamageNetwork,
        PhaseFieldSolver,
        generate_domain_points,
        visualize_solution,
        compute_strain,
        compute_energy_split,
    )
except ImportError:
    print("错误: 找不到 solver_pinn.py")
    print("请确保 solver_pinn.py 在当前目录下。")
    sys.exit(1)


# ===========================
# 采样点生成（SENT + notch 加密）
# ===========================

def generate_notch_line_points(config, n_notch: int = 300):
    """
    生成 notch line/band 上的点（带宽 notch_seed_radius），用于 d=1 约束损失。
    注意：x_domain 会避开 notch band，所以必须单独生成 x_notch。
    """

    L = float(config["L"])
    H = float(config["H"])
    a = float(config["notch_length"])
    rho = float(config["notch_seed_radius"])
    y0 = H / 2.0

    xs = np.random.uniform(0.0, a, size=n_notch)
    ys = y0 + np.random.uniform(-rho, rho, size=n_notch)
    pts = np.stack([xs, ys], axis=1)
 
    return torch.tensor(pts, dtype=torch.float32, requires_grad=True)


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

    # ------------------------------------------------
    # 1. 局部加密：只针对初始裂尖 (Local Refinement at Tip)
    # ------------------------------------------------
    # 即使不知道裂纹去哪，我们肯定知道它从尖端开始。
    # 这里分配 10% ~ 20% 的点用于捕捉起裂瞬间。
    n_tip = int(n_domain * 0.15)
    radius_tip = 0.05  # 局部加密半径 (只覆盖尖端周围一小圈)

    x_tip_list = []
    for _ in range(n_tip):
        # 在圆内随机撒点
        r = np.random.uniform(0, radius_tip)
        theta = np.random.uniform(0, 2 * np.pi)
        x = notch_tip[0] + r * np.cos(theta)
        y = notch_tip[1] + r * np.sin(theta)

        # 边界检查
        if 0 <= x <= L and 0 <= y <= H:
            # 还要避开 Notch 内部空洞
            if not (x <= notch_length and abs(y - H / 2) <= config["notch_seed_radius"]):
                x_tip_list.append([x, y])

    # ------------------------------------------------
    # 2. 全场均匀采样：背景 (Global Uniform)
    # ------------------------------------------------
    # 剩下的点全部均匀撒在整个矩形里。
    # 这是最“诚实”的做法，不假设任何路径。
    n_uniform = n_domain - len(x_tip_list)

    x_uniform_list = []
    while len(x_uniform_list) < n_uniform:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)

        # 避开 Notch 内部空洞
        if x <= notch_length and abs(y - H / 2) <= config.get("notch_seed_radius", 0.01):
            continue

        x_uniform_list.append([x, y])

    # 合并
    x_domain = np.vstack((x_tip_list, x_uniform_list))

    # # 70% 均匀 + 30% notch 附近
    # n_uniform = int(n_domain * 0.7)
    # n_concentrated = n_domain - n_uniform
    #
    # x_domain_list = []
    #
    # # 1) 均匀采样（略避开 notch 凹槽）
    # while len(x_domain_list) < n_uniform:
    #     x = np.random.uniform(0, L)
    #     y = np.random.uniform(0, H)
    #
    #     notch_band = float(config.get("notch_seed_radius", 0.01))
    #     if x <= notch_length and abs(y - H / 2) <= notch_band:
    #         continue
    #
    #     x_domain_list.append([x, y])
    #
    # # 2) notch 尖端附近加密
    # radius_local = 0.02
    # for _ in range(n_concentrated):
    #     angle = np.random.uniform(0, 2 * np.pi)
    #     r = np.random.uniform(0, radius_local)
    #
    #     x = notch_tip[0] + r * np.cos(angle)
    #     y = notch_tip[1] + r * np.sin(angle)
    #
    #     if 0 <= x <= L and 0 <= y <= H:
    #         notch_band = float(config.get("notch_seed_radius", 0.01))
    #         if not (x <= notch_length and abs(y - H / 2) <= notch_band):
    #             x_domain_list.append([x, y])
    #
    # x_domain = torch.tensor(x_domain_list, dtype=torch.float32)
    #
    # 边界点：下边固定，上边施加位移
    n_bc_half = n_bc // 2
    x_bottom = np.linspace(0, L, n_bc_half)
    y_bottom = np.zeros_like(x_bottom)
    bc_bottom = np.stack([x_bottom, y_bottom], axis=1)

    x_top = np.linspace(0, L, n_bc_half)
    y_top = np.ones_like(x_top) * H
    bc_top = np.stack([x_top, y_top], axis=1)

    x_bc = torch.tensor(np.vstack([bc_bottom, bc_top]), dtype=torch.float32)

    return torch.tensor(x_domain, dtype=torch.float32, requires_grad=True), x_bc, x_notch


# ===========================
# notch 初始损伤种子
# ===========================
def initialize_notch_damage(d_net, x_domain, config):
    """
        规范 notch 初始化（与 FE 一致）：
        1) 线裂纹带：x<=a 且 |y-H/2|<=rho 处 d_target=1
        2) 裂尖平滑：在 tip 周围叠加 gaussian（可选）
        3) 远场压制：对非 line 区域在 r>cut_radius 时强制 0
    """

    notch_length = config["notch_length"]
    H = config["H"]
    initial_d = config["initial_d"]
    seed_radius = config["notch_seed_radius"]
    n_epochs = config["notch_init_epochs"]

    notch_tip = torch.tensor([notch_length, H / 2.0])
    x = x_domain[:, 0]
    y = x_domain[:, 1]
    y0 = H / 2.0

    # (1) line notch band
    line_mask = (x <= notch_length) & (torch.abs(y - y0) <= seed_radius)

    # (2) tip gaussian smoothing
    distances = torch.norm(x_domain - notch_tip, dim=1)
    d_gauss = initial_d * torch.exp(-(distances / seed_radius) ** 2)
    d_target = d_gauss.unsqueeze(1).clamp(0.0, 1.0)
    d_target[line_mask] = 1.0
    d_target = d_target.detach()

    # (3) far clamp ONLY outside the line region
    cut_radius = 1.5 * seed_radius
    far_mask = (distances > cut_radius) & (~line_mask)
    d_target[far_mask] = 0.0

    # (4) very close points near tip can be strengthened (optional)
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

    # 同步更新
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        d_pred = d_net(x_domain)

        loss_mse = torch.mean((d_pred - d_target) ** 2)

        tip_points = distances < seed_radius
        if tip_points.sum() > 0:
            loss_tip = torch.mean((d_pred[tip_points] - 0.95) ** 2)
        else:
            loss_tip = 0.0

        # line enforcement: keep d≈1 on the pre-crack band
        if line_mask.sum() > 0:
            loss_line = torch.mean((d_pred[line_mask] - 1.0) ** 2)
        else:
            loss_line = 0.0

        # IMPORTANT: far penalty must exclude the notch band, otherwise it will fight loss_line
        far_points = (distances > cut_radius) & (~line_mask)
        if far_points.sum() > 0:
            loss_far = torch.mean(d_pred[far_points] ** 2)
        else:
            loss_far = 0.0

        loss = loss_mse + 2.0 * loss_line + 1.0 * loss_tip + 2.0 * loss_far

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
def test_sent_with_notch(debug=False, config = None):
    """运行带 notch 的 SENT 相场测试"""

    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
    readable_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    base_output_dir = get_output_dir()
    experiment_name = f"Baseline_N{config['n_domain']}_Gc{config['G_c']}_{timestamp_str}"
    output_dir = os.path.join(base_output_dir, experiment_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"🚀 本次实验输出目录: {output_dir}")
    # 将时间戳存入 config，方便后续调用
    config["timestamp"] = readable_time
    config["run_id"] = timestamp_str

    save_experiment_log(output_dir, config)

    print(f"输出目录: {output_dir}")

    print("=" * 70)
    print("  SENT Test with Notch Initialization")
    print("=" * 70)

    """
        Phase-1 主入口
        Args:
            debug: 是否调试模式
            config: 可选的配置字典。如果为None，内部创建。
        """

    # 1. 配置
    print("\n[1/7] Loading configuration...")
    if config is None:
        from config import create_config
        config = create_config(debug=debug)

    print_config(config)

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 2. 采样点
    print("\n[2/7] Generating sampling points (concentrated near notch)...")
    x_domain, x_bc = generate_sent_with_notch_points(config)

    print(f"  Domain points: {x_domain.shape[0]}")

    # notch band points (must exist, since x_domain avoids notch band)
    x_notch = generate_notch_line_points(config, n_notch=int(config.get("n_notch", 400)))

    # 保存采样点图
    plt.figure(figsize=(6, 4))
    pts = x_domain.detach().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.5)
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

    # Use a seed set that actually includes the notch band
    x_seed = torch.cat([x_domain, x_notch], dim=0)
    d_net = initialize_notch_damage(d_net, x_seed, config)

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

    # ================================================================
    # 【关键修复】Zero-load relaxation (预热位移场)
    # ================================================================
    print("\n[关键修复] Zero-load relaxation (预热位移场)...")

    # 1) 构造零载荷边界条件（上下边界全部为零位移）
    get_bc_zero = get_bc_function_sent(config)
    u_bc_zero = get_bc_zero(0.0, x_bc)

    # 2) 冻结 d_net —— 保护 notch 初始化
    for p in solver.d_net.parameters():
        p.requires_grad = False

    # 3) 预热训练：只训练 u_net，让其趋近于零位移场
    n_relax = 400   # 300~600 都可以
    for epoch in range(n_relax):
        solver.optimizer_u.zero_grad()

        L_energy = solver.drm_loss.compute_energy_loss(
            x_domain, solver.u_net, solver.d_net, d_prev=solver.d_prev
        )
        L_bc = solver.drm_loss.compute_bc_loss(
            x_bc, u_bc_zero, solver.u_net, weight=200.0
        )
        loss = L_energy + L_bc
        loss.backward()
        solver.optimizer_u.step()

        if epoch % 100 == 0 or epoch == n_relax - 1:
            print(f"  [Relax] Epoch {epoch:4d} | Loss={loss.item():.3e}, "
                  f"E={L_energy.item():.2e}, BC={L_bc.item():.2e}")

    # 4) 解冻 d_net
    for p in solver.d_net.parameters():
        p.requires_grad = True

    print("  ✓ 预热完成：u_net 已接近物理零载荷平衡态\n")
    # ================================================================


    # notch 区域掩码（用于统计 & notch 保持损失）
    notch_tip = torch.tensor([config["notch_length"], config["H"] / 2])
    distances_to_tip = torch.norm(x_domain - notch_tip, dim=1)

    # 这里定义了哪里是“远场” (far_region)
    # 凡是距离裂尖大于 0.25 (config中定义的半径) 的点，都算远场

    far_region = distances_to_tip > config["far_region_radius"]

    # Diagnostics regions: keep your tip/far metrics if you want,
    # but notch hold MUST be applied on x_notch (line band), not on x_domain.
    notch_tip = torch.tensor([config["notch_length"], config["H"] / 2])
    distances_to_tip = torch.norm(x_domain - notch_tip, dim=1)
    far_region = distances_to_tip > config["far_region_radius"]

    # 确保 solver.d_prev 已经初始化 (在 initialize_fields 中已完成)
    # 如果没有初始化，手动初始化一次
    if solver.d_prev is None:
        with torch.no_grad():
            solver.d_prev = solver.d_net(x_domain).detach().clone()


    with torch.no_grad():
        d_prev_global = solver.d_net(x_domain).detach().clone()

    for n, load_value in enumerate(loading_steps):
        print("\n" + "=" * 60)
        print(f"Step {n + 1}/{len(loading_steps)} | Load = {load_value:.6f}")
        print("=" * 60)

        d_prev_step = d_prev_global.detach().clone()
        # 更新边界条件
        u_bc = get_bc(load_value, x_bc)

        # -----------------------------------------------------------
        # [Step A 修复] 历史场准备
        # 在开始这一步训练前，solver.d_prev 存储的是 "上一步结束时的 d"
        # 我们需要确保它在这一步的训练中保持不变（作为锚点），
        # 并且我们要用 detach() 确保没有梯度回传。
        # -----------------------------------------------------------

        # 调试打印：确认历史场的状态
        with torch.no_grad():
            hist_max = solver.d_prev.max().item()
            print(f"  [History] Start of Step {n + 1}: d_prev_max = {hist_max:.4f}")

        solver.u_net.train()
        solver.d_net.train()

        # 根据阶段设置 Epoch 数
        if n < config["n_epochs_switch"]:
            n_epochs = config["n_epochs_initial"]
        else:
            n_epochs = config["n_epochs_later"]


        # ====================
        #  Staggered Training
        # ====================
        Ku = config.get("stagger_u_steps", 200)

        # [Patch 1] Load=0 时，强制跳过损伤更新
        if n == 0 or load_value == 0.0:
            Kd = 0
            print("  [Info] Step 0 (Load=0): Skipping damage update (Kd=0).")
        else:
            Kd = config.get("stagger_d_steps", 100)

        for epoch in range(n_epochs):
            # === Phase 1: 更新 u（冻结 d）===
            for p in solver.d_net.parameters(): p.requires_grad = False
            for p in solver.u_net.parameters(): p.requires_grad = True

            for _ in range(Ku):
                solver.optimizer_u.zero_grad()

                # 传入 solver.d_prev 用于硬约束逻辑 (max)
                L_energy_u = solver.drm_loss.compute_energy_loss(
                    x_domain, solver.u_net, solver.d_net, d_prev=d_prev_step
                )
                L_bc_u = solver.drm_loss.compute_bc_loss(
                    x_bc, u_bc, solver.u_net, 200.0
                )
                loss_u = L_energy_u + L_bc_u
                loss_u.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(solver.u_net.parameters(), 1.0)
                solver.optimizer_u.step()

            loss_d = torch.tensor(0.0, device=solver.device)
            L_irrev = torch.tensor(0.0, device=solver.device)

            # === Phase 2: 更新 d（冻结 u）===
            for p in solver.d_net.parameters(): p.requires_grad = True
            for p in solver.u_net.parameters(): p.requires_grad = False

            for _ in range(Kd):
                solver.optimizer_d.zero_grad()

                # 1. 能量 Loss (包含硬约束 max)
                L_energy_d = solver.drm_loss.compute_energy_loss(
                    x_domain, solver.u_net, solver.d_net, d_prev=d_prev_step
                )

                # 2. 不可逆 Loss (Soft Constraint)
                # 【报错修复点】：这里传入 solver.d_prev，而不是未定义的 d_prev
                L_irrev = solver.drm_loss.compute_irreversibility_loss(
                    x_domain, solver.d_net, solver.d_prev, config["weight_irrev_phase1"]
                )
                L_irrev = torch.tensor(0.0, device=solver.device)

                # 3. Notch Loss (强力锚点)
                # 建议把 notch_hold_weight 设大，例如 5000.0
                d_notch_pred = solver.d_net(x_notch)
                notch_weight = float(config.get("notch_hold_weight", 5000.0))
                target_notch_d = float(config["notch_hold_target"])
                L_notch = notch_weight * torch.mean((d_notch_pred - target_notch_d) ** 2)

                # [新增] 远场抑制损失
                # 逻辑：如果在 far_region 里的点 d 不为 0，就罚款
                if far_region.sum() > 0:
                    # 1. 选出远场的点对应的预测损伤值
                    d_far_pred = solver.d_net(x_domain[far_region])

                    # 2. 给予一个权重 (建议和 Notch Weight 同量级，例如 100.0)
                    # 你的 config["notch_hold_weight"] 大概是 10.0~20.0，建议这里给大一点，比如 100.0
                    w_far = 100.0

                    # 3. 计算均方误差 (目标是 0)
                    L_far = w_far * torch.mean(d_far_pred ** 2)
                else:
                    L_far = torch.tensor(0.0, device=solver.device)

                loss_d = L_energy_d + L_notch + L_far # No more L_iir
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(solver.d_net.parameters(), 1.0)
                solver.optimizer_d.step()

            with torch.no_grad():
                d_raw_now = solver.d_net(x_domain).detach()
                d_prev_global = torch.max(d_prev_global, d_raw_now)

            # === 解冻 u_net (为下一轮做准备) ===
            for p in solver.u_net.parameters(): p.requires_grad = True

            # === 打印诊断 ===
            if epoch % 100 == 0 or epoch == n_epochs - 1:
                # 重新计算用于打印的 Loss
                with torch.no_grad():
                    # 必须手动再做一次 max 才能看到真实的物理状态
                    d_curr_raw = solver.d_net(x_domain)
                    d_phys = torch.max(d_curr_raw, solver.d_prev)

                    d_max = d_phys.max().item()
                    d_mean = d_phys.mean().item()
                    d_std = d_phys.std().item()

                    # 打印 raw 和 phys 的区别，帮助 Debug
                    d_raw_max = d_curr_raw.max().item()

                    d_notch_raw = solver.d_net(x_notch)
                    # d_notch_phys = torch.max(d_notch_raw, solver.d_prev(x_notch))
                    d_notch_raw_val = d_notch_raw.mean().item()
                    #d_notch_phys_val = d_notch_phys.mean().item()

                    # 历史场的状态
                    hist_mean = solver.d_prev.mean().item()
                    hist_max = solver.d_prev.max().item()
                    print(f"  [Diag] Step {n + 1} Start | History: mean={hist_mean:.4f}, max={hist_max:.4f}")

                    # 检查是否有全域扩散的迹象
                    if hist_mean > 0.25:  # 假设 0.25 是一个危险阈值
                        print("  [Warning] History field shows signs of global damage spreading!")

                print(
                    f"  Epoch {epoch:4d} | "
                    f"Loss_u={loss_u.item():.2e}, Loss_d={loss_d.item():.2e} | "
                    f"d_phys_max={d_max:.3f} (raw={d_raw_max:.3f}), "
                    f"d_mean={d_mean:.3f}, "
                    f"notch_raw_val={d_notch_raw_val:.3f},"
                    f"IrrLoss={L_irrev.item():.2e}"
                )

            # 在 test_sent_pinn.py 的训练循环中添加（每 500 epoch）
            if epoch % 500 == 0:
                # 1. 必须开启梯度才能计算应变 (compute_strain 需要 autograd)
                #    这里不需要 with torch.no_grad()，因为我们需要图来求导
                u_pred_diag = solver.u_net(x_domain)
                epsilon = compute_strain(u_pred_diag, x_domain)

                with torch.no_grad():
                    # epsilon = compute_strain(solver.u_net(x_domain), x_domain)
                    psi_plus, psi_minus = compute_energy_split(epsilon, config["E"], config["nu"])
                    d = solver.d_net(x_domain)

                    E_char = config["G_c"] / config["l"]

                    psi_ratio = psi_plus / E_char

                    print(f"\n  [Energy Diagnostics]")
                    print(f"    E_char (threshold) = {E_char:.2e}")
                    print(f"    psi_plus: mean={psi_plus.mean():.2e}, max={psi_plus.max():.2e}")
                    print(f"    psi_ratio (ψ⁺/E_char): mean={psi_ratio.mean():.3f}, max={psi_ratio.max():.3f}")
                    print(f"    > 1 means damage should grow")
                    print(f"    Points with psi_ratio > 1: {(psi_ratio > 1).sum().item()}")

        # ==========================================================
        # [Step A 关键] 每一步 Load 结束时，更新历史场
        # ==========================================================
        with torch.no_grad():
            # 获取当前这一步训练出来的 Raw Output
            d_current_step_raw = solver.d_net(x_domain)

            # 融合历史：新历史 = max(当前输出, 旧历史)
            # 这样保证了 solver.d_prev 永远单调递增，绝不回头
            d_new_history = torch.max(d_current_step_raw, solver.d_prev)

            # 更新 solver 内部状态
            solver.d_prev = d_new_history.detach().clone()

            print(f"  [End of Step {n + 1}] History updated. New Max: {solver.d_prev.max().item():.4f}")

        d_final_phys = solver.d_prev  # 使用物理值(max后)来统计

        d_max_f = d_final_phys.max().item()
        d_mean_f = d_final_phys.mean().item()
        d_std_f = d_final_phys.std().item()

        # 计算局部化指标 loc_index
        # 需要用到 far_region 掩码 (确保它在循环外已经定义好)
        if far_region.sum() > 0:
            d_far_f = d_final_phys[far_region].mean().item()
        else:
            d_far_f = 0.0

        # 获取 Notch 区域均值
        d_notch_f = solver.d_net(x_notch).mean().item()

        # 计算 loc_index (避免除以0)
        loc_index_f = d_notch_f / (d_far_f + 1e-6) if d_far_f > 0 else 0.0

        print(f"  [End of Step {n + 1}] History updated. New Max: {d_max_f:.4f}, Loc: {loc_index_f:.1f}")

        # 记录统计信息 (使用更新后的 history/d_phys)
        history.append({
            "step": n,
            "load": load_value,
            "d_max": d_max_f,
            "d_mean": d_mean_f,
            "d_std": d_std_f,
            "d_notch": d_notch_f,
            "d_far": d_far_f,
            "loc_index": loc_index_f,
        })

        print(
            f"Step summary: d_max={d_max:.4f}, "
            f"d_mean={d_mean:.4f}, d_std={d_std:.4f}"
        )

    # 7. 可视化
    print("\n[7/7] Visualization...")
    nx, ny = 150, 150
    x_grid = generate_domain_points(
        nx, ny, x_range=(0, config["L"]), y_range=(0, config["H"])
    )

    result_path = os.path.join(output_dir, "sent_with_notch.png")

    try:
        visualize_solution(solver, x_grid, nx, ny, save_path=result_path)
        plt.close('all')  # ✅ 强制关闭所有图窗，防止阻塞
    except Exception as e:
        print(f"  Visualization warning: {e}")

    print(f"  Damage field saved to: {result_path}")

    # 统计图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    loads = [h["load"] for h in history]
    d_max_list = [h["d_max"] for h in history]
    d_mean_list = [h["d_mean"] for h in history]
    d_std_list = [h["d_std"] for h in history]
    loc_list = [h["loc_index"] for h in history]
    info_str = (
        f"Baseline (Uniform) | Time: {readable_time}"
        f" Config: N={config['n_domain']} | Gc={config['G_c']} | l={config['l']} | Load={config['max_displacement']:.4f}"
    )


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
    plt.figtext(0.5, 0.01, info_str, ha="center", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 3})
    # 为了防止文字被切掉，调整一下底边距
    plt.subplots_adjust(bottom=0.15)

    stats_path = os.path.join(output_dir, "stats_with_notch.png",info_str)
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

    # ✅ 保存 Phase-1 检查点（供 Phase-2 使用）
    if BRIDGE_AVAILABLE:  # ✅ 使用全局标志控制是否执行
        try:
            checkpoint_path = save_phase1_checkpoint(solver, history, config)
            print(f"  - {checkpoint_path}")
        except Exception as e:
            print(f"  ⚠️  保存检查点失败: {e}")
    else:
        print("  ⚠️  phase1_phase2_bridge.py 不存在，跳过检查点保存")

    # [新增 5] 结束时更新 Log
    save_experiment_log(output_dir, config, history)

    return solver, history


# ===========================
# main
# ===========================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SENT with Notch - Phase-field PINN")
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