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


def get_notch_band_mask(xy: torch.Tensor, config: dict) -> torch.Tensor:
    """
    统一的 notch band 掩码

    定义：x <= notch_length AND |y - H/2| <= notch_seed_radius

    这与 FE 中的定义完全一致，所有地方必须使用同一定义！
    """
    x = xy[:, 0]
    y = xy[:, 1]

    notch_length = float(config["notch_length"])
    H = float(config["H"])
    rho = float(config.get("notch_seed_radius", 0.025))
    y_center = H / 2.0

    mask = (x <= notch_length) & (torch.abs(y - y_center) <= rho)
    return mask


def get_far_region_mask(xy: torch.Tensor, config: dict) -> torch.Tensor:
    """
    远场区域掩码（正确版本）

    条件：
    1. 距离裂尖 > far_region_radius
    2. 且不在 notch band 内（关键修复！）
    """
    notch_length = float(config["notch_length"])
    H = float(config["H"])
    far_radius = float(config.get("far_region_radius", 0.25))

    tip = torch.tensor([notch_length, H / 2.0])
    distances = torch.norm(xy - tip, dim=1)

    far_by_distance = distances > far_radius
    notch_band = get_notch_band_mask(xy, config)

    # 关键：两个条件都满足才算远场
    far_region = far_by_distance & (~notch_band)

    return far_region


# ============================================================================
# [修复] 三类点集生成（清晰分离）
# ============================================================================

def generate_all_points(config):
    """
    统一生成所有点集（清晰分离版）

    返回三类互不重叠的点集：
    1. x_notch_line: 覆盖整条 notch band，用于 loss_line (d→1)
    2. x_tip: 裂尖小圆区域，用于 loss_tip (d→0.85，可选)
    3. x_domain: 实体域（避开 notch band），用于 loss_energy + loss_far
    4. x_bc: 边界点
    """
    L = float(config["L"])
    H = float(config["H"])
    a = float(config["notch_length"])
    rho = float(config["notch_seed_radius"])
    n_domain = int(config["n_domain"])
    n_bc = int(config["n_bc"])

    y0 = H / 2.0

    # ================================================================
    # 1. x_notch_line: 覆盖整条 notch band
    # ================================================================
    # 关键：必须足够密！沿 x 均匀 + y 在 baçnd 内抖动
    n_notch = max(400, int(a / rho * 10 ))

    # 网格化
    n_x = int(np.sqrt(n_notch) * 2)
    n_y = max(5, int(np.sqrt(n_notch) / 2))

    xs_grid = np.linspace(0, a, n_x)
    ys_grid = np.linspace(y0 - rho, y0 + rho, n_y)
    XX, YY = np.meshgrid(xs_grid, ys_grid)
    x_notch_grid = np.stack([XX.flatten(), YY.flatten()], axis=1)

    # 加一些随机点
    n_rand = n_notch - len(x_notch_grid)
    if n_rand > 0:
        xs_rand = np.random.uniform(0, a, n_rand)
        ys_rand = y0 + np.random.uniform(-rho, rho, n_rand)
        x_notch_rand = np.stack([xs_rand, ys_rand], axis=1)
        x_notch_line = np.vstack([x_notch_grid, x_notch_rand])
    else:
        x_notch_line = x_notch_grid

    print(f"  [Points] x_notch_line: {len(x_notch_line)} points (notch band)")

    # ================================================================
    # 2. x_tip: 裂尖小圆区域（独立于 x_domain）
    # ================================================================
    r_tip = 2.5 * rho
    n_tip = 800

    tip_center = np.array([a, y0])

    r_samples = np.sqrt(np.random.uniform(0, 1, n_tip * 2)) * r_tip
    theta_samples = np.random.uniform(0, 2 * np.pi, n_tip * 2)

    xs_tip = tip_center[0] + r_samples * np.cos(theta_samples)
    ys_tip = tip_center[1] + r_samples * np.sin(theta_samples)

    # 过滤：在域内，且不在 notch band 内
    valid = (xs_tip >= 0) & (xs_tip <= L) & (ys_tip >= 0) & (ys_tip <= H)
    in_notch = (xs_tip <= a) & (np.abs(ys_tip - y0) <= rho)
    valid = valid & (~in_notch)

    x_tip = np.stack([xs_tip[valid][:n_tip], ys_tip[valid][:n_tip]], axis=1)

    print(f"  [Points] x_tip: {len(x_tip)} points (tip region, r < {r_tip:.4f})")

    # ================================================================
    # 3. x_domain: 实体域（避开 notch band）
    # ================================================================
    n_uniform = int(n_domain * 0.85)
    n_near_tip = n_domain - n_uniform

    # 3.1 全局均匀
    x_uniform = []
    while len(x_uniform) < n_uniform:
        x = np.random.uniform(0, L)
        y = np.random.uniform(0, H)
        if x <= a and abs(y - y0) <= rho:
            continue
        x_uniform.append([x, y])
    x_uniform = np.array(x_uniform)

    # 3.2 裂尖外围（r_tip < r < 2*r_tip 的环形区域）
    r_outer = 2 * r_tip
    x_near_tip = []
    attempts = 0
    while len(x_near_tip) < n_near_tip and attempts < n_near_tip * 10:
        r = np.random.uniform(r_tip, r_outer)
        theta = np.random.uniform(0, 2 * np.pi)
        x = tip_center[0] + r * np.cos(theta)
        y = tip_center[1] + r * np.sin(theta)
        attempts += 1

        if not (0 <= x <= L and 0 <= y <= H):
            continue
        if x <= a and abs(y - y0) <= rho:
            continue
        x_near_tip.append([x, y])

    x_near_tip = np.array(x_near_tip) if x_near_tip else np.empty((0, 2))
    x_domain = np.vstack([x_uniform, x_near_tip]) if len(x_near_tip) > 0 else x_uniform

    print(f"  [Points] x_domain: {len(x_domain)} points (uniform: {len(x_uniform)}, near_tip: {len(x_near_tip)})")

    # ================================================================
    # 4. x_bc: 边界点
    # ================================================================
    n_bc_half = n_bc // 2

    x_bottom = np.linspace(0, L, n_bc_half)
    bc_bottom = np.stack([x_bottom, np.zeros_like(x_bottom)], axis=1)

    x_top = np.linspace(0, L, n_bc_half)
    bc_top = np.stack([x_top, np.full_like(x_top, H)], axis=1)

    x_bc = np.vstack([bc_bottom, bc_top])

    # 转为 tensor
    x_domain = torch.tensor(x_domain, dtype=torch.float32, requires_grad=True)
    x_notch_line = torch.tensor(x_notch_line, dtype=torch.float32, requires_grad=True)
    x_tip = torch.tensor(x_tip, dtype=torch.float32, requires_grad=True)
    x_bc = torch.tensor(x_bc, dtype=torch.float32)

    return x_domain, x_notch_line, x_tip, x_bc


# ============================================================================
# [修复] Notch 初始化（使用 x_notch_line）
# ============================================================================

def initialize_notch_damage(d_net, x_domain, x_notch_line, x_tip, config):
    """
    修复版 notch 初始化

    核心改动：
    - loss_line 在 x_notch_line 上计算，目标 d=1
    - loss_tip 在 x_tip 上计算，目标 d=0.85（不是 1！）
    - loss_far 在 x_domain 远场计算，目标 d=0
    - 三个损失互不干扰
    """
    notch_length = config["notch_length"]
    H = config["H"]
    initial_d = config["initial_d"]
    rho = config["notch_seed_radius"]
    n_epochs = config["notch_init_epochs"]

    tip_pos = torch.tensor([notch_length, H / 2.0])

    # ========================================
    # 构建目标场
    # ========================================

    # 1) x_notch_line 目标：全部为 1.0
    d_target_notch = torch.ones((x_notch_line.shape[0], 1), dtype=torch.float32)

    # 2) x_tip 目标：0.85（平滑过渡，不要设 1！）
    d_target_tip = torch.full((x_tip.shape[0], 1), 0.85, dtype=torch.float32)

    # 3) x_domain 目标：tip 附近高斯衰减，远场为 0
    distances_domain = torch.norm(x_domain - tip_pos, dim=1)
    d_target_domain = initial_d * torch.exp(-(distances_domain / rho) ** 2)
    d_target_domain = d_target_domain.unsqueeze(1).clamp(0.0, 1.0)

    # 远场强制为 0
    cut_radius = 2.5 * rho
    far_domain = distances_domain > cut_radius
    d_target_domain[far_domain] = 0.0

    d_target_notch = d_target_notch.detach()
    d_target_tip = d_target_tip.detach()
    d_target_domain = d_target_domain.detach()

    print("\n  初始化 notch 损伤种子 (修复版):")
    print(f"    x_notch_line: {x_notch_line.shape[0]} points, target=1.0")
    print(f"    x_tip:        {x_tip.shape[0]} points, target=0.85")
    print(f"    x_domain:     {x_domain.shape[0]} points, gaussian decay")
    print(f"    tip 位置: ({notch_length:.2f}, {H / 2:.2f})")

    optimizer = torch.optim.Adam(d_net.parameters(), lr=5e-4)

    best_loss = float("inf")
    patience = 0

    print(f"    训练 d_net 拟合目标（{n_epochs} epochs）...")

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # 预测
        d_pred_notch = d_net(x_notch_line)
        d_pred_tip = d_net(x_tip)
        d_pred_domain = d_net(x_domain)

        # 损失1: x_notch_line 上必须 = 1（最重要！）
        loss_line = 5.0 * torch.mean((d_pred_notch - d_target_notch) ** 2)

        # 损失2: x_tip 上 = 0.85（平滑过渡）
        loss_tip = 1.0 * torch.mean((d_pred_tip - d_target_tip) ** 2)

        # 损失3: x_domain MSE
        loss_domain = torch.mean((d_pred_domain - d_target_domain) ** 2)

        # 损失4: 远场抑制（必须排除 notch band，但 x_domain 已避开）
        if far_domain.sum() > 0:
            loss_far = 2.0 * torch.mean(d_pred_domain[far_domain] ** 2)
        else:
            loss_far = torch.tensor(0.0)

        loss = loss_line + loss_tip + loss_domain + loss_far

        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            with torch.no_grad():
                d_max = d_pred_domain.max().item()
                d_mean = d_pred_domain.mean().item()
                d_notch_mean = d_pred_notch.mean().item()
                d_tip_mean = d_pred_tip.mean().item()
            print(f"      Epoch {epoch:4d}: loss={loss.item():.4e} | "
                  f"notch={d_notch_mean:.3f}, tip={d_tip_mean:.3f}, "
                  f"d_max={d_max:.3f}, d_mean={d_mean:.3f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            if patience > 200 and epoch > 500:
                print(f"      Early stopping at epoch {epoch}")
                break

    with torch.no_grad():
        d_final_notch = d_net(x_notch_line)
        d_final_domain = d_net(x_domain)

    print("\n    ✓ 初始化完成:")
    print(f"      notch_line mean: {d_final_notch.mean().item():.3f} (target=1.0)")
    print(f"      domain d_max:    {d_final_domain.max().item():.3f}")
    print(f"      domain d_mean:   {d_final_domain.mean().item():.3f}")

    return d_net


# ============================================================================
# [修复] 损失计算器
# ============================================================================

class NotchLossComputer:
    """
    清晰的 notch 相关损失计算器

    - loss_line: 在 x_notch_line 上，d → 1
    - loss_tip: 在 x_tip 上，d → 0.85（可选）
    - loss_far: 在 x_domain 远场，d → 0
    """

    def __init__(self, config):
        self.config = config
        self.notch_length = float(config["notch_length"])
        self.H = float(config["H"])
        self.rho = float(config.get("notch_seed_radius", 0.025))
        self.far_radius = float(config.get("far_region_radius", 0.25))

    def compute_loss_line(self, d_net, x_notch_line, weight=500.0):
        """Notch 带约束：d → 1"""
        d_pred = d_net(x_notch_line)
        return weight * torch.mean((d_pred - 1.0) ** 2)

    def compute_loss_tip(self, d_net, x_tip, weight=50.0, target=0.85):
        """裂尖平滑：d → 0.85（可选）"""
        if x_tip is None or len(x_tip) == 0:
            return torch.tensor(0.0)
        d_pred = d_net(x_tip)
        return weight * torch.mean((d_pred - target) ** 2)

    def compute_loss_far(self, d_net, x_domain, weight=100.0):
        """远场抑制：d → 0（正确排除 notch band）"""
        far_region = get_far_region_mask(x_domain, self.config)

        if far_region.sum() == 0:
            return torch.tensor(0.0)

        d_far = d_net(x_domain[far_region])
        return weight * torch.mean(d_far ** 2)


# ===========================
# 边界条件
# ===========================
def get_bc_function_sent(config):
    """拉伸：下边固定，上边 y 向位移 = load_value"""
    H = config["H"]

    def get_bc(load_value, x_bc):
        n_bc = x_bc.shape[0]
        u_bc = torch.zeros(n_bc, 2)
        u_bc[: n_bc // 2, :] = 0.0
        u_bc[n_bc // 2:, 0] = 0.0
        u_bc[n_bc // 2:, 1] = load_value
        return u_bc

    return get_bc

# ===========================
# 可视化点集（调试用）
# ===========================
def visualize_point_sets(x_domain, x_notch_line, x_tip, config, save_path):
    """可视化三类点集"""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(x_domain[:, 0].detach().numpy(),
               x_domain[:, 1].detach().numpy(),
               c='blue', s=1, alpha=0.3, label=f'x_domain ({len(x_domain)})')

    ax.scatter(x_notch_line[:, 0].detach().numpy(),
               x_notch_line[:, 1].detach().numpy(),
               c='red', s=8, alpha=0.8, label=f'x_notch_line ({len(x_notch_line)})')

    if len(x_tip) > 0:
        ax.scatter(x_tip[:, 0].detach().numpy(),
                   x_tip[:, 1].detach().numpy(),
                   c='green', s=10, alpha=0.8, label=f'x_tip ({len(x_tip)})')

    a = config["notch_length"]
    H = config["H"]
    rho = config.get("notch_seed_radius", 0.025)

    ax.plot([0, a, a, 0, 0],
            [H / 2 - rho, H / 2 - rho, H / 2 + rho, H / 2 + rho, H / 2 - rho],
            'k--', linewidth=2, label='Notch band')
    ax.plot(a, H / 2, 'k*', markersize=20, label='Crack tip')

    r_far = config.get("far_region_radius", 0.25)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(a + r_far * np.cos(theta), H / 2 + r_far * np.sin(theta),
            'orange', linestyle='--', linewidth=1, label=f'Far boundary (r={r_far})')

    ax.set_xlim(-0.05, config["L"] + 0.05)
    ax.set_ylim(-0.05, config["H"] + 0.05)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('Point Sets: x_domain (blue), x_notch_line (red), x_tip (green)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Viz] Point sets saved to: {save_path}")

# ===========================
# 主测试函数
# ===========================
def test_sent_with_notch(debug=False, config = None):

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
        config = create_config(debug= False)

    print_config(config)

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

    # 设置随机种子
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 2. 采样点(三类点集)
    print("\n[2/7] Generating sampling points (concentrated near notch)...")
    x_domain, x_notch_line, x_tip, x_bc = generate_all_points(config)

    print(f"  Total points:")
    print(f"    x_domain:     {x_domain.shape[0]}")
    print(f"    x_notch_line: {x_notch_line.shape[0]}")
    print(f"    x_tip:        {x_tip.shape[0]}")
    print(f"    x_bc:         {x_bc.shape[0]}")

    # 保存点集可视化
    point_sets_path = os.path.join(output_dir, "point_sets.png")
    visualize_point_sets(x_domain, x_notch_line, x_tip, config, point_sets_path)

    # 保存采样点图（兼容旧版）
    plt.figure(figsize=(8, 6))
    pts = x_domain.detach().numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.3, c='blue', label='x_domain')
    plt.scatter(x_notch_line[:, 0].detach().numpy(), x_notch_line[:, 1].detach().numpy(),
                s=5, alpha=0.8, c='red', label='x_notch_line')
    plt.scatter(config["notch_length"], config["H"] / 2, s=100, c="black", marker="*", label="Tip")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sampling Points Distribution")
    plt.legend()
    plt.axis('equal')
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
    d_net = initialize_notch_damage(d_net, x_domain, x_notch_line, x_tip, config)

    # 5. 求解器
    print("\n[5/7] Creating solver...")
    solver = PhaseFieldSolver(config, u_net, d_net)

    # 创建损失计算器
    notch_loss_computer = NotchLossComputer(config)


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
    # Zero-load relaxation (预热位移场)
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

    # 这里定义了哪里是“远场” (far_region)
    # 凡是距离裂尖大于config中定义的半径的点，都算远场

    far_region = get_far_region_mask(x_domain, config)
    print(f"  [Info] far_region: {far_region.sum().item()}/{len(far_region)} points")

    # 验证：far_region 与 notch band 无重叠
    notch_in_domain = get_notch_band_mask(x_domain, config)
    overlap = (far_region & notch_in_domain).sum().item()
    print(f"  [Check] far_region ∩ notch_band = {overlap} (should be 0)")

    if solver.d_prev is None:
        with torch.no_grad():
            solver.d_prev = solver.d_net(x_domain).detach().clone()

    with torch.no_grad():
        d_prev_global = solver.d_net(x_domain).detach().clone()


    with torch.no_grad():
        d_prev_global = solver.d_net(x_domain).detach().clone()


    # 训练主循环
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
            # L_irrev = torch.tensor(0.0, device=solver.device)

            # === Phase 2: 更新 d（冻结 u）===
            for p in solver.d_net.parameters(): p.requires_grad = True
            for p in solver.u_net.parameters(): p.requires_grad = False

            for _ in range(Kd):
                solver.optimizer_d.zero_grad()

                # 1. 能量 Loss (包含硬约束 max)
                L_energy_d = solver.drm_loss.compute_energy_loss(
                    x_domain, solver.u_net, solver.d_net, d_prev=d_prev_step
                )

                # 2. [修复] Notch Line Loss（使用 x_notch_line）
                L_notch = notch_loss_computer.compute_loss_line(
                    solver.d_net, x_notch_line,
                    weight=float(config.get("notch_hold_weight", 500.0))
                )

                ## 2. 不可逆 Loss (Soft Constraint)
                # # 【报错修复点】：这里传入 solver.d_prev，而不是未定义的 d_prev
                # L_irrev = solver.drm_loss.compute_irreversibility_loss(
                #     x_domain, solver.d_net, solver.d_prev, config["weight_irrev_phase1"]
                # )
                # L_irrev = torch.tensor(0.0, device=solver.device)

                # 3. [修复] Far Loss（正确排除 notch band）
                L_far = notch_loss_computer.compute_loss_far(
                    solver.d_net, x_domain, weight=100.0
                )

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

                    d_tip_raw = solver.d_net(x_tip)
                    d_line_raw = solver.d_net(x_notch_line)
                    d_tip_raw_val = d_tip_raw.mean().item()
                    d_line_raw_val = d_line_raw.mean().item()

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
                    f"line_raw_val={d_line_raw_val:.3f}, tip_raw_val={d_tip_raw_val:.3f}"
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
        d_line_f = solver.d_net(x_notch_line).mean().item()
        d_tip_f = solver.d_net(x_tip).mean().item()

        # 计算 loc_index (避免除以0)
        loc_index_f = d_line_f / (d_far_f + 1e-6) if d_far_f > 0 else 0.0

        print(f"  [End of Step {n + 1}] History updated. New Max: {d_max_f:.4f}, Loc: {loc_index_f:.1f}")

        # 记录统计信息 (使用更新后的 history/d_phys)
        history.append({
            "step": n,
            "load": load_value,
            "d_max_phy": d_max_f,
            "d_mean": d_mean_f,
            "d_std": d_std_f,
            "d_tip": d_tip_raw_val,
            "d_line": d_line_f,
            "d_tip": d_tip_f,
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