"""
scripts/analyze_fe_vs_pinn.py
独立后处理脚本：加载训练好的 X-RAS 模型，与 FE 真值进行对比
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================
# 1. 环境设置：添加根目录到路径
# ===========================
# 这样才能导入根目录下的 config 和 phase_field_vpinn
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

from config import create_config
from solver_pinn import DisplacementNetwork, DamageNetwork, generate_domain_points
from fe_baseline_utils import load_fe_2d, interpolate_fe_to_grid, plot_midline_comparison

# ===========================
# 2. 模型加载函数
# ===========================
def load_xras_model(checkpoint_path, device='cpu'):
    """加载 X-RAS 模型的权重"""
    print(f"正在加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 注意：这里假设你想加载的是 X-RAS 的组合解，或者其中一个子域
    # 简单起见，我们加载 Phase-1 的全局解，或者你需要修改这里来加载两个子网
    # 如果是 X-RAS，通常保存了 'u_net_1', 'u_net_2' 等。
    # 这里演示加载标准 VPINN 或 Phase-1 模型结构：

    u_net = DisplacementNetwork().to(device)
    d_net = DamageNetwork().to(device)

    # 尝试匹配键值
    if 'u_net_state_dict' in checkpoint:
        u_net.load_state_dict(checkpoint['u_net_state_dict'])
        d_net.load_state_dict(checkpoint['d_net_state_dict'])
    else:
        # 可能是 X-RAS 的复杂结构，这里需要根据你的 save 逻辑调整
        # 如果你之前没保存 X-RAS 的完整 checkpoint，可能需要去 test_sent_phase2 里加保存代码
        print("⚠️ Checkpoint 格式可能是 X-RAS 子域格式，尝试加载子域 1 作为演示...")
        u_net.load_state_dict(checkpoint['model_u1']) # 示例键名
        d_net.load_state_dict(checkpoint['model_d1'])

    u_net.eval()
    d_net.eval()
    return u_net, d_net

# ===========================
# 3. 主分析逻辑
# ===========================
def main():
    # A. 配置
    config = create_config(debug=False)
    device = torch.device("cpu")

    # B. 加载数据和模型
    fe_path = os.path.join(project_root, config["fe_2d_path"])
    ckpt_path = os.path.join(project_root, "outputs/phase1_checkpoint.pth") # 暂时用 Phase-1 演示

    # 1. 加载 FE 真值
    fe_pts, fe_d, _ = load_fe_2d(fe_path)
    if fe_pts is None: return

    # 2. 加载 PINN 模型
    try:
        u_net, d_net = load_xras_model(ckpt_path, device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # C. 执行对比
    print("\n开始生成对比图...")

    # 生成评估网格
    N = 200
    x_line = np.linspace(0, config['L'], N)
    y_line = np.linspace(0, config['H'], N)
    x_grid, y_grid = np.meshgrid(x_line, y_line)

    # 1. FE 插值
    _, _, d_fe_grid = interpolate_fe_to_grid(fe_pts, fe_d, x_line, y_line)

    # 2. PINN 预测
    pts_tensor = torch.tensor(np.stack([x_grid.flatten(), y_grid.flatten()], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        d_pinn_grid = d_net(pts_tensor).cpu().numpy().reshape(N, N)

    # 3. 计算误差
    diff = np.abs(d_fe_grid - d_pinn_grid)
    l2_err = np.sqrt(np.mean(diff**2))
    print(f"全场 L2 误差: {l2_err:.4e}")

    # D. 绘图 (三子图: FE, PINN, Diff)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # FE
    im0 = axes[0].contourf(x_grid, y_grid, d_fe_grid, levels=50, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title("FE Ground Truth")
    plt.colorbar(im0, ax=axes[0])

    # PINN
    im1 = axes[1].contourf(x_grid, y_grid, d_pinn_grid, levels=50, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"PINN Prediction")
    plt.colorbar(im1, ax=axes[1])

    # Diff
    im2 = axes[2].contourf(x_grid, y_grid, diff, levels=50, cmap='magma')
    axes[2].set_title(f"Absolute Error (L2={l2_err:.2e})")
    plt.colorbar(im2, ax=axes[2])

    save_path = os.path.join(project_root, "outputs/final_paper_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"✓ 最终论文图表已保存: {save_path}")

if __name__ == "__main__":
    main()