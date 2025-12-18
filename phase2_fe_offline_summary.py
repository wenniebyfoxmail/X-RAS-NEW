# phase2_fe_offline_summary.py
import torch
import numpy as np
from pathlib import Path

from solver_pinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver
)
from solver_xras import XRASPINNSolver, SubdomainModels
from fe_baseline_utils import (
    compute_global_L2_for_u_and_d,
    plot_fe_vs_pinn_midline,
)


def load_phase2_solvers_from_ckpt(ckpt_path: str, device: str = "cpu"):
    """从 phase2_raw_*.pth 里恢复 VPINN 和 X-RAS 的 solver（只用于预测，不再训练）"""
    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    config = dict(config)  # 防止某些配置是自定义类型
    config["device"] = device

    # ----- VPINN -----
    u_net_v = DisplacementNetwork()
    d_net_v = DamageNetwork()
    u_net_v.load_state_dict(ckpt["vpinn_state"]["u_net"])
    d_net_v.load_state_dict(ckpt["vpinn_state"]["d_net"])
    vpinn_solver = PhaseFieldSolver(config, u_net_v, d_net_v)

    # ----- X-RAS -----
    # 先实例化 4 个子网
    u_net_1 = DisplacementNetwork()
    d_net_1 = DamageNetwork()
    u_net_2 = DisplacementNetwork()
    d_net_2 = DamageNetwork()

    # 加载权重
    xras_state = ckpt["xras_state"]
    u_net_1.load_state_dict(xras_state["u_net_1"])
    d_net_1.load_state_dict(xras_state["d_net_1"])
    u_net_2.load_state_dict(xras_state["u_net_2"])
    d_net_2.load_state_dict(xras_state["d_net_2"])

    # 用 dataclass 正确构造 SubdomainModels
    models = SubdomainModels(
        u_net_1=u_net_1,
        d_net_1=d_net_1,
        u_net_2=u_net_2,
        d_net_2=d_net_2,
    )

    crack_center = torch.tensor(
        [config["notch_length"], config["H"] / 2.0],
        dtype=torch.float32,
    )
    r_sing = config.get("r_sing", 0.15)

    xras_solver = XRASPINNSolver(config, models, crack_center, r_sing=r_sing)

    return config, vpinn_solver, xras_solver, ckpt


def phase2_vs_fe_summary_offline(
    phase2_ckpt_path: str,
    fe_phase_path: str,
    out_npz_path: str | None = None,
    orientation: str = "horizontal",
):
    """
    离线 FE 对比：

    - 不再训练网络，只加载 phase2_raw_*.pth
    - 在 FE 节点上评估 VPINN / X-RAS
    - 输出：全域 L2、midline L2，以及两张 FE vs PINN 的中线对比图
    """
    device = "cpu"
    phase2_ckpt_path = Path(phase2_ckpt_path)
    fe_phase_path = Path(fe_phase_path)

    config, vpinn_solver, xras_solver, ckpt = load_phase2_solvers_from_ckpt(
        str(phase2_ckpt_path), device=device
    )

    print(f"[Offline FE Summary] 使用 FE 基准: {fe_phase_path}")
    print(f"  Phase-2 checkpoint: {phase2_ckpt_path}")

    # 1) 全域 L2 对比
    stats_v = compute_global_L2_for_u_and_d(vpinn_solver, str(fe_phase_path), verbose=False)
    stats_x = compute_global_L2_for_u_and_d(xras_solver, str(fe_phase_path), verbose=False)

    print("[Global L2 (damage d)]")
    print(f"  VPINN: L2={stats_v['l2_d']:.3e}, rel={stats_v['rel_l2_d']:.3e}")
    print(f"  X-RAS: L2={stats_x['l2_d']:.3e}, rel={stats_x['rel_l2_d']:.3e}")

    # 2) midline 对比（默认 y=H/2 的水平中线）
    png_v = phase2_ckpt_path.with_name(phase2_ckpt_path.stem + "_fe_vs_vpinn_midline.png")
    png_x = phase2_ckpt_path.with_name(phase2_ckpt_path.stem + "_fe_vs_xras_midline.png")

    mid_v = plot_fe_vs_pinn_midline(
        vpinn_solver,
        str(fe_phase_path),
        component="d",
        orientation=orientation,
        save_path=str(png_v),
    )
    mid_x = plot_fe_vs_pinn_midline(
        xras_solver,
        str(fe_phase_path),
        component="d",
        orientation=orientation,
        save_path=str(png_x),
    )

    print(f"  midline 图像已保存:")
    print(f"    - {png_v}")
    print(f"    - {png_x}")

    # 3) 打包 summary，方便后面直接读 npz 画表格
    summary = {
        "exp_name": ckpt.get("exp_name", "Phase2"),
        "config": config,
        "metrics_phase2": ckpt.get("metrics", {}),
        "stats_vpinn_global": stats_v,
        "stats_xras_global": stats_x,
        "stats_vpinn_midline": mid_v,
        "stats_xras_midline": mid_x,
    }

    if out_npz_path is None:
        out_npz_path = phase2_ckpt_path.with_suffix(".fe_summary.npz")
    else:
        out_npz_path = Path(out_npz_path)

    np.savez(out_npz_path, summary=summary)
    print(f"  ✓ FE summary 已保存到: {out_npz_path}")

    return summary


if __name__ == "__main__":
    # 示例：手动跑离线 summary
    base = Path(__file__).parent
    ckpt = base / "outputs" / "phase2_raw_Baseline.pth"
    fe_npz = base / "data" / "fe_sent_phasefield.npz"

    phase2_vs_fe_summary_offline(str(ckpt), str(fe_npz))
