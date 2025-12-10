"""
fe_baseline_utils.py

FX: FEniCSx 相场 FE 基准 与 PINN/X-RAS 的对比工具
-------------------------------------------------
最小可用版本，支持：
- 加载 FE phase-field npz
- 计算全域 L2 误差 (u, d)
- 水平/竖直中线的 damage / 位移对比
- 画 FE vs PINN 的 midline 曲线

假定 FE npz 由 sent_phasefield_fe.py 生成，字段包括：
    coords : (Ndof_u, 2)  由 Vu.tabulate_dof_coordinates() 得到
    u      : (Ndof_u, n_steps)
    d      : (Ndof_d, n_steps)
    load_steps : (n_steps,)
    reactions  : (n_steps,)
    L, H, notch_length, E, nu, G_c, l

注意：
- 这里假定 Ndof_u = 2 * Ndof_d（P1 向量空间 + P1 标量空间）
- coords 中每两个 DOF 共用一个节点坐标，对应 (ux, uy)
"""

from __future__ import annotations

import os
from typing import Dict, Literal, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# 1. 基本加载
# -----------------------------------------------------------------------------

def load_fe_phasefield_npz(fe_path: str) -> Dict:
    """加载由 sent_phasefield_fe.py 生成的 FE phase-field npz

    新版约定：
        coords_d : (Ndof_d, 2)  来自 Vd.tabulate_dof_coordinates()
        coords_u : (Ndof_u, 2)  来自 Vu.tabulate_dof_coordinates()
        u        : (Ndof_u, n_steps)
        d        : (Ndof_d, n_steps)
    """
    if not os.path.exists(fe_path):
        raise FileNotFoundError(f"FE npz not found: {fe_path}")

    data = np.load(fe_path)

    u_hist = data["u"]              # (Ndof_u, n_steps_u)
    d_hist = data["d"]              # (Ndof_d, n_steps_d)
    load_steps = data["load_steps"] # (n_steps_l,)
    reactions = data["reactions"]   # (n_steps_r,)

    # 对齐时间步
    n_steps = min(
        u_hist.shape[1],
        d_hist.shape[1],
        load_steps.shape[0],
        reactions.shape[0],
    )
    u_hist = u_hist[:, :n_steps]
    d_hist = d_hist[:, :n_steps]
    load_steps = load_steps[:n_steps]
    reactions = reactions[:n_steps]

    Ndof_d = d_hist.shape[0]

    # --- 1) 坐标：优先用 coords_d ---
    if "coords_d" in data:
        coords_nodes = data["coords_d"]  # (Ndof_d, 2)
    elif "coords" in data:
        # 兼容旧版本：coords 是 Vu 的 DOF 坐标
        coords_all = data["coords"]
        Ndof_u, dim = coords_all.shape
        if Ndof_u >= 2 * Ndof_d:
            # 取每隔 dim 个 DOF 当做节点坐标
            dim = Ndof_u // Ndof_d
            coords_nodes = coords_all[0:dim * Ndof_d:dim, :]
        else:
            coords_nodes = coords_all[:Ndof_d, :]
    else:
        raise KeyError("npz file must包含 'coords_d' 或旧格式 'coords'")

    # --- 2) 从 u_hist 还原每个节点的 (ux, uy) ---
    u_last_nodes = None
    Ndof_u = u_hist.shape[0]
    if Ndof_u >= 2 * Ndof_d:
        dim = Ndof_u // Ndof_d  # 通常 = 2
        u_last_full = u_hist[:, -1]
        u_last_nodes = np.zeros((Ndof_d, dim), dtype=u_last_full.dtype)
        for comp in range(dim):
            u_last_nodes[:, comp] = u_last_full[comp:dim * Ndof_d:dim]

    d_last = d_hist[:, -1]  # (Ndof_d,)

    out = {
        "coords_nodes": coords_nodes,
        "u_last": u_last_nodes,
        "d_last": d_last,
        "u_hist": u_hist,
        "d_hist": d_hist,
        "load_steps": load_steps,
        "reactions": reactions,
    }

    # 标量参数带上
    for key in ["L", "H", "notch_length", "E", "nu", "G_c", "l"]:
        if key in data:
            out[key] = float(data[key])

    return out



# -----------------------------------------------------------------------------
# 2. 全域 L2 误差 (u, d)
# -----------------------------------------------------------------------------

def _to_torch_points(coords: np.ndarray, solver) -> torch.Tensor:
    """将 numpy 坐标转成放到 solver.device 上的 torch 张量

    - 如果 FE 给的是 (N, 3)（例如 (x, y, 0)），而 PINN 只需要 (x, y)，就自动裁成前两列。
    """
    coords_np = coords.astype(np.float32)

    # 如果是 (N, dim)，且 dim > 2，则只取前两个分量作为 (x, y)
    if coords_np.ndim == 2 and coords_np.shape[1] > 2:
        coords_np = coords_np[:, :2]

    x = torch.from_numpy(coords_np)
    device = getattr(solver, "device", "cpu")
    return x.to(device)




def compute_global_L2_for_u_and_d(
    solver,
    fe_phase_path: str,
    verbose: bool = True,
) -> Dict[str, float]:
    """在 FE 节点上评估 PINN/X-RAS，并与 FE phase-field 做全域 L2 对比

    返回:
        {
            "l2_u": ...,
            "rel_l2_u": ...,
            "l2_d": ...,
            "rel_l2_d": ...,
        }
    """
    fe = load_fe_phasefield_npz(fe_phase_path)
    coords = fe["coords_nodes"]              # (Nc, dim)
    d_fe = fe["d_last"]                      # (Nd,)

    # ---- 先对齐 coords 与 d_fe 的长度 ----
    Nc = coords.shape[0]
    Nd = d_fe.shape[0]
    N0 = min(Nc, Nd)

    if Nc != N0:
        coords = coords[:N0, :]
    if Nd != N0:
        d_fe = d_fe[:N0]

    # 评估 PINN/X-RAS（在对齐后的 coords 上）
    x_t = _to_torch_points(coords, solver)   # 这里会自动裁成 (N, 2)
    u_pred, d_pred = solver.predict(x_t)
    u_np = u_pred.detach().cpu().numpy()           # (N, 2)
    d_np = d_pred.detach().cpu().numpy().reshape(-1)  # (N,)

    # 再保守对齐一次，防止 predict 的 N 和 d_fe 不一致
    N = min(d_np.shape[0], d_fe.shape[0])
    d_np = d_np[:N]
    d_fe = d_fe[:N]

    # ---- damage L2 ----
    diff_d = d_np - d_fe
    l2_d = float(np.sqrt(np.mean(diff_d ** 2)))
    denom_d = float(np.sqrt(np.mean(d_fe ** 2)) + 1e-12)
    rel_l2_d = l2_d / denom_d

    # ---- displacement L2 (如果 u_last 可用) ----
    u_last = fe.get("u_last", None)
    if u_last is not None:
        # 对齐 u_np 与 u_last 的节点数
        Nu = u_np.shape[0]
        Nul = u_last.shape[0]
        N_common = min(Nu, Nul)

        u_np_c = u_np[:N_common, :]
        u_last_c = u_last[:N_common, :]

        diff_u = u_np_c - u_last_c
        diff_u_norm2 = np.sum(diff_u ** 2, axis=1)
        u_norm2 = np.sum(u_last_c ** 2, axis=1)

        l2_u = float(np.sqrt(np.mean(diff_u_norm2)))
        denom_u = float(np.sqrt(np.mean(u_norm2)) + 1e-12)
        rel_l2_u = l2_u / denom_u
    else:
        # 如果无法可靠拆分 ux/uy，就返回 NaN
        l2_u = float("nan")
        rel_l2_u = float("nan")

    if verbose:
        print("[FE Global L2]")
        print(f"  L2(d):      {l2_d:.3e}  (rel={rel_l2_d:.3e})")
        print(f"  L2(u_vec):  {l2_u:.3e}  (rel={rel_l2_u:.3e})")

    return {
        "l2_u": l2_u,
        "rel_l2_u": rel_l2_u,
        "l2_d": l2_d,
        "rel_l2_d": rel_l2_d,
    }



# -----------------------------------------------------------------------------
# 3. 中线 (midline) 对比：damage / displacement
# -----------------------------------------------------------------------------

def _extract_fe_midline(
    fe_data: Dict,
    component: Literal["d", "ux", "uy"],
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> Tuple[np.ndarray, np.ndarray]:
    """从 FE 数据中抽取中线上的值（坐标和数值一一对应）

    返回:
        s      : 参数坐标 (比如 x 或 y)
        values : 对应的 FE 数值
    """
    coords = fe_data["coords_nodes"]      # (N, 2)，与 d_last 同一顺序
    d_last = fe_data["d_last"]           # (N,)
    u_last = fe_data.get("u_last", None) # (N, 2) or None

    L = float(fe_data.get("L", 1.0))
    H = float(fe_data.get("H", 1.0))

    x = coords[:, 0]
    y = coords[:, 1]

    # 选择字段
    if component == "d":
        values_all = d_last
    elif component in ("ux", "uy") and u_last is not None:
        comp_idx = 0 if component == "ux" else 1
        values_all = u_last[:, comp_idx]
    else:
        raise ValueError(f"Component '{component}' not available in FE data.")

    # 选择“最接近中线”的这一排 DOF
    if orientation == "horizontal":
        # 找到所有不同的 y 值，然后选离 H/2 最近的那一条
        ys_unique = np.unique(np.round(y, decimals=10))
        y_target = ys_unique[np.argmin(np.abs(ys_unique - H / 2.0))]
        mask = np.isclose(y, y_target, atol=1e-8)
        s = x[mask]
        vals = values_all[mask]
        order = np.argsort(s)
    elif orientation == "vertical":
        xs_unique = np.unique(np.round(x, decimals=10))
        x_target = xs_unique[np.argmin(np.abs(xs_unique - L / 2.0))]
        mask = np.isclose(x, x_target, atol=1e-8)
        s = y[mask]
        vals = values_all[mask]
        order = np.argsort(s)
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    s_sorted = s[order]
    vals_sorted = vals[order]
    return s_sorted, vals_sorted



def compare_fe_pinn_midline(
    solver,
    fe_path: str,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    component: Literal["d", "ux", "uy"] = "d",
    verbose: bool = True,
) -> Dict:
    """计算 FE vs PINN 在中线上的 L2 误差

    返回:
        {
            "s": s,
            "fe": fe_vals,
            "pinn": pinn_vals,
            "l2": ...,
            "rel_l2": ...,
        }
    """
    fe = load_fe_phasefield_npz(fe_path)
    s, fe_vals = _extract_fe_midline(fe, component=component, orientation=orientation)

    # 在同一批点上评估 PINN/X-RAS
    coords = fe["coords_nodes"]
    L = float(fe.get("L", 1.0))
    H = float(fe.get("H", 1.0))

    x = coords[:, 0]
    y = coords[:, 1]

    tol = min(L, H) * 1e-3
    if orientation == "horizontal":
        mask = np.abs(y - H / 2.0) < tol
        pts = np.stack([x[mask], y[mask]], axis=1)[np.argsort(x[mask])]
    else:
        mask = np.abs(x - L / 2.0) < tol
        pts = np.stack([x[mask], y[mask]], axis=1)[np.argsort(y[mask])]

    x_t = _to_torch_points(pts, solver)
    u_pred, d_pred = solver.predict(x_t)
    u_np = u_pred.detach().cpu().numpy()
    d_np = d_pred.detach().cpu().numpy().reshape(-1)

    if component == "d":
        pinn_vals = d_np
    elif component == "ux":
        pinn_vals = u_np[:, 0]
    elif component == "uy":
        pinn_vals = u_np[:, 1]
    else:
        raise ValueError(f"Unknown component: {component}")

    diff = pinn_vals - fe_vals
    l2 = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.sqrt(np.mean(fe_vals ** 2)) + 1e-12)
    rel_l2 = l2 / denom

    if verbose:
        print("[FE Midline]")
        print(f"  component = {component}, orientation = {orientation}")
        print(f"  L2 = {l2:.3e}, rel = {rel_l2:.3e}")

    return {
        "s": s,
        "fe": fe_vals,
        "pinn": pinn_vals,
        "l2": l2,
        "rel_l2": rel_l2,
    }


def plot_fe_vs_pinn_midline(
    solver,
    fe_path: str,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    component: Literal["d", "ux", "uy"] = "d",
    save_path: str | os.PathLike | None = None,
) -> Dict:
    """画 FE vs PINN 在中线上的对比曲线，并返回误差指标"""

    stats = compare_fe_pinn_midline(
        solver,
        fe_path=fe_path,
        orientation=orientation,
        component=component,
        verbose=False,
    )

    s = stats["s"]
    fe_vals = stats["fe"]
    pinn_vals = stats["pinn"]

    plt.figure(figsize=(6, 4))
    plt.plot(s, fe_vals, "k-", linewidth=2, label=f"FE ({component})")
    plt.plot(s, pinn_vals, "r--", linewidth=2, label=f"PINN/X-RAS ({component})")
    plt.xlabel("x" if orientation == "horizontal" else "y")
    plt.ylabel(component)
    plt.title(f"Midline: {component} ({orientation})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

    return stats


# -----------------------------------------------------------------------------
# 4. 针对 damage 的快捷接口（给 summary 用）
# -----------------------------------------------------------------------------

def compare_damage_midline(
    solver,
    fe_phase_path: str,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    verbose: bool = True,
) -> Dict:
    """专门用于 damage (d) 的 midline 误差对比"""

    stats = compare_fe_pinn_midline(
        solver,
        fe_phase_path,
        orientation=orientation,
        component="d",
        verbose=verbose,
    )
    # 为了和 compute_global_L2_for_u_and_d 的 key 风格统一
    return {
        "s": stats["s"],
        "fe": stats["fe"],
        "pinn": stats["pinn"],
        "l2": stats["l2"],
        "rel_l2": stats["rel_l2"],
    }


# -----------------------------------------------------------------------------
# 5. 载荷–反力曲线对比（占位实现）
# -----------------------------------------------------------------------------

def compare_load_reaction_curve(
    fe_phase_path: str,
    pinn_load: np.ndarray,
    pinn_reaction: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """对比 FE 与 PINN 的载荷–反力曲线

    当前是一个占位实现：只基于 FE 的 load_steps / reactions 与 PINN 提供的
    load / reaction 序列进行 L2 计算。你可以在后续把 PINN 的反力统计接入这里。
    """
    fe = load_fe_phasefield_npz(fe_phase_path)
    fe_load = fe["load_steps"]
    fe_react = fe["reactions"]

    # 简单假设：PINN 与 FE 使用相同的 load 序列
    # 如果不同，可以在外部插值后再传进来
    if len(pinn_load) != len(fe_load):
        n = min(len(pinn_load), len(fe_load))
        pinn_load = pinn_load[:n]
        pinn_reaction = pinn_reaction[:n]
        fe_load = fe_load[:n]
        fe_react = fe_react[:n]

    diff = pinn_reaction - fe_react
    l2 = float(np.sqrt(np.mean(diff ** 2)))
    denom = float(np.sqrt(np.mean(fe_react ** 2)) + 1e-12)
    rel_l2 = l2 / denom

    if verbose:
        print("[FE Load-Reaction]")
        print(f"  L2 = {l2:.3e}, rel = {rel_l2:.3e}")

    return {
        "l2": l2,
        "rel_l2": rel_l2,
        "fe_load": fe_load,
        "fe_reaction": fe_react,
    }
