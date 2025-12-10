import numpy as np
import matplotlib.pyplot as plt


def load_fe_midline(path="sent_fe_midline_1d.npz"):
    """读取 FEniCS FE 中线 1D 基线数据."""
    data = np.load(path)
    x = data["x"]
    ux = data["ux"]
    return x, ux


def compare_fe_pinn_midline(get_pinn_u, path="sent_fe_midline_1d.npz"):
    """
    get_pinn_u: 一个函数，输入 x (N,1) 的 numpy / torch tensor，返回预测位移 (N,1)。
    返回：x, ux_fe, ux_pinn, rel_L2
    """
    x, ux_fe = load_fe_midline(path)

    # 统一成 (N,1) 形式
    x_input = x.reshape(-1, 1)

    # 调用你的 PINN / X-RAS 模型（假定你内部处理好 numpy/torch）
    ux_pinn = get_pinn_u(x_input)

    # 转 numpy （如果是 torch）
    try:
        import torch

        if isinstance(ux_pinn, torch.Tensor):
            ux_pinn = ux_pinn.detach().cpu().numpy()
    except ImportError:
        pass

    ux_pinn = ux_pinn.reshape(-1)

    err = ux_pinn - ux_fe
    rel_L2 = np.linalg.norm(err) / np.linalg.norm(ux_fe)

    return x, ux_fe, ux_pinn, rel_L2


def plot_fe_vs_pinn_midline(x, ux_fe, ux_pinn, save_path="compare_fe_pinn_midline.png"):
    """画 FE vs PINN 的中线对比图."""
    plt.figure()
    plt.plot(x, ux_fe, label="FE midline ux")
    plt.plot(x, ux_pinn, "--", label="X-RAS-PINN midline ux")
    plt.xlabel("x")
    plt.ylabel("ux")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved comparison figure to {save_path}")

