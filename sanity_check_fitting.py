import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from phase_field_vpinn import DamageNetwork  # 确保能导入你的网络定义


def run_sanity_check(fe_npz_path="data/fe_sent_phasefield.npz"):
    print("=" * 60)
    print("  SANITY CHECK: Can the network fit the sharp FE spike?")
    print("=" * 60)

    # 1. 加载 FE 真值
    if not os.path.exists(fe_npz_path):
        print(f"❌ 找不到 FE 文件: {fe_npz_path}")
        return

    data = np.load(fe_npz_path)
    # 获取最后一步的场
    d_fe = data['d'][:, -1]  # (N_fe, )
    coords = data['coords_d']  # (N_fe, 2)

    # 转为 Tensor
    x_train = torch.tensor(coords[:, :2], dtype=torch.float32)
    d_target = torch.tensor(d_fe, dtype=torch.float32).unsqueeze(1)

    # 2. 初始化网络
    d_net = DamageNetwork()
    optimizer = torch.optim.Adam(d_net.parameters(), lr=1e-3)

    # 3. 监督学习 (Supervised Fitting)
    print(f"  开始拟合 {x_train.shape[0]} 个 FE 数据点...")
    epochs = 2000
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        d_pred = d_net(x_train)

        # 纯 MSE Loss (强制拟合)
        loss = torch.mean((d_pred - d_target) ** 2)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d} | Loss (MSE): {loss.item():.6e}")

    # 4. 可视化对比 (Midline)
    print("\n  生成对比图...")
    plt.figure(figsize=(10, 5))

    # 提取中线数据
    y_mid = 0.5
    tolerance = 0.02
    mask_mid = np.abs(coords[:, 1] - y_mid) < tolerance

    # 排序以便画线
    x_mid = coords[mask_mid, 0]
    sort_idx = np.argsort(x_mid)
    x_mid = x_mid[sort_idx]

    d_fe_mid = d_fe[mask_mid][sort_idx]

    with torch.no_grad():
        d_pred_mid = d_net(x_train[mask_mid]).numpy().flatten()[sort_idx]

    plt.plot(x_mid, d_fe_mid, 'k-', linewidth=2, label='FE Ground Truth (Target)')
    plt.plot(x_mid, d_pred_mid, 'r--', linewidth=2, label='Network Fitting')
    plt.title(f"Sanity Check: Supervised Fitting (MSE={loss_history[-1]:.1e})")
    plt.xlabel("x")
    plt.ylabel("d")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = "outputs/sanity_check_result.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ 结果已保存: {out_path}")
    plt.show()


if __name__ == "__main__":
    run_sanity_check()