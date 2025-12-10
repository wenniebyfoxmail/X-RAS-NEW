import numpy as np

# 1. 读取 FE 基线数据（test_el.py 生成的）
data = np.load("sent_fe_baseline.npz")
coords = data["coords"]   # (ndof, 2)
u = data["u"]             # (ndof,)

# 拆成 ux, uy
ux = u[0::2]
uy = u[1::2]

print("coords shape:", coords.shape)
print("u shape:", u.shape)
print("ux range:", ux.min(), "->", ux.max())
print("uy range:", uy.min(), "->", uy.max())

# 2. 提取中线（y ≈ Ly/2），做 1D baseline
Ly = 0.2      # 要和 test_el.py 里保持一致！
y_mid = Ly / 2
tol = Ly / 20  # 容差，比如 ±Ly/20

x_all = coords[:, 0]
y_all = coords[:, 1]

mask_mid = np.abs(y_all - y_mid) < tol

x_mid = x_all[mask_mid]
ux_mid = ux[mask_mid]

# 按 x 排序
idx = np.argsort(x_mid)
x_mid = x_mid[idx]
ux_mid = ux_mid[idx]

print("1D mid-line samples:", x_mid.shape[0])
print("x_mid range:", x_mid.min(), "->", x_mid.max())

# 3. 存成 1D baseline，给 X-RAS-PINN 用
np.savez("sent_fe_midline_1d.npz", x=x_mid, ux=ux_mid)
print("saved 1D baseline to sent_fe_midline_1d.npz")
