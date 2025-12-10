import numpy as np

# 文件路径 (根据你报错信息里的路径)
file_path = "/Users/wenxiaofang/Documents/GitHub/X-RAS-NEW/outputs/phase2_vs_fe_summary_Baseline.npz"

try:
    # 加载数据
    data = np.load(file_path)

    print("=" * 50)
    print("      实验结果数据解读      ")
    print("=" * 50)

    # 打印每个指标
    # 使用 item() 将 numpy 标量转为 python float 以便格式化

    print(f"实验名称: {data['exp_name']}")
    print(f"FE 真值路径: {data['fe_path']}")
    print("-" * 50)

    print(f"{'Metric':<25} | {'VPINN (Baseline)':<15} | {'X-RAS (Yours)':<15}")
    print("-" * 60)

    # 全局位移误差
    u_v = data['rel_l2_u_vpinn'].item()
    u_x = data['rel_l2_u_xras'].item()
    print(f"{'Rel L2 u (Displacement)':<25} | {u_v:.2e}        | {u_x:.2e}")

    # 全局损伤误差
    d_v = data['rel_l2_d_vpinn'].item()
    d_x = data['rel_l2_d_xras'].item()
    print(f"{'Rel L2 d (Damage)':<25} | {d_v:.2e}        | {d_x:.2e}")

    # 中线损伤误差
    md_v = data['rel_l2_d_mid_vpinn'].item()
    md_x = data['rel_l2_d_mid_xras'].item()
    print(f"{'Midline L2 d (Local)':<25} | {md_v:.2e}        | {md_x:.2e}")

    print("=" * 50)

    # 简单的胜负判断
    if d_x < d_v:
        print("✅ 结论: X-RAS 在损伤场预测上优于 Baseline！")
    else:
        print("⚠️ 结论: X-RAS 表现不如 Baseline，可能需要调整参数 (r_sing, N_add)。")

except Exception as e:
    print(f"读取失败: {e}")