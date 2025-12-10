import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime




def analyze_and_plot(filename):
    # 1. 加载数据
    try:
        data = np.load(filename)
        print(f"成功加载文件: {filename}")
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        return

    # 2. 读取并打印文件中的列 (Keys)
    print("-" * 30)
    print("文件包含的列 (Keys):")
    print(data.files)
    print("-" * 30)

    # 提取数据
    load_steps = data['load_steps']  # 位移
    reactions = data['reactions']  # 反力
    damage = data['d']  # 损伤场
    coords = data['coords_d']  # 坐标

    x = coords[:, 0]
    y = coords[:, 1]

    print("G_c = ", data['G_c'])
    # 确定需要分析的步数索引
    # 包含 0, 5, 10, 15 和最后一步 (-1 表示最后一步)
    # 注意：确保索引不超过数据总长度
    total_steps = len(load_steps)
    print("step数量为：", total_steps)
    target_indices = [0, 5, 9, 10, 11, 12, 13, 14, 15, 16, total_steps - 1]

    # 过滤掉超出范围的索引（以防万一数据少于16步）
    valid_indices = [i for i in target_indices if i < total_steps]

    # 3. 打印指定步的结果 (位移和反力)
    print(f"{'Step':<10} | {'Displacement (mm)':<20} | {'Reaction Force (N)':<20}")
    print("-" * 56)
    for idx in valid_indices:
        u_val = load_steps[idx]
        r_val = reactions[idx]
        print(f"{idx:<10} | {u_val:<20.5f} | {r_val:<20.5f} ")
    print("-" * 56)

    # 4. 画图：位移-反力曲线
    plt.figure(figsize=(8, 6))
    plt.plot(load_steps, reactions, '-o', markersize=4, label='Reaction Force')

    # 标记出特定点
    plt.plot(load_steps[valid_indices], reactions[valid_indices], 'rx', markersize=10, label='Selected Steps')

    plt.title('Displacement - Reaction Force Curve')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Reaction Force (N)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    curve_name = 'force_displacement_curve_marked.png'
    plt.savefig(curve_name, dpi=300)
    print(f"\n[图表] 位移-反力曲线已保存为: {curve_name}")
    plt.close()

    # 5. 画图：损伤云图 (0, 5, 10, 15, Final)
    # 创建子图：1行 N列
    num_plots = len(valid_indices)
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5), constrained_layout=True)

    # 如果只有一个图，axes 不是列表，需要处理
    if num_plots == 1:
        axes = [axes]

    for i, step_idx in enumerate(valid_indices):
        ax = axes[i]
        d_field = damage[:, step_idx]

        # 绘制云图
        # levels=np.linspace(0, 1, 11) 设定0到1之间10个等级
        contour = ax.tricontourf(x, y, d_field, levels=np.linspace(0, 1, 11), cmap='Reds')

        # 标题处理：如果是最后一步，额外标注
        title_str = f'Step {step_idx}'
        if step_idx == total_steps - 1:
            title_str += ' (Final)'

        ax.set_title(title_str)
        ax.set_aspect('equal')
        ax.axis('off')  # 不显示坐标轴

    # 添加共用的颜色条
    cbar = fig.colorbar(contour, ax=axes, shrink=0.6, location='right')
    cbar.set_label('Damage Phase Field (d)')

    contour_name = 'damage_contours_steps.png'
    plt.savefig(contour_name, dpi=300)
    print(f"[图表] 损伤云图已保存为: {contour_name}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("当前时间戳:", timestamp)

if __name__ == "__main__":
    # 文件名
    filename =  "data/fe_sent_phasefield.npz"
    analyze_and_plot(filename)