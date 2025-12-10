import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 配置路径
# ==========================================
# 你刚才上传的文件 (包含 L2 误差统计)
summary_path = "outputs/phase2_raw_Baseline.fe_summary.npz"
# FE 基准文件 (包含物理历史数据，用于画载荷曲线)
fe_data_path = "data/fe_sent_phasefield.npz"


def check_structure_and_verify_physics():
    print("=" * 60)
    print(" 🛠️  X-RAS-PINN: 数据结构检查与物理验证助手")
    print("=" * 60)

    # ---------------------------------------------------------
    # 任务 1: 检查 Summary 文件 (查看 L2 误差)
    # ---------------------------------------------------------
    if os.path.exists(summary_path):
        print(f"\n[1] 正在检查 Summary 文件: {summary_path}")
        try:
            data = np.load(summary_path, allow_pickle=True)
            # summary 通常保存为一个 object 数组，需要提取出来
            if data.files:
                # 假设保存时使用的是 key='summary' 或者默认的 'arr_0'
                key = data.files[0]
                content = data[key].item()  # 提取字典

                print("\n   >>> 关键指标摘要 (Metrics):")
                if 'metrics_phase2' in content:
                    m = content['metrics_phase2']
                    print(f"   - L2 Error (Global): {m.get('l2_all', 'N/A')}")
                    print(f"   - L2 Error (Near Tip): {m.get('l2_near', 'N/A')}")
                    print(f"   - Max Damage (VPINN): {m.get('d_max_vpinn', 'N/A'):.4f}")
                    print(f"   - Max Damage (X-RAS): {m.get('d_max_xras', 'N/A'):.4f}")

                if 'stats_xras_global' in content:
                    gx = content['stats_xras_global']
                    print(f"\n   >>> X-RAS 全局统计:")
                    print(f"   - L2(d): {gx.get('l2_d', 'N/A'):.4e}")
                    print(f"   - Rel L2(d): {gx.get('rel_l2_d', 'N/A'):.4e}")
            else:
                print("   [Error] 文件中没有 keys.")
        except Exception as e:
            print(f"   [Error] 读取 Summary 失败: {e}")
    else:
        print(f"   [Warning] 找不到文件: {summary_path}")

    # ---------------------------------------------------------
    # 任务 2: 绘制载荷-位移曲线 (验证 d=1.0 的合理性)
    # ---------------------------------------------------------
    print(f"\n[2] 正在进行物理验证 (Load-Reaction Curve): {fe_data_path}")

    if not os.path.exists(fe_data_path):
        print(f"   ❌ 错误: 找不到 FE 基准文件! 请确认 {fe_data_path} 存在。")
        print("   无法判断是否发生破坏。")
        return

    try:
        fe_data = np.load(fe_data_path, allow_pickle=True)

        # 尝试获取载荷步和反力
        # 注意：不同版本的代码可能 key 不一样，这里做防御性编程
        keys = fe_data.files

        u_steps = None
        reactions = None

        # 尝试常见的 key 名
        if 'load_steps' in keys:
            u_steps = fe_data['load_steps']
        elif 'u_hist' in keys:
            u_steps = np.linspace(0, 0.01, len(fe_data['d_hist']))  # 估算

        if 'reactions' in keys: reactions = fe_data['reactions']

        if u_steps is None or reactions is None:
            print(f"   ❌ 数据缺失: 无法找到 'load_steps' 或 'reactions'。Keys: {keys}")
            return

        # 确保维度匹配
        min_len = min(len(u_steps), len(reactions))
        u_steps = u_steps[:min_len]
        reactions = reactions[:min_len]

        # 寻找峰值载荷
        max_load_idx = np.argmax(reactions)
        max_load = reactions[max_load_idx]
        final_load = reactions[-1]

        # 绘图
        plt.figure(figsize=(8, 6), dpi=120)
        plt.plot(u_steps, reactions, 'b-o', linewidth=2, label='FE Reaction Force')
        plt.plot(u_steps[max_load_idx], max_load, 'rx', markersize=12, markeredgewidth=3, label='Peak Load')

        plt.title("Load-Displacement Curve (FE Ground Truth)", fontsize=14)
        plt.xlabel("Displacement (mm)", fontsize=12)
        plt.ylabel("Reaction Force (N)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # --- 核心判断逻辑 ---
        print("\n" + "=" * 40)
        print("   🤖 联合导师物理诊断报告")
        print("=" * 40)
        print(f"   * 峰值载荷: {max_load:.4f} N (at u={u_steps[max_load_idx]:.4f})")
        print(f"   * 最终载荷: {final_load:.4f} N")

        # 判断是否软化
        is_softening = final_load < (0.95 * max_load)  # 如果下降超过 5%

        if is_softening:
            status_msg = "✅ 发生软化 (Softening)！"
            phy_msg = (
                "结论：材料已经越过了极限承载点，裂纹必然已经失稳扩展。\n"
                "      这意味着裂尖核心的物理损伤值 d 理论上应该达到 1.0 (完全破坏)。\n"
                "      --> 你的 X-RAS 预测出 d=1.0 是【物理正确】的！\n"
                "      --> FE 的 d=0.95 可能是数值截断或网格锁死导致的误差。"
            )
            color = 'green'
        else:
            status_msg = "⚠️ 尚未明显软化 (Hardening/Elastic)"
            phy_msg = (
                "结论：载荷仍在上升或持平，裂纹可能尚未完全贯穿，或者处于塑性/损伤起始阶段。\n"
                "      --> 此时 d < 1.0 是合理的。\n"
                "      --> 如果你的 X-RAS 预测出 d=1.0，可能是对损伤演化过于敏感 (Aggressive)。"
            )
            color = 'orange'

        print(f"   * 状态判定: {status_msg}")
        print("-" * 40)
        print(phy_msg)
        print("=" * 40)

        # 在图上标注
        plt.text(0.05, 0.5, status_msg, transform=plt.gca().transAxes,
                 fontsize=12, color=color, fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.8))

        save_path = "outputs/check_physics_load_curve.png"
        plt.savefig(save_path)
        print(f"\n   📊 曲线图已保存至: {save_path}")
        plt.show()

    except Exception as e:
        print(f"   [Error] 处理 FE 数据时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_structure_and_verify_physics()
    import numpy as np

    # 读取数据
    data = np.load("data/fe_sent_phasefield.npz")
    reactions = data['reactions']

    # 1. 处理符号：取绝对值（假设是单轴拉伸，反力为负）
    abs_force = np.abs(reactions)
    steps = np.arange(len(abs_force))

    # 2. 找到最大载荷（承载能力极限）
    peak_idx = np.argmax(abs_force)
    peak_load = abs_force[peak_idx]
    final_load = abs_force[-1]

    # 3. 计算刚度（斜率）的变化
    # stiffness ~ d|F| / d(step)
    stiffness = np.diff(abs_force)

    print("=" * 50)
    print(" 🧐 载荷-反力数值显微镜 (Load-Reaction Diagnosis)")
    print("=" * 50)

    print(f"总步数 (Total Steps): {len(abs_force)}")
    print(f"峰值载荷 (Peak Load): {peak_load:.6f} N (第 {peak_idx} 步)")
    print(f"最终载荷 (Final Load): {final_load:.6f} N (第 {len(abs_force) - 1} 步)")

    print("-" * 50)
    print(">>>以此判断是否发生软化 (Softening):")

    if peak_idx == len(abs_force) - 1:
        print("❌ [结论]：未发生软化！(No Softening)")
        print("   现象：载荷一直增加，直到最后一步都是最大值。")
        print("   物理含义：裂纹还在稳定扩展期，或者仅处于损伤累积阶段，尚未发生失稳断裂。")
        print("   推论：此时 d=1.0 (X-RAS) 确实是'过冲'了，FE 的 0.95 可能更准确。")
    else:
        drop_ratio = (peak_load - final_load) / peak_load * 100
        print(f"✅ [结论]：发生了软化！(Softening Detected)")
        print(f"   现象：载荷在第 {peak_idx} 步达到峰值，随后下降。")
        print(f"   下降幅度：{drop_ratio:.2f}%")
        print("   物理含义：结构承载能力下降，必然伴随主裂纹的宏观扩展。")

    print("-" * 50)
    print(">>> 最后 5 步的载荷数值 (及斜率):")
    print("Step |  Load (Abs)  |  Delta (Slope)")
    for i in range(max(0, len(abs_force) - 6), len(abs_force)):
        val = abs_force[i]
        if i > 0:
            delta = val - abs_force[i - 1]
            delta_str = f"{delta:+.6f}"
        else:
            delta_str = "N/A"

        mark = " <--- MAX" if i == peak_idx else ""
        print(f"{i:4d} |  {val:.6f}    |  {delta_str} {mark}")
    print("=" * 50)