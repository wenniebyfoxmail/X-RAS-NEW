"""
config_sent.py

SENT断裂问题的统一配置管理
Phase-1 (VPINN baseline) 和 Phase-2 (X-RAS-PINN) 共用此配置，确保公平对比

使用方法:
    from config_sent import create_config
    config = create_config(debug=True)  # 快速测试
    config = create_config(debug=False) # 精细实验
"""

def create_config(debug=True):
    """
    统一的SENT配置函数

    Args:
        debug: True=快速调试模式（MacBook友好）, False=论文级精细模式

    Returns:
        config: 配置字典，包含所有物理、数值、训练参数
    """

    # ========================================
    # 几何与材料常数（固定不变）
    # ========================================
    L = 1.0              # 板宽度
    H = 1.0              # 板高度
    notch_length = 0.3   # notch长度（从左边界延伸）

    E = 210.0            # 杨氏模量（无量纲）
    nu = 0.3             # 泊松比
    k = 1e-6             # 残余刚度参数

    device = "cpu"       # 计算设备

    base = {
        "L": L,
        "H": H,
        "notch_length": notch_length,
        "E": E,
        "nu": nu,
        "k": k,
        "device": device,
    }

    # ========================================
    # 调试模式 vs 精细模式
    # ========================================
    if debug:
        print(">> 配置模式: DEBUG (快速调试)")
        base.update({
            # -----------------------------------
            # 相场材料参数
            # -----------------------------------
            "G_c": 10e-3,           # 断裂能（略大，避免过早破坏）
            "l": 0.003,             # 长度尺度参数（略小，裂纹更集中）

            # -----------------------------------
            # 采样点数
            # -----------------------------------
            "n_domain": 1500,       # 域内采样点（VPINN baseline）
            "n_bc": 100,            # 边界采样点

            # -----------------------------------
            # 载荷参数
            # -----------------------------------
            "max_displacement": 0.012,   # 最大位移
            "n_loading_steps": 10,        # 载荷步数（Phase-1准静态）

            # -----------------------------------
            # Notch初始化参数
            # -----------------------------------
            "notch_seed_radius": 0.02,   # 初始损伤种子的高斯半径
            "initial_d": 0.5,            # notch尖端初始损伤峰值
            "notch_init_epochs": 300,    # notch预训练epoch数

            # -----------------------------------
            # 训练参数（Phase-1 VPINN）
            # -----------------------------------
            "lr_u": 2e-4,                # 位移网络学习率
            "lr_d": 2e-4,                # 损伤网络学习率
            "n_epochs_vpinn": 800,       # VPINN baseline训练epoch（Phase-2单步对比）
            "weight_bc": 100.0,          # 边界条件权重
            "weight_irrev": 0.0,         # 不可逆性约束（Phase-2简化，可设为0）

            # -----------------------------------
            # X-RAS-PINN参数
            # -----------------------------------
            "r_sing": 0.15,              # 奇异区半径（以notch_tip为中心）
            "beta_indicator": 0.5,       # 融合指标参数：(1-β)*SED + β*|∇d|
            "temperature_ras": 2.0,      # RAS采样温度参数
            "N_add_ras": 600,            # 每次RAS迭代新增点数

            # X-RAS三阶段训练
            "n_epochs_phase1": 400,      # Phase 1: 预训练
            "n_epochs_phase2": 400,      # Phase 2: 聚焦+自适应采样
            "n_epochs_phase3": 200,      # Phase 3: 联合微调
            "weight_interface": 50.0,    # 接口损失权重
            "n_interface": 150,          # 接口采样点数
        })
    else:
        print(">> 配置模式: FULL (论文级精细)")
        base.update({
            # 相场材料参数
            "G_c": 7e-3,
            "l": 0.004,

            # 采样点数
            "n_domain": 3000,
            "n_bc": 200,

            # 载荷参数
            "max_displacement": 0.010,
            "n_loading_steps": 12,

            # Notch初始化
            "notch_seed_radius": 0.03,
            "initial_d": 0.65,
            "notch_init_epochs": 800,

            # 训练参数（VPINN）
            "lr_u": 2e-4,
            "lr_d": 2e-4,
            "n_epochs_vpinn": 1500,
            "weight_bc": 100.0,
            "weight_irrev": 0.0,

            # X-RAS-PINN参数
            "r_sing": 0.15,
            "beta_indicator": 0.5,
            "temperature_ras": 2.0,
            "N_add_ras": 1000,

            # X-RAS三阶段
            "n_epochs_phase1": 600,
            "n_epochs_phase2": 600,
            "n_epochs_phase3": 300,
            "weight_interface": 50.0,
            "n_interface": 200,
        })

    # ========================================
    # 计算辅助参数
    # ========================================
    base["notch_tip"] = [notch_length, H / 2.0]  # notch尖端坐标

    return base


def print_config(config):
    """打印配置摘要（便于日志记录）"""
    print("\n" + "="*70)
    print("  SENT Configuration Summary")
    print("="*70)
    print(f"  几何: L={config['L']}, H={config['H']}, notch_length={config['notch_length']}")
    print(f"  材料: E={config['E']}, ν={config['nu']}")
    print(f"  相场: G_c={config['G_c']:.2e}, l={config['l']:.4f}")
    print(f"  采样: n_domain={config['n_domain']}, n_bc={config['n_bc']}")
    print(f"  载荷: max_δ={config['max_displacement']:.4f}, steps={config['n_loading_steps']}")
    print(f"  VPINN训练: epochs={config['n_epochs_vpinn']}")
    print(f"  X-RAS训练: Phase1={config['n_epochs_phase1']}, "
          f"Phase2={config['n_epochs_phase2']}, Phase3={config['n_epochs_phase3']}")
    print(f"  奇异区半径: r_sing={config['r_sing']}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("测试配置生成:\n")

    # DEBUG模式
    config_debug = create_config(debug=True)
    print_config(config_debug)

    # FULL模式
    config_full = create_config(debug=False)
    print_config(config_full)