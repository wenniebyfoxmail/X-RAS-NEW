"""
统一配置管理 - Phase-1 & Phase-2
功能：
1. 所有物理参数、训练超参只写一份
2. 支持 DEBUG/FULL 模式切换
3. Phase-1 和 Phase-2 共用配置保证一致性
"""


def create_config(debug=True):
    """
    统一管理所有物理 & 数值参数

    Args:
        debug: bool
            True:  快速调参配置 (DEBUG_MODE)
            False: 论文级精细配置 (FULL_MODE)

    Returns:
        config: dict - 完整配置字典
    """
    # =====================================================================
    # 基础参数 (DEBUG/FULL 共用)
    # =====================================================================
    base_config = {
        # 几何参数
        "L": 1.0,                      # 试样长度
        "H": 1.0,                      # 试样高度
        "notch_length": 0.3,           # notch 长度

        # 材料参数
        "E": 3e10,                    # 杨氏模量
        "nu": 0.2,                     # 泊松比
        "k": 1e-6,                     # 残余刚度

        # 计算设置
        "device": "cpu",
        "lr_u": 1e-4,                  # 位移网络学习率
        "lr_d": 1e-4,                  # 损伤网络学习率
        "seed": 1234,                  # 随机种子

        # 坐标范围 (for Phase-2)
        "x_min": 0.0,
        "x_max": 1.0,
        "y_min": 0.0,
        "y_max": 1.0,

        # 动态划分
        "use_dynamic_sing": True,       # 打开动态 Ω_sing
        "dynamic_sing_d_min": 0.10,     # 裂尖窗口下限
        "dynamic_sing_d_max": 0.60,     # 裂尖窗口上限
        "dynamic_sing_min_points": 200, # 太少就回退到圆形


        # 数据路径
        "fe_data_path": "data/fe_sent_truth.csv",  # ✅ 统一管理
        "fe_2d_path": "data/fe_sent_truth.csv",  # 2D 全场真值 (Abaqus/FEniCS/Mock)
        "fe_1d_path": "data/sent_fe_midline_1d.npz",  # 1D 中线位移真值
        "ckpt_path": "outputs/phase2_xras_checkpoint.pth",  # 训练好的模型路径
    }

    # =====================================================================
    # DEBUG 模式 - 快速调参
    # =====================================================================
    if debug:
        print(">> Running in DEBUG_MODE (快速调参配置)")
        # base_config.update({
        #     # ================================================================
        #     # 相场材料参数
        #     # ================================================================
        #     "G_c": 20,             # 断裂能 (略大,裂纹容易扩展)
        #     "l": 0.005,                # 相场长度尺度(50mm裂纹）
        #
        #
        #     # ================================================================
        #     # 采样 & 载荷 (Phase-1)
        #     # ================================================================
        #     "n_domain": 2000,          # 域内采样点数
        #     "n_bc": 100,               # 边界采样点数
        #     "max_displacement": 0.0003, # 1mm 对应应变0.1%
        #     "n_loading_steps": 10,      # 载荷步数
        #
        #     # ================================================================
        #     # notch 初始化 (Phase-1)
        #     # ================================================================
        #     "notch_seed_radius": 0.025, # 高斯核半径
        #     "initial_d": 0.5,          # 初始损伤峰值
        #     "notch_init_epochs": 700,  # notch 预训练 epoch
        #
        #     # ================================================================
        #     # 准静态训练 (Phase-1)
        #     # ================================================================
        #     "n_epochs_initial": 700,   # 前几个 load step 的 epoch
        #     "n_epochs_later": 500,     # 后面 load step 的 epoch
        #     "n_epochs_switch": 3,      # 在第几步切换 epoch (n < 3 用 initial)
        #     "weight_irrev_phase1": 0.0,  # Phase-1 不可逆约束权重
        #
        #     # notch 保持项 (第1步强化 notch 区域损伤)
        #     "notch_region_radius": 0.02,
        #     "notch_hold_weight": 10.0,
        #     "notch_hold_target": 0.8,
        #
        #     # 远场区域半径 (用于 d_far 统计)
        #     "far_region_radius": 0.25,
        #
        #     # ================================================================
        #     # Phase-2: VPINN baseline
        #     # ================================================================
        #     "n_epochs_vpinn": 800,     # VPINN 单域训练总 epoch
        #
        #     # ================================================================
        #     # Phase-2: X-RAS-PINN
        #     # ================================================================
        #     "n_epochs_phase1": 400,    # Phase 1: 预训练
        #     "n_epochs_phase2": 400,    # Phase 2: 聚焦 + RAS
        #     "n_epochs_phase3": 200,    # Phase 3: 联合微调
        #
        #     "r_sing": 0.04,            # 奇异域半径
        #     "N_add_ras": 600,          # RAS 增加的点数
        #
        #     "indicator_beta": 0.6,     # 指标函数权重 (d_grad vs energy)
        #     "weight_interface": 50.0,  # 接口连续性损失权重
        #     "weight_irrev_phase2": 0.0, # Phase-2 不可逆约束权重 (通常为0)
        #
        #     # ================================================================
        #     # 实验控制开关 (Phase-2 消融实验用)
        #     # ================================================================
        #     "use_partition": True,     # 是否使用域分解
        #     "use_ras": True,           # 是否使用自适应采样
        #     "use_interface_loss": True, # 是否使用接口损失
        #     "budget_mode": False,      # 是否固定总点数预算
        #
        #     # ================================================================
        #     # Phase-2 与 Phase-1 集成 (关键改进)
        #     # ================================================================
        #     "use_phase1_result": True,     # 是否从 Phase-1 加载结果
        #     "phase1_load_step": -1,        # 使用哪一步 (-1=最后一步, 或指定步数)
        #     "phase1_model_path": "outputs/phase1_checkpoint.pth",  # Phase-1 模型路径
        #     "phase2_refine_epochs": 500,   # 在 Phase-1 基础上的精细化训练
        #     "field_init_epochs": 300,      # 场初始化的拟合 epoch
        #
        #     # ================================================================
        #     # 误差分析 (Phase-2)
        #     # ================================================================
        #     "r_near": 0.1,             # 近场误差半径
        # })

    # =====================================================================
    # FULL 模式 - 论文级精细配置
    # =====================================================================
    else:
        print(">> Running in FULL MODE (精细实验配置)")
        base_config.update({
            # ================================================================
            # 相场材料参数 (更精细)
            # ================================================================
            "G_c": 30,             # 断裂能 (较小,裂纹更集中)
            "l": 0.004,                # 相场长度尺度

            # ================================================================
            # 采样 & 载荷 (Phase-1) - 更密集
            # ================================================================
            "max_displacement": 0.003,  # 最大位移载荷
            "n_loading_steps": 10,  # 载荷步数

            "initial_d": 0.35,  # 初始损伤峰值
            "n_domain": 6000,          # 域内采样点数
            "n_notch": 800,
            "n_tip": 800,
            "n_bc": 400,               # 边界采样点数
            "tip_radius": 0.01, # 高斯核半径, 0.5 * 特征长度l 到 1 * l
            "notch_seed_radius" : 0.002,

            # Patch B
            "enable_nucleation_threshold": True,
            "nucleation_threshold_tau": 0.0,

            # Patch C
            "c_0": 1.0,
            "k": 1e-5,

            # Patch D
            "stagger_u_steps": 200,
            "stagger_d_steps": 100,

            # ================================================================
            # 准静态训练 (Phase-1) - 更多 epoch
            # ================================================================
            "notch_init_epochs" : 1000,
            "n_epochs_initial": 300,  # 前几个 load step
            "n_epochs_later": 800,     # 后面 load step
            "n_epochs_switch": 0,      # 切换步数
            "weight_irrev_phase1": 1200.0,

            # notch 保持项
            "notch_region_radius": 0.03, # l的5-10倍
            "notch_hold_weight": 5000.0,
            "notch_hold_target": 1.0,

            # 远场区域
            "far_region_radius": 0.1,

            # ================================================================
            # Phase-2: VPINN baseline
            # ================================================================
            "n_epochs_vpinn": 2000,    # 更多 epoch

            # ================================================================
            # Phase-2: X-RAS-PINN
            # ================================================================
            "n_epochs_phase1": 800,
            "n_epochs_phase2": 800,
            "n_epochs_phase3": 400,

            "r_sing": 0.04,
            "N_add_ras": 800,          # 更多 RAS 点

            "indicator_beta": 0.6,
            "weight_interface": 50.0,
            "weight_irrev_phase2": 0.0,

            # ================================================================
            # 实验控制
            # ================================================================
            "use_partition": True,
            "use_ras": True,
            "use_interface_loss": True,
            "budget_mode": False,

            # ================================================================
            # Phase-2 与 Phase-1 集成
            # ================================================================
            "use_phase1_result": True,
            "phase1_load_step": -1,
            "phase1_model_path": "outputs/phase1_checkpoint.pth",
            "phase2_refine_epochs": 800,
            "field_init_epochs": 500,

            # ================================================================
            # 误差分析
            # ================================================================
            "r_near": 0.1,


        })

    return base_config


def print_config(config):
    """
    打印配置摘要

    Args:
        config: dict - 配置字典
    """
    print("\n" + "=" * 70)
    print("  Configuration Summary")
    print("=" * 70)

    # 几何 & 材料
    print("\n  [1] Geometry & Material:")
    print(f"    L × H:        {config['L']:.2f} × {config['H']:.2f}")
    print(f"    Notch length: {config['notch_length']:.2f}")
    print(f"    E:            {config['E']:.1f}")
    print(f"    ν:            {config['nu']:.2f}")

    # 相场参数
    print("\n  [2] Phase-field:")
    print(f"    G_c:          {config['G_c']:.3e}")
    print(f"    l:            {config['l']:.4f}")

    # Phase-1 采样
    print("\n  [3] Phase-1 Sampling & Loading:")
    print(f"    n_domain:     {config['n_domain']}")
    print(f"    n_bc:         {config['n_bc']}")
    print(f"    Load steps:   {config['n_loading_steps']}")
    print(f"    Max δ:        {config['max_displacement']:.4f}")

    # Phase-1 训练
    print("\n  [4] Phase-1 Training:")
    print(f"    Notch init:   {config['notch_init_epochs']} epochs")
    print(f"    Initial:      {config['n_epochs_initial']} epochs")
    print(f"    Later:        {config['n_epochs_later']} epochs")
    print(f"    Weight irrev: {config['weight_irrev_phase1']:.1f}")

    # Phase-2 VPINN
    print("\n  [5] Phase-2 VPINN baseline:")
    print(f"    Epochs:       {config['n_epochs_vpinn']}")

    # Phase-2 X-RAS
    print("\n  [6] Phase-2 X-RAS-PINN:")
    print(f"    Phase 1:      {config['n_epochs_phase1']} epochs")
    print(f"    Phase 2:      {config['n_epochs_phase2']} epochs")
    print(f"    Phase 3:      {config['n_epochs_phase3']} epochs")
    print(f"    r_sing:       {config['r_sing']:.2f}")
    print(f"    N_add (RAS):  {config['N_add_ras']}")

    # 实验控制
    print("\n  [7] Experiment Controls:")
    print(f"    Partition:    {config['use_partition']}")
    print(f"    RAS:          {config['use_ras']}")
    print(f"    Interface:    {config['use_interface_loss']}")
    print(f"    Budget mode:  {config['budget_mode']}")

    # 其他
    print("\n  [8] Misc:")
    print(f"    Device:       {config['device']}")
    print(f"    Seed:         {config['seed']}")
    print(f"    lr_u / lr_d:  {config['lr_u']:.1e} / {config['lr_d']:.1e}")

    print("=" * 70 + "\n")


def get_experiment_configs():
    """
    返回几组预定义的实验配置 (用于参数扫描)

    Returns:
        experiments: list of dict
            每个 dict 包含实验名称和对应的配置修改
    """
    experiments = []

    # 实验 2.1: 基础对比 (默认配置)
    experiments.append({
        "name": "baseline_comparison",
        "description": "VPINN vs X-RAS-PINN with default settings",
        "config_updates": {}
    })

    # 实验 2.2: 预算模式 (固定总点数)
    experiments.append({
        "name": "fixed_budget",
        "description": "Fixed total point budget comparison",
        "config_updates": {
            "budget_mode": True,
            "n_domain": 1500,      # 总预算
            "N_add_ras": 500,      # X-RAS 额外点数
        }
    })

    # 实验 2.3a: 消融 - 无 RAS
    experiments.append({
        "name": "ablation_no_ras",
        "description": "Domain partition only (no adaptive sampling)",
        "config_updates": {
            "use_ras": False,
            "N_add_ras": 0,
        }
    })

    # 实验 2.3b: 消融 - 无域分解
    experiments.append({
        "name": "ablation_no_partition",
        "description": "Single domain with RAS",
        "config_updates": {
            "use_partition": False,
        }
    })

    # 实验 2.3c: 消融 - 完整 X-RAS
    experiments.append({
        "name": "ablation_full",
        "description": "Full X-RAS-PINN (partition + RAS + interface)",
        "config_updates": {
            "use_partition": True,
            "use_ras": True,
            "use_interface_loss": True,
        }
    })

    # 实验 2.4a: r_sing 扫描
    for r_sing in [0.10, 0.15, 0.20]:
        experiments.append({
            "name": f"param_scan_rsing_{r_sing:.2f}",
            "description": f"r_sing = {r_sing}",
            "config_updates": {
                "r_sing": r_sing,
            }
        })

    # 实验 2.4b: N_add 扫描
    for N_add in [300, 600, 900]:
        experiments.append({
            "name": f"param_scan_nadd_{N_add}",
            "description": f"N_add = {N_add}",
            "config_updates": {
                "N_add_ras": N_add,
            }
        })

    return experiments


# ============================================================================
# 辅助函数: 应用配置更新
# ============================================================================

def apply_config_updates(base_config, updates):
    """
    在基础配置上应用更新

    Args:
        base_config: dict - 基础配置
        updates: dict - 需要更新的字段

    Returns:
        new_config: dict - 更新后的配置
    """
    import copy
    new_config = copy.deepcopy(base_config)
    new_config.update(updates)
    return new_config


# ============================================================================
# 测试
# ============================================================================

# if __name__ == "__main__":
#     print("\n" + "=" * 70)
#     print("  Config Module Test")
#     print("=" * 70)
#
#     # 测试 DEBUG 模式
#     print("\n[1/3] Testing DEBUG mode...")
#     config_debug = create_config(debug=True)
#     print_config(config_debug)
#
#     # 测试 FULL 模式
#     print("\n[2/3] Testing FULL mode...")
#     config_full = create_config(debug=False)
#     print_config(config_full)
#
#     # 测试实验配置
#     print("\n[3/3] Testing experiment configurations...")
#     experiments = get_experiment_configs()
#     print(f"\nTotal predefined experiments: {len(experiments)}")
#     for i, exp in enumerate(experiments[:3], 1):  # 只打印前3个
#         print(f"  {i}. {exp['name']}: {exp['description']}")
#     print(f"  ... and {len(experiments)-3} more")
#
#     print("\n" + "=" * 70)
#     print("  Config module test completed!")
#     print("=" * 70)
