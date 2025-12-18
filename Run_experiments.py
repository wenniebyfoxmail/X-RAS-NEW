#!/usr/bin/env python3
"""
run_experiments.py
一键运行完整实验流程

实验包括:
1. Phase-1 (DEBUG/FULL)
2. Phase-2 基础对比
3. Phase-2 预算模式
4. Phase-2 消融实验
5. Phase-2 参数扫描

使用方法:
    python run_experiments.py --exp all          # 运行所有实验
    python run_experiments.py --exp phase1       # 只运行 Phase-1
    python run_experiments.py --exp ablation     # 只运行消融实验
    python run_experiments.py --mode full        # 使用 FULL 模式
"""

import argparse
import os
import sys
import time
from pathlib import Path

# 确保模块可导入
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import create_config, print_config, apply_config_updates
# 导入核心模块
from solver_pinn import (
    DisplacementNetwork, DamageNetwork, PhaseFieldSolver,
    generate_domain_points
)
# 导入 X-RAS 求解器
from solver_xras import XRASPINNSolver, SubdomainModels

def run_phase1(mode="full"):
    """运行 Phase-1 实验"""
    print("\n" + "="*80)
    print(f"  实验 1: Phase-1 SENT with Notch ({mode.upper()} 模式)")
    print("="*80)

    debug = (mode == "debug")

    my_config = create_config()

    try:
        # 动态导入避免全局依赖
        from test_sent_pinn import test_sent_with_notch

        t0 = time.time()
        solver, history = test_sent_with_notch(config = my_config, debug=debug)
        t1 = time.time()

        print(f"\n✓ Phase-1 完成! 耗时: {t1-t0:.1f}s")
        return True
    except Exception as e:
        print(f"\n❌ Phase-1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_phase2_baseline(mode="debug", input_updates={}, exp_name="Baseline"):
    """运行 Phase-2 基础对比实验"""
    print("\n" + "="*80)
    print(f"  实验 2: Phase-2 Baseline Comparison ({mode.upper()} 模式)")
    print("="*80)

    debug = (mode == "debug")

    try:
        from test_sent_xras import test_sent_phase2

        # 1. 创建基础配置
        base_config = create_config(debug=debug)
        # 2. 应用外部更新 (用于消融实验的配置)
        config = apply_config_updates(base_config, input_updates)  #

        t0 = time.time()
        # 3. 传入配置字典
        success = test_sent_phase2(debug=debug, input_config=config, exp_name=exp_name)  #
        t1 = time.time()

        if success:
            print(f"\n✓ Phase-2 Baseline 完成! 耗时: {t1 - t0:.1f}s")
        return success
    except Exception as e:
        print(f"\n❌ Phase-2 Baseline 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_ablation_experiments(mode="debug"):
    """运行消融实验"""
    print("\n" + "="*80)
    print(f"  实验 3: Ablation Studies ({mode.upper()} 模式)")
    print("="*80)


    debug = (mode == "debug")

    # 定义消融实验
    ablation_configs = [
        {
            "name": "Full-XRAS",
            "description": "完整 X-RAS-PINN (Partition + RAS + Interface)",
            "updates": {
                "use_partition": True,
                "use_ras": True,
                "use_interface_loss": True,
            }
        },
        {
            "name": "No-Interface-Loss",
            "description": "X-RAS 无接口连续性损失",
            "updates": {
                "use_partition": True,
                "use_ras": True,
                "use_interface_loss": False,
                # 强制 r_sing 区域的两个网络都使用 Phase-1 初始化，防止解发散
                "field_init_epochs": 800,
            }
        },

        {
            "name": "No-RAS",
            "description": "域分解但无自适应采样 (固定粗网格, Ω_sing + Ω_far)",
            "updates": {
                "use_partition": True,
                "use_ras": False,
                "N_add_ras": 0,
                "use_interface_loss": True,
            }
        },

        {
            "name": "No-Partition",
            "description": "单域但有自适应采样 (VPINN + RAS)",
            "updates": {
                "use_partition": False,  # 禁用域分解
                "use_ras": True,  # 但保留 RAS
                "use_interface_loss": False,  # 单域无需接口损失
                # VPINN 只有一套网络，这里 N_add_ras 会被加到 VPINN 的点集中
            }
        },
    ]

    results = {}

    for exp in ablation_configs:
        print(f"\n{'='*70}")
        print(f"  消融实验: {exp['name']} - {exp['description']}")
        print(f"{'='*70}")

        exp_name = exp['name']  # ✅ 定义传递给 baseline 的名称

        # 应用配置更新
        # 核心修改：调用 run_phase2_baseline 并传入 updates
        config_name = exp['name'].lower().replace('-', '_')

        # 保存当前配置到临时文件供 test_sent_phase2 使用
        # (实际实现需要修改 test_sent_phase2 以接受 config 参数)
        print(f"  配置更新: {exp['updates']}")

        # 统一使用 run_phase2_baseline 函数, 传入 updates
        success = run_phase2_baseline(mode=mode, input_updates=exp["updates"], exp_name=exp_name)

        # ⚠️ 假设这里已经收集了量化指标 metrics
        results[exp_name] = "Success" if success else "Failure"

    return results


def run_parameter_sweep(mode="debug", param="r_sing"):
    """运行参数扫描实验"""
    print("\n" + "="*80)
    print(f"  实验 4: Parameter Sweep - {param} ({mode.upper()} 模式)")
    print("="*80)


   # base_config = create_config(debug=debug)

    # 定义参数扫描范围
    param_ranges = {
        "r_sing": [0.10, 0.15, 0.20],
        "N_add_ras": [300, 600, 900],
    }

    if param not in param_ranges:
        print(f"  ❌ 不支持的参数: {param}")
        return {}

    values = param_ranges[param]
    results = {}

    for value in values:
        print(f"\n{'='*70}")
        print(f"  参数扫描: {param} = {value}")
        print(f"{'='*70}")

        # 配置更新
        updates = {param: value}
        exp_name = f"{param}-{value}"  # ✅ 定义传递给 baseline 的名称 (例如 r_sing-0.10)

        # 核心修改：调用 run_phase2_baseline 并传入 updates
        success = run_phase2_baseline(mode=mode, input_updates=updates, exp_name=exp_name) # ✅ 传递 exp_name

        results[f"{param}={value}"] = "Success" if success else "Failure"

    return results


def main():
    parser = argparse.ArgumentParser(
        description="运行 SENT Phase-2 实验套件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验类型:
  phase1      - Phase-1: SENT with Notch
  phase2      - Phase-2: Baseline Comparison
  ablation    - 消融实验 (No-RAS, No-Partition, Full)
  sweep       - 参数扫描 (r_sing 或 N_add_ras)
  all         - 运行所有实验

示例:
  python run_experiments.py --exp phase1 --mode debug
  python run_experiments.py --exp ablation --mode full
  python run_experiments.py --exp sweep --param r_sing
  python run_experiments.py --exp all --mode full
        """
    )

    parser.add_argument(
        "--exp",
        type=str,
        default="phase1",
        choices=["phase1", "phase2", "ablation", "sweep", "all"],
        help="实验类型"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="debug",
        choices=["debug", "full"],
        help="运行模式: debug=快速测试, full=精细实验"
    )

    parser.add_argument(
        "--param",
        type=str,
        choices=["r_sing", "N_add_ras"],
        help="参数扫描的参数名（仅用于 --exp sweep）"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("  SENT Phase-2 实验套件")
    print("="*80)
    print(f"  实验类型: {args.exp}")
    print(f"  运行模式: {args.mode.upper()}")
    print("="*80)

    t_start = time.time()

    # 运行实验
    if args.exp == "phase1" or args.exp == "all":
        run_phase1(args.mode)

    if args.exp == "phase2" or args.exp == "all":
        run_phase2_baseline(args.mode)

    if args.exp == "ablation" or args.exp == "all":
        run_ablation_experiments(args.mode)

    if args.exp == "sweep":
        run_parameter_sweep(args.mode, args.param)

    t_end = time.time()

    print("\n" + "="*80)
    print(f"  实验完成! 总耗时: {t_end-t_start:.1f}s")
    print("="*80)


if __name__ == "__main__":
    main()