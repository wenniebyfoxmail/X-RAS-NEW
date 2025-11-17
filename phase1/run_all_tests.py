#!/usr/bin/env python
"""
一键运行脚本 - 快速验证所有功能

使用方法:
    python run_all_tests.py

或指定测试类型:
    python run_all_tests.py --quick      # 快速测试 (~1分钟)
    python run_all_tests.py --standard   # 标准测试 (~5分钟)
    python run_all_tests.py --full       # 完整测试 (~15分钟)
"""

import sys
import time
import argparse


def run_quick_test():
    """快速测试 - 验证基本功能"""
    print("\n" + "="*70)
    print("  QUICK TEST MODE (~1 minute)")
    print("="*70)
    
    from test_phase_field_vpinn import test_simple_convergence
    
    # 简单收敛测试
    test_simple_convergence()
    
    print("\n✓ Quick test completed!")


def run_standard_test():
    """标准测试 - SENT基准问题"""
    print("\n" + "="*70)
    print("  STANDARD TEST MODE (~5 minutes)")
    print("="*70)
    
    from test_phase_field_vpinn import test_sent_problem
    
    # SENT测试（减少epochs）
    solver, history = test_sent_problem(
        n_loading_steps=5,
        n_epochs_per_step=500
    )
    
    print("\n✓ Standard test completed!")
    return solver, history


def run_full_test():
    """完整测试 - 所有问题"""
    print("\n" + "="*70)
    print("  FULL TEST MODE (~15 minutes)")
    print("="*70)
    
    from test_phase_field_vpinn import test_sent_problem
    from example_custom_problem import solve_center_crack
    
    # SENT测试（更多epochs）
    print("\n[1/2] Running SENT benchmark...")
    solver1, history1 = test_sent_problem(
        n_loading_steps=10,
        n_epochs_per_step=1000
    )
    
    # 中心裂纹测试
    print("\n[2/2] Running center crack problem...")
    solver2, history2 = solve_center_crack(
        n_loading_steps=8,
        n_epochs_per_step=1000
    )
    
    print("\n✓ Full test completed!")
    return (solver1, history1), (solver2, history2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Phase-Field VPINN/DRM Solver - Test Runner'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test (convergence only, ~1 min)'
    )
    parser.add_argument(
        '--standard', action='store_true',
        help='Standard test (SENT benchmark, ~5 min)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Full test (all problems, ~15 min)'
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print("\n" + "="*70)
    print("  Phase-Field VPINN/DRM Solver - Automated Test Runner")
    print("="*70)
    print("\n  This script will test the solver implementation.")
    print("  Generated files will be saved to the outputs folder.\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 根据参数运行测试
    if args.quick:
        run_quick_test()
    elif args.full:
        run_full_test()
    else:  # 默认或 --standard
        run_standard_test()
    
    # 计算运行时间
    elapsed_time = time.time() - start_time
    
    # 打印总结
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print("  Status: SUCCESS ✓")
    print("\n  Generated files:")
    print("    - sent_result.png")
    print("    - damage_evolution.png")
    if args.full:
        print("    - center_crack_result.png")
    print("\n  Next steps:")
    print("    1. View the generated images")
    print("    2. Read PROJECT_README.md for theory")
    print("    3. Try example_custom_problem.py")
    print("    4. Create your own problem!")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
