"""
t-SNE可视化快速演示脚本
展示如何使用可视化工具的各种功能
"""

import argparse
import sys
from pathlib import Path


def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("GDN t-SNE 可视化工具 - 快速演示")
    print("="*60)
    print("\n选择运行模式:\n")
    print("  1. 快速模式 (2-5分钟)")
    print("     - 适合快速测试")
    print("     - perplexity=15, n_iter=500, sample_size=2000")
    print()
    print("  2. 平衡模式 (10-15分钟) [推荐]")
    print("     - 速度和质量平衡")
    print("     - perplexity=30, n_iter=1000, sample_size=5000")
    print()
    print("  3. 高质量模式 (30+分钟)")
    print("     - 最佳可视化效果")
    print("     - perplexity=50, n_iter=1500, sample_size=10000")
    print()
    print("  4. 自定义参数")
    print()
    print("  0. 退出")
    print()
    print("-"*60)


def get_mode_config(mode):
    """根据模式返回配置"""
    configs = {
        1: {
            'name': '快速模式',
            'perplexity': 15,
            'n_iter': 500,
            'sample_size': 2000
        },
        2: {
            'name': '平衡模式',
            'perplexity': 30,
            'n_iter': 1000,
            'sample_size': 5000
        },
        3: {
            'name': '高质量模式',
            'perplexity': 50,
            'n_iter': 1500,
            'sample_size': 10000
        }
    }
    return configs.get(mode)


def run_visualization(dataset='swat', perplexity=30, n_iter=1000, sample_size=5000):
    """运行可视化脚本"""
    import subprocess
    
    cmd = [
        'python', 'visualize_tsne.py',
        '--dataset', dataset,
        '--perplexity', str(perplexity),
        '--n_iter', str(n_iter),
        '--output_dir', './tsne_results',
        '--batch_size', '256',
        '--sample_size', str(sample_size)
    ]
    
    print("\n执行命令:")
    print(" ".join(cmd))
    print("\n开始处理... 请耐心等待\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 可视化失败")
        print(f"错误信息: {e}")
        return False


def main():
    """主函数"""
    print("\n")
    print("█" * 60)
    print("█" + " "*58 + "█")
    print("█" + "   GDN t-SNE 可视化工具 - 快速启动向导".center(58) + "█")
    print("█" + " "*58 + "█")
    print("█" * 60)
    
    while True:
        print_menu()
        
        try:
            choice = input("请选择 (0-4): ").strip()
            
            if choice == '0':
                print("\n再见！")
                sys.exit(0)
            
            elif choice in ['1', '2', '3']:
                mode = int(choice)
                config = get_mode_config(mode)
                
                print(f"\n✓ 已选择: {config['name']}")
                print(f"  参数: perplexity={config['perplexity']}, "
                      f"n_iter={config['n_iter']}, "
                      f"sample_size={config['sample_size']}")
                
                # 选择数据集
                print("\n选择数据集:")
                print("  1. SWAT (默认)")
                print("  2. MSL")
                dataset_choice = input("选择 (1-2) [默认1]: ").strip() or "1"
                
                dataset = 'msl' if dataset_choice == '2' else 'swat'
                print(f"\n✓ 已选择数据集: {dataset.upper()}")
                
                # 确认执行
                print(f"\n准备执行 {config['name']}:")
                print(f"  数据集: {dataset.upper()}")
                print(f"  困惑度: {config['perplexity']}")
                print(f"  迭代次数: {config['n_iter']}")
                print(f"  采样数: {config['sample_size']}")
                
                confirm = input("\n是否继续? (y/n) [默认y]: ").strip().lower() or 'y'
                
                if confirm == 'y':
                    success = run_visualization(
                        dataset=dataset,
                        perplexity=config['perplexity'],
                        n_iter=config['n_iter'],
                        sample_size=config['sample_size']
                    )
                    
                    if success:
                        print("\n" + "="*60)
                        print("✓ 可视化完成!")
                        print("="*60)
                        print("\n结果保存在: ./tsne_results/")
                        print("\n生成的文件:")
                        print("  - {}_tsne_by_label.png".format(dataset))
                        print("  - {}_tsne_by_density.png".format(dataset))
                        print("  - {}_embedding_distribution.png".format(dataset))
                        print("  - {}_tsne_results.csv".format(dataset))
                        print("  - {}_embeddings_sample.csv".format(dataset))
                    else:
                        print("\n✗ 可视化失败，请检查错误信息")
                else:
                    print("\n已取消")
                
                cont = input("\n是否继续? (y/n) [默认n]: ").strip().lower()
                if cont != 'y':
                    print("\n再见！")
                    sys.exit(0)
            
            elif choice == '4':
                print("\n自定义参数配置:")
                
                dataset = input("数据集 (swat/msl) [默认swat]: ").strip().lower() or 'swat'
                if dataset not in ['swat', 'msl']:
                    dataset = 'swat'
                
                try:
                    perplexity = int(input("困惑度 (5-100) [默认30]: ") or "30")
                    n_iter = int(input("迭代次数 (300-2000) [默认1000]: ") or "1000")
                    sample_size = int(input("采样数 (1000-20000) [默认5000]: ") or "5000")
                except ValueError:
                    print("输入错误，使用默认值")
                    perplexity, n_iter, sample_size = 30, 1000, 5000
                
                print(f"\n确认参数:")
                print(f"  数据集: {dataset.upper()}")
                print(f"  困惑度: {perplexity}")
                print(f"  迭代次数: {n_iter}")
                print(f"  采样数: {sample_size}")
                
                confirm = input("\n是否继续? (y/n): ").strip().lower() or 'y'
                if confirm == 'y':
                    success = run_visualization(
                        dataset=dataset,
                        perplexity=perplexity,
                        n_iter=n_iter,
                        sample_size=sample_size
                    )
                    if success:
                        print("\n✓ 可视化完成!")
                    else:
                        print("\n✗ 可视化失败")
                
                cont = input("\n是否继续? (y/n) [默认n]: ").strip().lower()
                if cont != 'y':
                    print("\n再见！")
                    sys.exit(0)
            
            else:
                print("\n✗ 无效选择，请重试")
        
        except KeyboardInterrupt:
            print("\n\n已中断")
            sys.exit(0)
        except Exception as e:
            print(f"\n✗ 发生错误: {e}")
            print("请重试")


if __name__ == '__main__':
    main()
