"""
t-SNE可视化工具 - 依赖检查和安装脚本
"""

import sys
import subprocess
import importlib


def check_package(package_name, import_name=None):
    """检查包是否安装"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"  ✓ {package_name:20} {version}")
        return True
    except ImportError:
        print(f"  ✗ {package_name:20} 未安装")
        return False


def main():
    print("\n" + "="*60)
    print("GDN t-SNE 可视化工具 - 依赖检查")
    print("="*60 + "\n")
    
    print("检查依赖包...\n")
    
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
    ]
    
    optional_packages = [
        ('seaborn', 'seaborn'),
        ('pytorch-geometric', 'torch_geometric'),
    ]
    
    print("必需包:")
    missing_required = []
    for pkg_name, import_name in required_packages:
        if not check_package(pkg_name, import_name):
            missing_required.append(pkg_name)
    
    print("\n可选包:")
    for pkg_name, import_name in optional_packages:
        check_package(pkg_name, import_name)
    
    # 检查GPU
    print("\nGPU支持:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA 已启用")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA版本: {torch.version.cuda}")
        else:
            print(f"  ⚠ CUDA 不可用（将使用CPU）")
    except:
        print(f"  ✗ 无法检查CUDA")
    
    # 处理缺失的包
    if missing_required:
        print("\n" + "="*60)
        print("缺失必需包！需要安装:")
        for pkg in missing_required:
            print(f"  - {pkg}")
        print("="*60)
        
        install = input("\n是否现在安装缺失的包? (y/n) [默认n]: ").strip().lower()
        
        if install == 'y':
            print("\n安装中...\n")
            for pkg in missing_required:
                print(f"安装 {pkg}...", end=" ", flush=True)
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
                    print("✓")
                except:
                    print("✗ 失败")
            
            print("\n重新检查依赖...\n")
            for pkg_name, import_name in required_packages:
                check_package(pkg_name, import_name)
        else:
            print("\n请手动安装缺失的包:")
            print(f"  pip install {' '.join(missing_required)}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("✓ 所有依赖检查完成！")
    print("="*60)
    
    print("\n现在可以运行t-SNE可视化工具:")
    print("\n  python demo_tsne.py            (交互式演示)")
    print("  python visualize_tsne.py       (完整控制)")
    print("  bash run_tsne.sh               (快速启动)")
    print("\n详细说明请参考: TSNE_QUICK_START.md")
    print()


if __name__ == '__main__':
    main()
