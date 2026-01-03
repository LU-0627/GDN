# -*- coding: utf-8 -*-
"""
调试工具函数集合
提供常用的调试功能，如张量统计、数据可视化等
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def print_tensor_stats(tensor, name="Tensor", show_values=False):
    """
    打印张量的详细统计信息
    
    Args:
        tensor: PyTorch张量
        name: 张量名称
        show_values: 是否显示部分值
    """
    print(f"\n{'='*60}")
    print(f"📊 {name} 统计信息")
    print(f"{'='*60}")
    print(f"  Shape:        {tensor.shape}")
    print(f"  Dtype:        {tensor.dtype}")
    print(f"  Device:       {tensor.device}")
    print(f"  Range:        [{tensor.min():.4f}, {tensor.max():.4f}]")
    print(f"  Mean:         {tensor.mean():.4f}")
    print(f"  Std:          {tensor.std():.4f}")
    print(f"  Has NaN:      {torch.isnan(tensor).any()}")
    print(f"  Has Inf:      {torch.isinf(tensor).any()}")
    
    if show_values and tensor.numel() <= 100:
        print(f"\n  Values:\n{tensor}")
    elif show_values:
        print(f"\n  First 5 values: {tensor.flatten()[:5]}")
        print(f"  Last 5 values:  {tensor.flatten()[-5:]}")
    print(f"{'='*60}\n")


def compare_tensors(tensor1, tensor2, name1="Tensor1", name2="Tensor2"):
    """
    比较两个张量的差异
    
    Args:
        tensor1, tensor2: 要比较的张量
        name1, name2: 张量名称
    """
    print(f"\n{'='*60}")
    print(f"🔄 比较 {name1} vs {name2}")
    print(f"{'='*60}")
    
    if tensor1.shape != tensor2.shape:
        print(f"  ⚠️ Shape不同: {tensor1.shape} vs {tensor2.shape}")
        return
    
    diff = (tensor1 - tensor2).abs()
    print(f"  平均绝对误差:    {diff.mean():.6f}")
    print(f"  最大绝对误差:    {diff.max():.6f}")
    print(f"  相对误差 (%):    {(diff / (tensor1.abs() + 1e-8)).mean() * 100:.2f}%")
    print(f"  相同元素比例:    {(tensor1 == tensor2).float().mean() * 100:.2f}%")
    print(f"{'='*60}\n")


def plot_batch_distribution(tensor, title="", save_dir="./debug_plots"):
    """
    可视化batch中数据的分布
    
    Args:
        tensor: 输入张量 [batch, sensors, time] 或 [batch, sensors]
        title: 图表标题
        save_dir: 保存目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 转换为numpy
    data = tensor.cpu().detach().numpy()
    
    plt.figure(figsize=(15, 5))
    
    # 子图1: 第一个样本的数据
    plt.subplot(1, 3, 1)
    if len(data.shape) == 3:  # [batch, sensors, time]
        plt.plot(data[0, :, :].T)
        plt.xlabel("传感器索引")
        plt.ylabel("值")
        plt.title(f"{title} - 第一个样本 (所有时间步)")
    else:  # [batch, sensors]
        plt.bar(range(len(data[0])), data[0])
        plt.xlabel("传感器索引")
        plt.ylabel("值")
        plt.title(f"{title} - 第一个样本")
    plt.grid(True, alpha=0.3)
    
    # 子图2: 所有样本的均值分布
    plt.subplot(1, 3, 2)
    if len(data.shape) == 3:
        mean_values = data.mean(axis=(1, 2))
    else:
        mean_values = data.mean(axis=1)
    plt.hist(mean_values, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("均值")
    plt.ylabel("样本数量")
    plt.title(f"{title} - 批次均值分布")
    plt.grid(True, alpha=0.3)
    
    # 子图3: 热力图 (传感器 × 批次)
    plt.subplot(1, 3, 3)
    if len(data.shape) == 3:
        heatmap_data = data.mean(axis=2)  # 对时间维度取平均
    else:
        heatmap_data = data
    plt.imshow(heatmap_data.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='值')
    plt.xlabel("批次索引")
    plt.ylabel("传感器索引")
    plt.title(f"{title} - 热力图")
    
    plt.tight_layout()
    filepath = Path(save_dir) / f"{title.replace(' ', '_')}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"📊 已保存可视化图片: {filepath}")


def plot_loss_history(loss_list, title="训练损失曲线", save_dir="./debug_plots"):
    """
    绘制损失曲线
    
    Args:
        loss_list: 损失值列表
        title: 图表标题
        save_dir: 保存目录
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, linewidth=1.5)
    plt.xlabel("迭代次数")
    plt.ylabel("损失值")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 添加移动平均线
    if len(loss_list) > 50:
        window = 50
        moving_avg = np.convolve(loss_list, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(loss_list)), moving_avg, 
                'r--', linewidth=2, label=f'{window}步移动平均')
        plt.legend()
    
    filepath = Path(save_dir) / f"{title.replace(' ', '_')}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"📈 已保存损失曲线: {filepath}")


def check_gradients(model, threshold=10.0):
    """
    检查模型梯度是否正常
    
    Args:
        model: PyTorch模型
        threshold: 梯度阈值，超过此值会发出警告
    
    Returns:
        grad_info: 梯度信息字典
    """
    print(f"\n{'='*60}")
    print(f"🔍 梯度检查")
    print(f"{'='*60}")
    
    total_params = 0
    total_grad_norm = 0
    max_grad = 0
    min_grad = float('inf')
    
    grad_info = {
        'has_nan': False,
        'has_inf': False,
        'exploding': False,
        'vanishing': False
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_params += param.numel()
            grad = param.grad
            grad_norm = grad.norm().item()
            total_grad_norm += grad_norm
            
            max_grad = max(max_grad, grad.abs().max().item())
            min_grad = min(min_grad, grad.abs().min().item())
            
            # 检查异常
            if torch.isnan(grad).any():
                print(f"  ⚠️ {name}: 包含NaN梯度")
                grad_info['has_nan'] = True
            
            if torch.isinf(grad).any():
                print(f"  ⚠️ {name}: 包含Inf梯度")
                grad_info['has_inf'] = True
            
            if grad_norm > threshold:
                print(f"  ⚠️ {name}: 梯度过大 (norm={grad_norm:.2f})")
                grad_info['exploding'] = True
            
            if grad_norm < 1e-7:
                print(f"  ⚠️ {name}: 梯度过小 (norm={grad_norm:.2e})")
                grad_info['vanishing'] = True
    
    avg_grad_norm = total_grad_norm / len(list(model.parameters()))
    
    print(f"\n  总参数数量:      {total_params:,}")
    print(f"  平均梯度范数:    {avg_grad_norm:.6f}")
    print(f"  最大梯度值:      {max_grad:.6f}")
    print(f"  最小梯度值:      {min_grad:.6e}")
    print(f"{'='*60}\n")
    
    return grad_info


def watch_variable(var, var_name="variable", epoch=None, batch=None, log_file="./debug_watch.log"):
    """
    持续监控某个变量的变化，并记录到文件
    
    Args:
        var: 要监控的变量（支持Tensor、数值等）
        var_name: 变量名
        epoch: 当前epoch
        batch: 当前batch
        log_file: 日志文件路径
    """
    import datetime
    
    # 准备日志信息
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    location = f"Epoch {epoch}, Batch {batch}" if epoch is not None else ""
    
    # 转换变量为字符串
    if isinstance(var, torch.Tensor):
        var_str = f"Tensor(shape={var.shape}, mean={var.mean():.4f}, std={var.std():.4f})"
    else:
        var_str = str(var)
    
    # 写入日志
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {location} - {var_name}: {var_str}\n")


if __name__ == "__main__":
    # 测试代码
    print("调试工具测试")
    
    # 创建测试张量
    x = torch.randn(32, 38, 15)
    y = torch.randn(32, 38)
    
    # 测试统计信息打印
    print_tensor_stats(x, "测试输入张量x", show_values=True)
    print_tensor_stats(y, "测试输出张量y")
    
    # 测试可视化
    plot_batch_distribution(x, "测试输入分布")
    plot_batch_distribution(y, "测试输出分布")
    
    # 测试损失曲线
    loss_list = [0.5 * (0.95 ** i) + 0.01 * np.random.randn() for i in range(1000)]
    plot_loss_history(loss_list, "测试损失曲线")
    
    print("\n✅ 所有测试完成！")
