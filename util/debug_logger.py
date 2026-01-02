# -*- coding: utf-8 -*-
"""
GDN 调试日志模块
用于在训练和测试过程中输出详细的日志信息，帮助初学者理解模型运行过程。

使用方法:
    from util.debug_logger import DebugLogger
    logger = DebugLogger(debug=True, log_dir='./logs', dataset_name='msl')
    logger.log_section("数据加载")
    logger.log("训练样本数", 1000)
    logger.log_tensor("输入x", x)
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import torch
import numpy as np


class Colors:
    """终端彩色输出 (Windows兼容)"""
    HEADER = '\033[95m'      # 紫色
    BLUE = '\033[94m'        # 蓝色
    CYAN = '\033[96m'        # 青色
    GREEN = '\033[92m'        # 绿色
    YELLOW = '\033[93m'      # 黄色
    RED = '\033[91m'         # 红色
    ENDC = '\033[0m'         # 结束
    BOLD = '\033[1m'         # 粗体
    
    @classmethod
    def disable(cls):
        """禁用颜色（用于文件输出）"""
        cls.HEADER = ''
        cls.BLUE = ''
        cls.CYAN = ''
        cls.GREEN = ''
        cls.YELLOW = ''
        cls.RED = ''
        cls.ENDC = ''
        cls.BOLD = ''


class DebugLogger:
    """
    调试日志器
    支持控制台彩色输出和文件记录
    """
    
    def __init__(self, debug=False, log_dir='./logs', dataset_name='default', 
                 debug_batch=1, debug_forward=False):
        """
        初始化日志器
        
        Args:
            debug: 是否开启调试模式
            log_dir: 日志保存目录
            dataset_name: 数据集名称（用于日志文件命名）
            debug_batch: 每N个batch打印一次
            debug_forward: 是否打印forward内部细节
        """
        self.debug = debug
        self.debug_batch = debug_batch
        self.debug_forward = debug_forward
        self.indent_level = 0
        self.log_file = None
        self.start_time = datetime.now()
        
        if self.debug:
            # 创建日志目录
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成唯一的日志文件名（处理重名问题）
            self.log_filename = self._get_unique_filename(dataset_name)
            self.log_file = open(self.log_filename, 'w', encoding='utf-8')
            
            # Windows 启用ANSI颜色支持
            if sys.platform == 'win32':
                os.system('color')
            
            self._print_header()
    
    def _get_unique_filename(self, dataset_name):
        """
        生成唯一的日志文件名，避免重名覆盖
        格式: dataset_YYYYMMDD_HHMMSS_序号.log
        """
        timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
        base_name = f"{dataset_name}_{timestamp}"
        
        # 检查是否存在同名文件，添加序号
        index = 0
        while True:
            if index == 0:
                filename = self.log_dir / f"{base_name}.log"
            else:
                filename = self.log_dir / f"{base_name}_{index}.log"
            
            if not filename.exists():
                return filename
            index += 1
    
    def _print_header(self):
        """打印日志头部信息"""
        header = f"""
{'═' * 60}
  GDN 调试日志 - Debug Log
  开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
  日志文件: {self.log_filename}
{'═' * 60}
"""
        self._write(header)
    
    def _write(self, message, to_console=True, to_file=True):
        """写入消息到控制台和文件"""
        if not self.debug:
            return
            
        if to_console:
            print(message, flush=True)
        
        if to_file and self.log_file:
            # 文件中不写入颜色代码
            clean_msg = message
            for attr in ['HEADER', 'BLUE', 'CYAN', 'GREEN', 'YELLOW', 'RED', 'ENDC', 'BOLD']:
                clean_msg = clean_msg.replace(getattr(Colors, attr), '')
            self.log_file.write(clean_msg + '\n')
            self.log_file.flush()
    
    def log_section(self, title, icon='📍'):
        """
        打印分节标题
        
        Args:
            title: 标题文字
            icon: 前置图标
        """
        if not self.debug:
            return
        
        msg = f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.ENDC}"
        msg += f"\n{Colors.BOLD}{icon} [{title}]{Colors.ENDC}"
        msg += f"\n{Colors.CYAN}{'═' * 60}{Colors.ENDC}"
        self._write(msg)
        self.indent_level = 0
    
    def log_subsection(self, title, icon='📌'):
        """打印小节标题"""
        if not self.debug:
            return
        
        msg = f"\n{Colors.YELLOW}  {icon} {title}{Colors.ENDC}"
        self._write(msg)
    
    def log(self, key, value=None, level=0):
        """
        打印键值对日志
        
        Args:
            key: 键名
            value: 值（可选）
            level: 缩进级别
        """
        if not self.debug:
            return
        
        indent = '  ' * (self.indent_level + level)
        tree_char = '├─' if level > 0 else ''
        
        if value is None:
            msg = f"{indent}{tree_char}{key}"
        else:
            msg = f"{indent}{tree_char}{Colors.GREEN}{key}{Colors.ENDC}: {value}"
        
        self._write(msg)
    
    def log_tensor(self, name, tensor, show_stats=True, level=0):
        """
        打印张量信息
        
        Args:
            name: 张量名称
            tensor: PyTorch张量或Numpy数组
            show_stats: 是否显示统计信息
            level: 缩进级别
        """
        if not self.debug:
            return
        
        indent = '  ' * (self.indent_level + level)
        
        if isinstance(tensor, torch.Tensor):
            shape = list(tensor.shape)
            dtype = str(tensor.dtype).replace('torch.', '')
            device = str(tensor.device)
            
            msg = f"{indent}{Colors.BLUE}{name}{Colors.ENDC}: shape={shape}, dtype={dtype}, device={device}"
            
            if show_stats and tensor.numel() > 0:
                t = tensor.float()
                stats = f" | min={t.min().item():.4f}, max={t.max().item():.4f}, mean={t.mean().item():.4f}"
                msg += stats
                
        elif isinstance(tensor, np.ndarray):
            shape = list(tensor.shape)
            dtype = str(tensor.dtype)
            
            msg = f"{indent}{Colors.BLUE}{name}{Colors.ENDC}: shape={shape}, dtype={dtype}"
            
            if show_stats and tensor.size > 0:
                stats = f" | min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}"
                msg += stats
        else:
            msg = f"{indent}{Colors.BLUE}{name}{Colors.ENDC}: {type(tensor)}"
        
        self._write(msg)
    
    def log_dict(self, d, title=None, level=0):
        """打印字典"""
        if not self.debug:
            return
        
        if title:
            self.log(title, level=level)
        
        for key, value in d.items():
            self.log(f"  {key}", value, level=level)
    
    def log_model_summary(self, model):
        """打印模型摘要"""
        if not self.debug:
            return
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log_subsection("模型参数统计")
        self.log("总参数量", f"{total_params:,}")
        self.log("可训练参数", f"{trainable_params:,}")
        self.log("不可训练参数", f"{total_params - trainable_params:,}")
    
    def log_batch(self, batch_idx, total_batches, data_dict):
        """
        打印批次信息
        
        Args:
            batch_idx: 当前批次索引
            total_batches: 总批次数
            data_dict: 数据字典 {name: tensor}
        """
        if not self.debug:
            return
        
        # 根据 debug_batch 决定是否打印
        if (batch_idx + 1) % self.debug_batch != 0:
            return
        
        self.log_subsection(f"Batch {batch_idx + 1}/{total_batches}", icon='📦')
        for name, tensor in data_dict.items():
            self.log_tensor(name, tensor, level=1)
    
    def log_epoch_start(self, epoch, total_epochs):
        """打印Epoch开始"""
        if not self.debug:
            return
        
        self.log_section(f"Epoch {epoch + 1}/{total_epochs}", icon='🔄')
    
    def log_epoch_end(self, epoch, train_loss, val_loss=None, best=False, time_elapsed=None):
        """打印Epoch结束"""
        if not self.debug:
            return
        
        self.log_subsection("Epoch 完成", icon='✅')
        self.log("训练损失", f"{train_loss:.6f}", level=1)
        
        if val_loss is not None:
            status = f"{val_loss:.6f}"
            if best:
                status += f" {Colors.GREEN}✓ 新最优{Colors.ENDC}"
            self.log("验证损失", status, level=1)
        
        if time_elapsed is not None:
            self.log("耗时", f"{time_elapsed:.2f}s", level=1)
    
    def log_training_complete(self, total_epochs, best_loss, model_path):
        """打印训练完成"""
        if not self.debug:
            return
        
        self.log_section("训练完成", icon='🎉')
        self.log("总Epoch数", total_epochs)
        self.log("最佳验证损失", f"{best_loss:.6f}")
        self.log("模型保存路径", model_path)
    
    def log_test_start(self):
        """打印测试开始"""
        if not self.debug:
            return
        
        self.log_section("测试阶段", icon='🧪')
    
    def log_test_complete(self, avg_loss, total_samples, pred_stats, gt_stats):
        """打印测试完成"""
        if not self.debug:
            return
        
        self.log_subsection("测试完成", icon='✅')
        self.log("总样本数", total_samples, level=1)
        self.log("平均损失", f"{avg_loss:.6f}", level=1)
        self.log("预测值", f"min={pred_stats[0]:.4f}, max={pred_stats[1]:.4f}, mean={pred_stats[2]:.4f}", level=1)
        self.log("真实值", f"min={gt_stats[0]:.4f}, max={gt_stats[1]:.4f}, mean={gt_stats[2]:.4f}", level=1)
    
    def log_evaluation_result(self, metrics):
        """
        打印评估结果
        
        Args:
            metrics: 指标字典 {'f1': ..., 'precision': ..., 'recall': ..., 'auc': ..., 'threshold': ...}
        """
        if not self.debug:
            return
        
        self.log_section("评估结果", icon='🏆')
        
        # 打印表格
        line = f"\n{'─' * 40}"
        header = f"│ {'指标':<12} │ {'数值':<20} │"
        self._write(line)
        self._write(header)
        self._write(line)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                row = f"│ {key:<12} │ {value:<20.4f} │"
            else:
                row = f"│ {key:<12} │ {str(value):<20} │"
            self._write(row)
        
        self._write(line)
    
    def log_forward_step(self, step_name, tensor=None, extra_info=None):
        """打印Forward过程中的步骤"""
        if not self.debug or not self.debug_forward:
            return
        
        msg = f"    ├─ {step_name}"
        if tensor is not None:
            if isinstance(tensor, torch.Tensor):
                msg += f": {list(tensor.shape)}"
        if extra_info:
            msg += f" ({extra_info})"
        
        self._write(msg)
    
    def log_loss(self, loss, pred_range=None, gt_range=None):
        """打印损失信息"""
        if not self.debug:
            return
        
        self.log_subsection("损失计算", icon='📉')
        if pred_range:
            self.log("预测值范围", f"[{pred_range[0]:.4f}, {pred_range[1]:.4f}]", level=1)
        if gt_range:
            self.log("真实值范围", f"[{gt_range[0]:.4f}, {gt_range[1]:.4f}]", level=1)
        self.log("MSE Loss", f"{loss:.6f}", level=1)
    
    def log_gradient(self, model):
        """打印梯度信息"""
        if not self.debug or not self.debug_forward:
            return
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log_subsection("反向传播", icon='📈')
        self.log("梯度范数", f"{total_norm:.4f}", level=1)
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            footer = f"""
{'═' * 60}
  日志结束
  结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
  总耗时: {duration:.2f}s
{'═' * 60}
"""
            self._write(footer)
            self.log_file.close()
            self.log_file = None
    
    def __del__(self):
        """析构时关闭文件"""
        self.close()


# 全局日志实例（方便在各模块中使用）
_global_logger = None

def init_global_logger(**kwargs):
    """初始化全局日志器"""
    global _global_logger
    _global_logger = DebugLogger(**kwargs)
    return _global_logger

def get_logger():
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = DebugLogger(debug=False)
    return _global_logger
