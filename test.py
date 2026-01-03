# -*- coding: utf-8 -*-
"""
测试/验证模块

该模块用于在验证集或测试集上评估GDN模型的性能。
主要功能：
1. 对数据集进行无梯度前向传播
2. 计算预测结果与真实值之间的MSE损失
3. 收集所有样本的预测值、真实值和标签

使用场景：
- 训练过程中的验证（val_dataloader）
- 训练完成后的测试（test_dataloader）
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.debug_logger import get_logger

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *



def test(model, dataloader):
    """
    测试/验证GDN模型
    
    在给定数据集上评估模型性能，不计算梯度，不更新参数。
    收集所有样本的预测结果用于后续的异常检测评估。
    
    Args:
        model: 训练好的GDN模型实例
        dataloader: PyTorch数据加载器，包含测试/验证数据
                   每个batch返回: (x, y, labels, edge_index)
                   - x: 输入时间序列 [batch, node_num, time_steps]
                   - y: 目标值 [batch, node_num]
                   - labels: 异常标签 [batch]  (0=正常, 1=异常)
                   - edge_index: 图边索引 [2, edge_num]
    
    Returns:
        avg_loss: float, 整个数据集的平均MSE损失
        result: list, 包含三个列表
            [0] test_predicted_list: 所有样本的预测值
            [1] test_ground_list: 所有样本的真实值
            [2] test_labels_list: 所有样本的异常标签
    """
    # 获取全局日志器实例
    logger = get_logger()
    
    # 定义损失函数：均方误差（MSE）
    loss_func = nn.MSELoss(reduction='mean')
    
    # 获取计算设备（GPU或CPU）
    device = get_device()

    # 用于记录每个batch的损失值
    test_loss_list = []
    
    # 记录开始时间，用于估算剩余时间
    now = time.time()

    # 最终结果列表（Python list格式，用于返回）
    test_predicted_list = []  # 预测值
    test_ground_list = []     # 真实值
    test_labels_list = []     # 异常标签

    # 临时tensor列表（用于GPU上的高效拼接）
    t_test_predicted_list = []  # 预测值tensor
    t_test_ground_list = []     # 真实值tensor
    t_test_labels_list = []     # 标签tensor

    # 数据集总批次数
    test_len = len(dataloader)
    
    # 记录测试开始日志
    logger.log_test_start()
    logger.log("总批次数", test_len)

    # 设置模型为评估模式（关闭Dropout、BatchNorm使用全局统计）
    model.eval()

    i = 0  # 当前batch索引
    acu_loss = 0  # 累计损失（用于计算平均值）
    
    # 遍历数据集的所有batch
    for x, y, labels, edge_index in dataloader:
        # 将数据移动到GPU/CPU并转换为float类型
        # x: [batch, node_num, time_steps] - 历史时间窗口
        # y: [batch, node_num] - 当前时刻的真实值
        # labels: [batch] - 该样本是否异常
        # edge_index: [2, edge_num] - 图的边连接关系
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        # 不计算梯度，节省内存和计算
        with torch.no_grad():
            # 模型前向传播，得到预测值
            # predicted: [batch, node_num] - 对当前时刻所有传感器的预测
            predicted = model(x, edge_index).float().to(device)
            
            # 计算预测值与真实值的MSE损失
            loss = loss_func(predicted, y)
            
            # 将标签扩展到与预测值相同的形状
            # 原始: [batch] -> 扩展后: [batch, node_num]
            # 这样每个传感器都对应同一个样本级别的标签
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            # 拼接所有batch的结果到tensor列表
            # 第一个batch直接赋值
            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                # 后续batch沿着batch维度拼接
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        # 记录当前batch的损失
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        # 每10个batch打印一次调试信息
        if i % 10 == 0:
            logger.log_batch(i, test_len, {
                '输入x': x,
                '预测': predicted,
                '真实y': y
            })
        
        i += 1

        # 每10000个batch打印一次进度（用于大数据集）
        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    # 将tensor转换为Python list（CPU内存）
    # 用于后续的评估和结果分析
    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    # 计算整个数据集的平均损失
    avg_loss = sum(test_loss_list)/len(test_loss_list)
    
    # 计算预测值和真实值的统计信息（用于调试）
    pred_np = t_test_predicted_list.cpu().numpy()
    gt_np = t_test_ground_list.cpu().numpy()
    
    # (最小值, 最大值, 平均值)
    pred_stats = (pred_np.min(), pred_np.max(), pred_np.mean())
    gt_stats = (gt_np.min(), gt_np.max(), gt_np.mean())
    
    # 总样本数（batch数 × batch_size）
    total_samples = len(test_predicted_list)
    
    # 记录测试完成日志
    logger.log_test_complete(avg_loss, total_samples, pred_stats, gt_stats)

    # 返回平均损失和所有样本的结果
    # 结果格式: [预测值列表, 真实值列表, 标签列表]
    # 每个列表的shape: [total_samples, node_num]
    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




