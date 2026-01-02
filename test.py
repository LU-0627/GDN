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
    测试/验证模型
    
    Args:
        model: GDN模型
        dataloader: 数据加载器
    
    Returns:
        avg_loss: 平均损失
        result: [predictions, ground_truth, labels]
    """
    # 获取日志器
    logger = get_logger()
    
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)
    
    logger.log_test_start()
    logger.log("总批次数", test_len)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            
            
            loss = loss_func(predicted, y)
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        # 每10个batch打印一次
        if i % 10 == 0:
            logger.log_batch(i, test_len, {
                '输入x': x,
                '预测': predicted,
                '真实y': y
            })
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)
    
    # 计算统计信息
    pred_np = t_test_predicted_list.cpu().numpy()
    gt_np = t_test_ground_list.cpu().numpy()
    
    pred_stats = (pred_np.min(), pred_np.max(), pred_np.mean())
    gt_stats = (gt_np.min(), gt_np.max(), gt_np.mean())
    
    total_samples = len(test_predicted_list)
    logger.log_test_complete(avg_loss, total_samples, pred_stats, gt_stats)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




