import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.debug_logger import get_logger
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr




def loss_func(y_pred, y_true):
    """
    计算MSE损失
    
    Args:
        y_pred: 预测值 [batch_size, node_num]
        y_true: 真实值 [batch_size, node_num]
    
    Returns:
        loss: 均方误差损失
    """
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):
    """
    训练GDN模型
    
    Args:
        model: GDN模型
        save_path: 模型保存路径
        config: 训练配置
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        feature_map: 特征映射
        test_dataloader: 测试数据加载器
        test_dataset: 测试数据集
        dataset_name: 数据集名称
        train_dataset: 训练数据集
    
    Returns:
        train_loss_list: 训练损失列表
    """
    # 获取日志器
    logger = get_logger()
    
    logger.log_section("训练阶段", icon='🎯')

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    
    logger.log_subsection("优化器配置", icon='⚙️')
    logger.log("优化器", "Adam")
    logger.log("学习率", 0.001)
    logger.log("权重衰减", config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15
    
    logger.log("总Epoch数", epoch)
    logger.log("早停窗口", early_stop_win)

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    total_batches = len(dataloader)
    
    logger.log("每Epoch批次数", total_batches)

    for i_epoch in range(epoch):
        epoch_start_time = time.time()
        
        # 打印Epoch开始
        logger.log_epoch_start(i_epoch, epoch)

        acu_loss = 0
        model.train()
        
        batch_idx = 0
        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
            
            # 打印Batch信息
            logger.log_batch(batch_idx, total_batches, {
                '输入x': x,
                '标签y': labels,
                '边索引': edge_index
            })

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            # 打印损失信息 (仅在debug_forward模式或首个batch)
            if batch_idx == 0 or (logger.debug and logger.debug_forward):
                pred_range = (out.min().item(), out.max().item())
                gt_range = (labels.min().item(), labels.max().item())
                logger.log_loss(loss.item(), pred_range, gt_range)
            
            loss.backward()
            
            # 打印梯度信息
            if batch_idx == 0:
                logger.log_gradient(model)
            
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1
            batch_idx += 1


        # each epoch
        epoch_time = time.time() - epoch_start_time
        avg_loss = acu_loss/len(dataloader)
        
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        avg_loss, acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)
            
            is_best = val_loss < min_loss

            if is_best:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
            
            # 打印Epoch结束信息
            logger.log_epoch_end(i_epoch, avg_loss, val_loss, best=is_best, time_elapsed=epoch_time)
            
            if stop_improve_count > 0:
                logger.log(f"⚠️ 早停计数器", f"{stop_improve_count}/{early_stop_win}")


            if stop_improve_count >= early_stop_win:
                logger.log("⚠️ 触发早停", f"连续{early_stop_win}个epoch无改善")
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss
            
            logger.log_epoch_end(i_epoch, avg_loss, time_elapsed=epoch_time)

    # 打印训练完成信息
    logger.log_training_complete(i_epoch + 1, min_loss, save_path)

    return train_loss_list
