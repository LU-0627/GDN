import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from util.logger import setup_logger, get_logger




def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None, log_dir=None):

    logger = setup_logger(name=f'train_{dataset_name}', log_dir=log_dir or 'logs')

    logger.info('='*80)
    logger.info('开始训练')
    logger.info('='*80)
    logger.info(f'数据集: {dataset_name}')
    logger.info(f'模型保存路径: {save_path}')
    logger.info(f'训练配置: {config}')
    logger.info(f'训练数据集大小: {len(train_dataset) if train_dataset else "N/A"}')
    logger.info(f'训练批次数: {len(train_dataloader)}')
    logger.info(f'验证批次数: {len(val_dataloader) if val_dataloader else "N/A"}')

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    logger.info(f'使用设备: {device}')

    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    logger.info(f'总训练轮数: {epoch}')
    logger.info(f'早停窗口: {early_stop_win}')
    logger.info('-'*80)

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        logger.info(f'Epoch {i_epoch+1}/{epoch} 开始')

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        avg_train_loss = acu_loss/len(dataloader)
        
        logger.info(f'Epoch {i_epoch+1}/{epoch} - 训练Loss: {avg_train_loss:.8f}, 累计Loss: {acu_loss:.8f}')
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        avg_train_loss, acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            logger.info('开始验证...')
            val_loss, val_result = test(model, val_dataloader)
            logger.info(f'验证Loss: {val_loss:.8f}')

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                logger.info(f'模型已保存到: {save_path} (验证Loss改善: {min_loss:.8f} -> {val_loss:.8f})')
                
                # 备份模型到pretrained目录
                import os
                pretrained_dir = './pretrained'
                os.makedirs(pretrained_dir, exist_ok=True)
                backup_model_path = os.path.join(pretrained_dir, f'{dataset_name}_model.pt')
                torch.save(model.state_dict(), backup_model_path)
                logger.info(f'模型已备份到: {backup_model_path}')
                
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1
                logger.info(f'验证Loss未改善 (连续未改善次数: {stop_improve_count}/{early_stop_win})')


            if stop_improve_count >= early_stop_win:
                logger.info(f'触发早停: 连续 {early_stop_win} 轮验证Loss未改善')
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                logger.info(f'模型已保存到: {save_path} (训练Loss改善: {min_loss:.8f} -> {acu_loss:.8f})')
                
                # 备份模型到pretrained目录
                import os
                pretrained_dir = './pretrained'
                os.makedirs(pretrained_dir, exist_ok=True)
                backup_model_path = os.path.join(pretrained_dir, f'{dataset_name}_model.pt')
                torch.save(model.state_dict(), backup_model_path)
                logger.info(f'模型已备份到: {backup_model_path}')
                
                min_loss = acu_loss

        logger.info('-'*80)

    logger.info('='*80)
    logger.info('训练完成')
    logger.info(f'最佳Loss: {min_loss:.8f}')
    logger.info(f'训练总轮数: {i_epoch+1}')
    logger.info('='*80)



    return train_loss_list
