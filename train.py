# -------------------------- 导入依赖库 --------------------------
# 导入numpy库，用于数值计算（数组操作、矩阵运算等）
import numpy as np
# 导入PyTorch核心库，用于构建和训练深度学习模型
import torch
# 导入matplotlib的pyplot模块，用于绘制可视化图表（此处代码未实际使用，可能为预留）
import matplotlib.pyplot as plt
# 导入PyTorch的神经网络模块，用于构建网络层（如nn.Module、nn.Linear等）
import torch.nn as nn
# 导入time库，用于计算程序运行时间（计时功能）
import time
# 从自定义工具包util的time模块导入所有内容（推测是时间格式化、计时辅助函数等）
from util.time import *
# 从自定义工具包util的env模块导入所有内容（推测是环境配置、设备获取等函数）
from util.env import *
# 从自定义工具包util的debug_logger模块导入get_logger函数，用于获取日志记录器
from util.debug_logger import get_logger
# 从sklearn.metrics导入均方误差（mean_squared_error），用于评估模型性能（此处代码未直接使用，可能为预留）
from sklearn.metrics import mean_squared_error
# 导入test模块的所有内容（推测是模型验证/测试的核心函数）
from test import *
# 导入PyTorch的函数式接口，用于调用激活函数、损失函数等（如F.mse_loss）
import torch.nn.functional as F
# 重复导入numpy库（冗余导入，可删除，不影响程序运行但不符合编码规范）
import numpy as np
# 从自定义evaluate模块导入三个评估函数，用于计算模型性能指标（最优性能、验证性能、误差分数）
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
# 从sklearn.metrics导入多个分类评估指标（精确率、召回率、AUC、F1分数，此处代码未直接使用，为预留）
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
# 从torch.utils.data导入数据加载相关工具，用于构建数据迭代器、数据集拆分、子集提取
from torch.utils.data import DataLoader, random_split, Subset
# 从scipy.stats导入四分位距（iqr，Interquartile Range），用于统计分析（此处代码未使用，为预留）
from scipy.stats import iqr

# -------------------------- 定义损失函数 --------------------------
def loss_func(y_pred, y_true):
    """
    计算MSE（均方误差）损失
    函数说明：用于衡量模型预测值与真实值之间的差异，是回归任务的常用损失函数
    
    Args:
        y_pred: 模型预测值，张量形状 [batch_size, node_num]（批次大小，节点数量）
        y_true: 真实标签值，张量形状 [batch_size, node_num]（与预测值形状一致）
    
    Returns:
        loss: 标量张量，均方误差损失的平均值
    """
    # 调用F.mse_loss计算均方误差
    # reduction='mean'：表示对所有元素的误差平方取平均值，返回标量损失
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss  # 返回计算得到的MSE损失

# -------------------------- 定义核心训练函数 --------------------------
def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):
    """
    训练GDN（Graph Deep Network，图深度网络）模型的核心函数
    功能：完成模型的迭代训练、验证、早停、模型保存等完整训练流程
    
    Args:
        model: 待训练的GDN模型实例（继承自torch.nn.Module）
        save_path: 最优模型参数的保存路径（字符串）
        config: 训练配置字典，包含epoch、seed、decay等超参数
        train_dataloader: 训练数据集的数据加载器（DataLoader实例），提供批量训练数据
        val_dataloader: 验证数据集的数据加载器（可选），用于评估模型泛化能力并早停
        feature_map: 特征映射字典（推测是节点/特征的索引映射，辅助模型推理）
        test_dataloader: 测试数据集的数据加载器（可选，此处训练阶段未使用，为预留参数）
        test_dataset: 测试数据集实例（可选，训练阶段未使用，为预留参数）
        dataset_name: 数据集名称（默认'swat'，用于日志记录和个性化配置）
        train_dataset: 训练数据集实例（可选，训练阶段未使用，为预留参数）
    
    Returns:
        train_loss_list: 所有训练批次的损失值列表，用于后续损失可视化分析
    """
    # 获取日志记录器实例，用于打印/记录训练过程中的日志信息（如超参数、损失、梯度等）
    logger = get_logger()
    
    # 记录训练阶段开始的日志（带图标🎯，便于日志可视化区分）
    logger.log_section("训练阶段", icon='🎯')

    # 从配置字典中获取随机种子（用于固定随机数，保证实验可复现，此处未直接使用，为预留）
    seed = config['seed']

    # 定义Adam优化器，用于更新模型参数
    # model.parameters()：获取模型所有可训练参数
    # lr=0.001：学习率，控制参数更新的步长
    # weight_decay=config['decay']：权重衰减，用于L2正则化，防止模型过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    
    # 记录优化器配置的子日志（带图标⚙️）
    logger.log_subsection("优化器配置", icon='⚙️')
    logger.log("优化器", "Adam")  # 记录优化器名称
    logger.log("学习率", 0.001)   # 记录学习率数值
    logger.log("权重衰减", config['decay'])  # 记录权重衰减数值

    # 记录训练开始的时间戳，用于后续计算总训练时长（此处未最终使用，为中间计时预留）
    now = time.time()
    
    # 初始化训练损失列表，用于存储每个批次的训练损失值
    train_loss_list = []
    # 初始化对比损失列表（此处代码全程未使用，为冗余变量，可删除）
    cmp_loss_list = []

    # 获取计算设备（CPU/GPU），通过自定义函数get_device()自动判断（优先使用GPU，无GPU则用CPU）
    device = get_device()

    # 初始化累计损失（用于计算每个Epoch的平均损失）
    acu_loss = 0
    # 初始化最小损失（用于筛选最优模型，初始值设为极大值1e+8，确保首次训练损失可更新）
    min_loss = 1e+8
    # 初始化最小F1分数（此处未使用，为预留变量，用于分类任务的最优模型筛选）
    min_f1 = 0
    # 初始化最小精确率（此处未使用，为预留变量）
    min_pre = 0
    # 初始化最优精确率（此处未使用，为预留变量）
    best_prec = 0

    # 初始化全局迭代次数计数器（记录所有批次的总迭代次数）
    i = 0
    # 从配置字典中获取总Epoch数（训练的总轮次）
    epoch = config['epoch']
    # 定义早停窗口大小（连续15个Epoch验证损失无下降则停止训练，防止过拟合和无效训练）
    early_stop_win = 15
    
    # 记录总Epoch数和早停窗口大小的日志
    logger.log("总Epoch数", epoch)
    logger.log("早停窗口", early_stop_win)

    # 将模型设置为训练模式
    # 作用：启用模型中的训练特有层（如Dropout、BatchNorm的训练模式），确保训练正常进行
    model.train()

    # 定义日志打印间隔（此处未使用，为预留参数，用于控制每多少批次打印一次日志）
    log_interval = 1000
    # 初始化早停计数器（记录连续验证损失无改善的Epoch数）
    stop_improve_count = 0

    # 将训练数据加载器赋值给局部变量dataloader（简化代码书写）
    dataloader = train_dataloader
    # 获取训练数据加载器的总批次数（每个Epoch包含的批次数量）
    total_batches = len(dataloader)
    
    # 记录每Epoch的批次数日志
    logger.log("每Epoch批次数", total_batches)

    # 开始Epoch循环（逐轮训练模型）
    for i_epoch in range(epoch):
        # 记录当前Epoch的开始时间戳，用于计算当前Epoch的训练耗时
        epoch_start_time = time.time()
        
        # 打印当前Epoch开始的日志（包含Epoch索引和总Epoch数）
        logger.log_epoch_start(i_epoch, epoch)

        # 重置当前Epoch的累计损失（确保每个Epoch的累计损失独立计算）
        acu_loss = 0
        # 再次将模型设为训练模式（防止验证阶段切换为eval模式后，当前Epoch训练未切换回来）
        model.train()
        
        # 初始化当前Epoch的批处理索引（记录当前Epoch内的批次序号）
        batch_idx = 0
        # 开始批处理循环（遍历当前Epoch的所有训练批次数据）
        for x, labels, attack_labels, edge_index in dataloader:
            # 记录当前批次的开始时间戳（此处未使用，为预留计时）
            _start = time.time()

            # 将输入数据、标签、边索引转换为float类型，并移至指定计算设备（CPU/GPU）
            # 列表推导式批量处理数据，简化代码；to(device)：实现数据与模型在同一设备上，避免计算错误
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
            
            # 打印当前批次的信息日志（包含批次索引、总批次数，以及各输入张量的基本信息）
            logger.log_batch(batch_idx, total_batches, {
                '输入x': x,          # 模型输入特征张量
                '标签y': labels,     # 真实标签张量
                '边索引': edge_index # 图结构的边索引张量（GDN模型的核心输入之一）
            })

            # 清空优化器的梯度缓存
            # 必要性：PyTorch的梯度会累积，若不清空，当前批次的梯度会叠加之前批次的梯度，导致参数更新错误
            optimizer.zero_grad()
            # 模型前向传播：输入x和edge_index，获取预测输出out
            # float().to(device)：确保预测输出的类型和设备与标签一致，避免损失计算错误
            out = model(x, edge_index).float().to(device)
            # 计算当前批次的损失：调用自定义的loss_func（MSE损失）
            loss = loss_func(out, labels)
            
            # 条件判断：仅在首个批次 或 开启debug模式且开启forward调试时，打印损失详细信息
            if batch_idx == 0 or (logger.debug and logger.debug_forward):
                # 计算预测值的最小值和最大值（用于分析预测值的分布范围）
                pred_range = (out.min().item(), out.max().item())
                # 计算真实标签的最小值和最大值（用于对比预测值与真实值的分布差异）
                gt_range = (labels.min().item(), labels.max().item())
                # 打印损失值、预测值范围、真实值范围的日志
                logger.log_loss(loss.item(), pred_range, gt_range)
            
            # 损失反向传播：计算模型所有可训练参数的梯度
            # 作用：根据当前批次的损失，反向推导每个参数对损失的贡献度（梯度）
            loss.backward()
            
            # 条件判断：仅在首个批次时，打印模型的梯度信息（用于调试梯度消失/爆炸问题）
            if batch_idx == 0:
                logger.log_gradient(model)
            
            # 优化器步进：根据计算得到的梯度，更新模型的可训练参数
            # 作用：实现模型参数的迭代优化，降低训练损失
            optimizer.step()

            # 将当前批次的损失值（转换为Python标量）加入训练损失列表
            train_loss_list.append(loss.item())
            # 累计当前Epoch的损失值（用于后续计算Epoch平均损失）
            acu_loss += loss.item()
                
            # 全局迭代次数计数器加1
            i += 1
            # 当前Epoch的批处理索引加1
            batch_idx += 1

        # -------------------------- 每个Epoch结束后的处理 --------------------------
        # 计算当前Epoch的训练耗时（当前时间 - Epoch开始时间）
        epoch_time = time.time() - epoch_start_time
        # 计算当前Epoch的平均损失（累计损失 / 总批次数）
        avg_loss = acu_loss/len(dataloader)
        
        # 打印当前Epoch的训练信息，flush=True：强制刷新输出缓冲区，确保日志实时打印
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        avg_loss, acu_loss), flush=True
            )

        # 判断是否存在验证数据加载器（若存在，使用验证集评估模型性能）
        if val_dataloader is not None:

            # 调用test函数，评估模型在验证集上的性能，返回验证损失和验证结果
            val_loss, val_result = test(model, val_dataloader)
            
            # 判断当前验证损失是否为历史最优（用于筛选最优模型）
            is_best = val_loss < min_loss

            # 若为最优模型
            if is_best:
                # 保存模型的状态字典（仅保存参数，不保存模型结构，占用空间小，加载灵活）
                torch.save(model.state_dict(), save_path)
                # 更新历史最小验证损失
                min_loss = val_loss
                # 重置早停计数器（验证损失改善，计数器清零）
                stop_improve_count = 0
            else:
                # 若不是最优模型，早停计数器加1
                stop_improve_count += 1
            
            # 打印当前Epoch结束的日志（包含训练损失、验证损失、是否最优、Epoch耗时）
            logger.log_epoch_end(i_epoch, avg_loss, val_loss, best=is_best, time_elapsed=epoch_time)
            
            # 若早停计数器大于0，打印当前计数器状态（提示早停风险）
            if stop_improve_count > 0:
                logger.log(f"⚠️ 早停计数器", f"{stop_improve_count}/{early_stop_win}")

            # 判断是否触发早停条件（计数器达到早停窗口大小）
            if stop_improve_count >= early_stop_win:
                # 打印触发早停的日志（提示连续多轮无改善）
                logger.log("⚠️ 触发早停", f"连续{early_stop_win}个epoch无改善")
                # 跳出Epoch循环，终止训练
                break

        # 若不存在验证数据加载器（仅使用训练损失筛选最优模型）
        else:
            # 判断当前Epoch的累计训练损失是否为历史最优
            if acu_loss < min_loss :
                # 保存最优模型参数
                torch.save(model.state_dict(), save_path)
                # 更新历史最小训练损失
                min_loss = acu_loss
            
            # 打印当前Epoch结束的日志（仅包含训练损失和Epoch耗时，无验证损失）
            logger.log_epoch_end(i_epoch, avg_loss, time_elapsed=epoch_time)

    # 打印训练完成的日志（包含训练的总Epoch数、最优损失、模型保存路径）
    logger.log_training_complete(i_epoch + 1, min_loss, save_path)

    # 返回所有训练批次的损失列表，用于后续可视化分析
    return train_loss_list