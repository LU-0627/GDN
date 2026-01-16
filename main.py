# -*- coding: utf-8 -*-
import pandas as pd  # 导入pandas库，用于数据处理和分析
import numpy as np   # 导入numpy库，用于数值计算
import torch         # 导入PyTorch深度学习框架
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
from torch.utils.data import DataLoader, random_split, Subset  # 从PyTorch导入数据加载器相关功能

from sklearn.preprocessing import MinMaxScaler  # 从sklearn导入数据缩放器

from util.env import get_device, set_device  # 从util.env模块导入设备获取和设置函数
from util.preprocess import build_loc_net, construct_data  # 从util.preprocess模块导入网络构建和数据构造函数
from util.net_struct import get_feature_map, get_fc_graph_struc  # 从util.net_struct模块导入特征映射和全连接图结构获取函数
from util.iostream import printsep  # 从util.iostream模块导入打印分隔符函数
from util.debug_logger import DebugLogger, init_global_logger, get_logger  # 从util.debug_logger模块导入调试日志相关类和函数

from datasets.TimeDataset import TimeDataset  # 从datasets.TimeDataset模块导入时间序列数据集类

from models.GDN import GDN  # 从models.GDN模块导入GDN图神经网络模型

from train import train  # 从train模块导入训练函数
from test import test   # 从test模块导入测试函数
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores  # 从evaluate模块导入评估相关函数

import sys             # 导入系统相关功能模块
from datetime import datetime  # 从datetime模块导入日期时间类

import os              # 导入操作系统接口模块
import argparse        # 导入命令行参数解析模块
from pathlib import Path  # 从pathlib模块导入路径处理类

import json            # 导入JSON处理模块
import random          # 导入随机数生成模块


class Main():  # 定义主类，封装整个程序的主要功能
    def __init__(self, train_config, env_config, debug=False, debug_config=None):  # 初始化方法，接收训练配置、环境配置和调试参数
        """
        初始化主程序
        
        Args:
            train_config: 训练配置
            env_config: 环境配置  
            debug: 是否开启调试模式
            debug_config: 调试配置 {'debug_batch': N, 'debug_forward': bool}
        """
        self.train_config = train_config  # 存储训练配置
        self.env_config = env_config      # 存储环境配置
        self.datestr = None               # 初始化日期字符串变量
        
        # 初始化日志器
        if debug_config is None:          # 如果调试配置为空
            debug_config = {}             # 创建空字典
        self.logger = init_global_logger(  # 初始化全局日志器
            debug=debug,                  # 调试模式开关
            log_dir='./logs',             # 日志保存目录
            dataset_name=env_config['dataset'],  # 数据集名称
            debug_batch=debug_config.get('debug_batch', 1),  # 调试批次间隔
            debug_forward=debug_config.get('debug_forward', False)  # 是否调试前向传播
        )
        
        # ========== 数据加载阶段 ==========
        self.logger.log_section("数据加载", icon='📊')  # 记录数据加载阶段开始
        
        dataset = self.env_config['dataset']  # 获取数据集名称
        self.logger.log("数据集", dataset)      # 记录数据集名称
        
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)  # 读取训练数据
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)    # 读取测试数据
        
        self.logger.log("原始训练数据", f"{train_orig.shape[0]} 行 × {train_orig.shape[1]} 列")  # 记录训练数据形状
        self.logger.log("原始测试数据", f"{test_orig.shape[0]} 行 × {test_orig.shape[1]} 列")    # 记录测试数据形状
       
        train, test = train_orig, test_orig  # 将原始数据赋值给train和test变量

        if 'attack' in train.columns:        # 如果训练数据中有attack列
            train = train.drop(columns=['attack'])  # 删除attack列
            self.logger.log("移除训练集attack列", "是")  # 记录删除操作

        feature_map = get_feature_map(dataset)      # 获取特征映射
        fc_struc = get_fc_graph_struc(dataset)      # 获取全连接图结构
        
        self.logger.log("特征数量", f"{len(feature_map)} 个传感器")  # 记录特征数量
        self.logger.log("特征列表", str(feature_map[:5]) + "..." if len(feature_map) > 5 else str(feature_map))  # 记录特征列表

        set_device(env_config['device'])  # 设置计算设备
        self.device = get_device()      # 获取当前设备
        self.logger.log("计算设备", str(self.device))  # 记录计算设备

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)  # 构建局部网络连接
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)  # 将边索引转换为PyTorch张量
        
        self.logger.log_tensor("图结构边索引 (fc_edge_index)", fc_edge_index, show_stats=False)  # 记录图结构边索引
        self.logger.log("边数量", fc_edge_index.shape[1])  # 记录边数量

        self.feature_map = feature_map  # 保存特征映射

        train_dataset_indata = construct_data(train, feature_map, labels=0)  # 构造训练数据
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())  # 构造测试数据
        
        # 计算异常样本比例
        attack_labels = test.attack.tolist()  # 获取攻击标签列表
        n_anomaly = sum(attack_labels)        # 计算异常样本数量
        n_total = len(attack_labels)          # 计算总样本数量
        self.logger.log("测试集异常样本", f"{n_anomaly}/{n_total} ({100*n_anomaly/n_total:.1f}%)")  # 记录异常样本比例

        # ========== 滑窗处理 ==========
        self.logger.log_subsection("滑窗处理", icon='🔄')  # 记录滑窗处理开始
        
        cfg = {  # 定义滑窗配置
            'slide_win': train_config['slide_win'],    # 滑动窗口大小
            'slide_stride': train_config['slide_stride'],  # 滑动步长
        }
        self.logger.log("滑动窗口", f"{cfg['slide_win']} 时间步")  # 记录滑动窗口大小
        self.logger.log("滑动步长", cfg['slide_stride'])  # 记录滑动步长

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)  # 创建训练数据集
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)    # 创建测试数据集
        
        self.logger.log("训练集样本数", len(train_dataset))  # 记录训练集样本数量
        self.logger.log("测试集样本数", len(test_dataset))    # 记录测试集样本数量
        
        # 打印单个样本的shape
        sample_x, sample_y, sample_label, sample_edge = train_dataset[0]  # 获取训练集第一个样本
        self.logger.log_tensor("单样本x (历史窗口)", sample_x, show_stats=False)  # 记录输入张量信息
        self.logger.log_tensor("单样本y (预测目标)", sample_y, show_stats=False)  # 记录输出张量信息

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset, 
            train_config['seed'], 
            train_config['batch'], 
            val_ratio=train_config['val_ratio']
        )  # 获取训练和验证数据加载器

        self.train_dataset = train_dataset  # 保存训练数据集
        self.test_dataset = test_dataset    # 保存测试数据集

        self.train_dataloader = train_dataloader  # 保存训练数据加载器
        self.val_dataloader = val_dataloader      # 保存验证数据加载器
        self.test_dataloader = DataLoader(
            test_dataset, 
            batch_size=train_config['batch'],
            shuffle=False, 
            num_workers=0
        )  # 创建测试数据加载器
        
        self.logger.log_subsection("数据加载器", icon='📦')  # 记录数据加载器信息
        self.logger.log("批次大小", train_config['batch'])  # 记录批次大小
        self.logger.log("训练批次数", len(train_dataloader))  # 记录训练批次数
        self.logger.log("验证批次数", len(val_dataloader))    # 记录验证批次数
        self.logger.log("测试批次数", len(self.test_dataloader))  # 记录测试批次数

        # ========== 模型初始化 ==========
        self.logger.log_section("模型初始化", icon='🏗️')  # 记录模型初始化开始
        
        edge_index_sets = []      # 初始化边索引集合
        edge_index_sets.append(fc_edge_index)  # 添加全连接边索引

        self.model = GDN(
            edge_index_sets, 
            len(feature_map),
            dim=train_config['dim'], 
            input_dim=train_config['slide_win'],
            out_layer_num=train_config['out_layer_num'],
            out_layer_inter_dim=train_config['out_layer_inter_dim'],
            topk=train_config['topk']
        ).to(self.device)  # 创建GDN模型实例并移动到指定设备
        
        self.logger.log("节点数量", len(feature_map))  # 记录节点数量
        self.logger.log("嵌入维度", train_config['dim'])  # 记录嵌入维度
        self.logger.log("输入窗口长度", train_config['slide_win'])  # 记录输入窗口长度
        self.logger.log("Top-K邻居数", train_config['topk'])  # 记录Top-K邻居数
        self.logger.log("输出层数", train_config['out_layer_num'])  # 记录输出层数
        self.logger.log("输出层中间维度", train_config['out_layer_inter_dim'])  # 记录输出层中间维度
        self.logger.log_model_summary(self.model)  # 记录模型摘要信息

    def run(self):  # 定义运行方法
        if len(self.env_config['load_model_path']) > 0:  # 如果有预训练模型路径
            model_save_path = self.env_config['load_model_path']  # 使用预训练模型路径
        else:  # 否则
            model_save_path = self.get_save_path()[0]  # 获取新的保存路径

            self.train_log = train(
                self.model, 
                model_save_path,
                config=train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            )  # 开始训练模型
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))  # 加载训练好的模型权重
        best_model = self.model.to(self.device)  # 将模型移动到指定设备

        _, self.test_result = test(best_model, self.test_dataloader)  # 对测试集进行测试
        _, self.val_result = test(best_model, self.val_dataloader)    # 对验证集进行测试

        self.get_score(self.test_result, self.val_result)  # 计算评估分数

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):  # 定义获取数据加载器的方法
        dataset_len = int(len(train_dataset))  # 获取数据集长度
        train_use_len = int(dataset_len * (1 - val_ratio))  # 计算训练集使用长度
        val_use_len = int(dataset_len * val_ratio)          # 计算验证集使用长度
        val_start_index = random.randrange(train_use_len)   # 随机选择验证集起始索引
        indices = torch.arange(dataset_len)                 # 创建索引张量

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])  # 构建训练子索引
        train_subset = Subset(train_dataset, train_sub_indices)  # 创建训练子集

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]  # 构建验证子索引
        val_subset = Subset(train_dataset, val_sub_indices)  # 创建验证子集

        train_dataloader = DataLoader(
            train_subset, 
            batch_size=batch,
            shuffle=True
        )  # 创建训练数据加载器

        val_dataloader = DataLoader(
            val_subset, 
            batch_size=batch,
            shuffle=False
        )  # 创建验证数据加载器

        return train_dataloader, val_dataloader  # 返回训练和验证数据加载器

    def get_score(self, test_result, val_result):  # 定义获取评分的方法
        """
        计算并打印评估分数
        
        Args:
            test_result: 测试结果
            val_result: 验证结果
        """
        feature_num = len(test_result[0][0])  # 获取特征数量
        np_test_result = np.array(test_result)  # 将测试结果转换为numpy数组
        np_val_result = np.array(val_result)    # 将验证结果转换为numpy数组

        test_labels = np_test_result[2, :, 0].tolist()  # 获取测试标签
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)  # 获取完整错误评分

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)   # 获取最佳性能数据
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)  # 获取验证性能数据

        # 使用日志器打印最终结果
        self.logger.log_section("最终评估结果", icon='🏆')  # 记录评估结果部分开始
        
        # 打印两种评估方式的对比
        self.logger.log_subsection("最优阈值结果 (Best)", icon='⭐')  # 记录最优阈值结果
        self.logger.log_evaluation_result({
            'F1 Score': top1_best_info[0],
            'Precision': top1_best_info[1],
            'Recall': top1_best_info[2],
            'ROC-AUC': top1_best_info[3],
            'Threshold': top1_best_info[4],
        })
        
        self.logger.log_subsection("验证阈值结果 (Val)", icon='📋')  # 记录验证阈值结果
        self.logger.log_evaluation_result({
            'F1 Score': top1_val_info[0],
            'Precision': top1_val_info[1],
            'Recall': top1_val_info[2],
            'ROC-AUC': top1_val_info[3],
            'Threshold': top1_val_info[4],
        })

        print('=========================** Result **============================\n')  # 打印结果分隔符

        info = None
        if self.env_config['report'] == 'best':  # 如果报告模式为best
            info = top1_best_info  # 使用最佳性能信息
        elif self.env_config['report'] == 'val':  # 如果报告模式为val
            info = top1_val_info   # 使用验证性能信息

        print(f'F1 score: {info[0]}')      # 打印F1分数
        print(f'precision: {info[1]}')     # 打印精确率
        print(f'recall: {info[2]}\n')      # 打印召回率

    def get_save_path(self, feature_name=''):  # 定义获取保存路径的方法
        dir_path = self.env_config['save_path']  # 获取保存路径
        
        if self.datestr is None:      # 如果日期字符串为空
            now = datetime.now()      # 获取当前时间
            self.datestr = now.strftime('%m|%d-%H-%M-%S')  # 格式化时间为字符串(Windows兼容)
        datestr = self.datestr          

        paths = [  # 定义路径列表
            f'./pretrained/{dir_path}/best_{datestr}.pt',  # 模型保存路径
            f'./results/{dir_path}/{datestr}.csv',         # 结果保存路径
        ]

        for path in paths:              # 遍历路径列表
            dirname = os.path.dirname(path)  # 获取目录名
            Path(dirname).mkdir(parents=True, exist_ok=True)  # 创建目录

        return paths  # 返回路径列表


if __name__ == "__main__":  # 当脚本作为主程序运行时
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器

    parser.add_argument('-batch', help='batch size', type=int, default=128)  # 批次大小参数
    parser.add_argument('-epoch', help='train epoch', type=int, default=100)  # 训练轮次参数
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)  # 滑动窗口参数
    parser.add_argument('-dim', help='dimension', type=int, default=64)       # 维度参数
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)  # 滑动步长参数
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')  # 保存路径模式参数
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='wadi')  # 数据集参数
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')   # 设备参数
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)  # 随机种子参数
    parser.add_argument('-comment', help='experiment comment', type=str, default='')  # 实验注释参数
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)  # 输出层数参数
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)  # 输出层中间维度参数
    parser.add_argument('-decay', help='decay', type=float, default=0)  # 衰减参数
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)  # 验证比例参数
    parser.add_argument('-topk', help='topk num', type=int, default=20)  # topk参数
    parser.add_argument('-report', help='best / val', type=str, default='best')  # 报告模式参数
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')  # 加载模型路径参数
    
    # 调试参数
    parser.add_argument('--debug', help='开启调试日志', action='store_true')  # 调试模式参数
    parser.add_argument('--debug_batch', help='每N个batch打印一次', type=int, default=1)  # 调试批次参数
    parser.add_argument('--debug_forward', help='打印forward内部细节', action='store_true')  # 调试前向传播参数

    args = parser.parse_args()  # 解析命令行参数

    random.seed(args.random_seed)        # 设置随机种子
    np.random.seed(args.random_seed)     # 设置numpy随机种子
    torch.manual_seed(args.random_seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed(args.random_seed)  # 设置CUDA随机种子
    torch.cuda.manual_seed_all(args.random_seed)  # 设置所有CUDA随机种子
    torch.backends.cudnn.benchmark = False      # 禁用CUDNN基准测试
    torch.backends.cudnn.deterministic = True   # 启用CUDNN确定性模式
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)  # 设置Python哈希种子

    train_config = {  # 定义训练配置字典
        'batch': args.batch,                    # 批次大小
        'epoch': args.epoch,                    # 训练轮次
        'slide_win': args.slide_win,            # 滑动窗口大小
        'dim': args.dim,                        # 维度
        'slide_stride': args.slide_stride,      # 滑动步长
        'comment': args.comment,                # 注释
        'seed': args.random_seed,               # 随机种子
        'out_layer_num': args.out_layer_num,    # 输出层数
        'out_layer_inter_dim': args.out_layer_inter_dim,  # 输出层中间维度
        'decay': args.decay,                    # 衰减
        'val_ratio': args.val_ratio,            # 验证比例
        'topk': args.topk,                      # topk参数
    }

    env_config = {  # 定义环境配置字典
        'save_path': args.save_path_pattern,    # 保存路径
        'dataset': args.dataset,                # 数据集名称
        'report': args.report,                  # 报告模式
        'device': args.device,                  # 计算设备
        'load_model_path': args.load_model_path  # 预训练模型路径
    }
    
    # 调试配置
    debug_config = {  # 定义调试配置字典
        'debug_batch': args.debug_batch,        # 调试批次间隔
        'debug_forward': args.debug_forward,    # 调试前向传播
    }

    main = Main(train_config, env_config, debug=args.debug, debug_config=debug_config)  # 创建Main实例
    main.run()  # 运行主程序

    # 关闭日志
    if args.debug:  # 如果开启调试模式
        from util.debug_logger import get_logger  # 导入日志器
        get_logger().close()  # 关闭日志器