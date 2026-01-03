"""
GDN (Graph Deviation Network) 模型
用于多变量时间序列异常检测的图神经网络

主要组件:
1. OutLayer: 输出层MLP
2. GNNLayer: 图神经网络层（包含GraphLayer + BatchNorm + ReLU）
3. GDN: 主模型，包含节点嵌入、图结构学习、GNN层和输出层
"""
import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch深度学习框架
import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import torch.nn as nn  # 导入PyTorch神经网络模块
import time  # 导入时间处理模块
import math  # 导入数学运算模块
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块
from util.time import *  # 从util.time模块导入所有函数
from util.env import *  # 从util.env模块导入所有函数
from util.debug_logger import get_logger  # 从util.debug_logger模块导入日志器
from torch_geometric.nn import GCNConv, GATConv, EdgeConv  # 从PyTorch Geometric导入图卷积层

from .graph_layer import GraphLayer  # 从当前包导入GraphLayer类


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """
    将边索引扩展到batch维度

    Args:
        org_edge_index: 原始边索引 [2, edge_num]
        batch_num: batch大小
        node_num: 节点数量

    Returns:
        batch_edge_index: 扩展后的边索引 [2, edge_num * batch_num]
    """
    edge_index = org_edge_index.clone().detach()  # 复制原始边索引并分离梯度
    edge_num = org_edge_index.shape[1]  # 获取边数量
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()  # 重复边索引以适应批次大小

    for i in range(batch_num):  # 遍历每个批次
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num  # 为每个批次的节点添加偏移

    return batch_edge_index.long()  # 返回长整型的批次边索引


class OutLayer(nn.Module):
    """
    输出层: 多层感知机(MLP)
    将GNN输出映射到最终预测值
    """
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        # 初始化输出层
        super(OutLayer, self).__init__()  # 调用父类初始化方法

        modules = []  # 初始化模块列表

        for i in range(layer_num):  # 遍历每层
            # last layer, output shape:1
            if i == layer_num - 1:  # 如果是最后一层
                # 添加线性层，输出维度为1
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:  # 如果不是最后一层
                layer_in_num = in_num if i == 0 else inter_num  # 确定输入维度
                modules.append(nn.Linear(layer_in_num, inter_num))  # 添加线性层
                modules.append(nn.BatchNorm1d(inter_num))  # 添加批归一化层
                modules.append(nn.ReLU())  # 添加ReLU激活函数

        self.mlp = nn.ModuleList(modules)  # 将模块列表转换为ModuleList

    def forward(self, x):
        # 定义前向传播方法
        out = x  # 初始化输出为输入

        for mod in self.mlp:  # 遍历MLP中的每个模块
            if isinstance(mod, nn.BatchNorm1d):  # 如果是批归一化层
                out = out.permute(0, 2, 1)  # 调整维度顺序
                out = mod(out)  # 应用批归一化
                out = out.permute(0, 2, 1)  # 恢复维度顺序
            else:  # 如果不是批归一化层
                out = mod(out)  # 应用模块

        return out  # 返回输出


class GNNLayer(nn.Module):
    """
    GNN层: 图注意力层 + BatchNorm + ReLU
    """
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        # 初始化GNN层
        super(GNNLayer, self).__init__()  # 调用父类初始化方法

        self.gnn = GraphLayer(
            in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False
        )  # 创建GraphLayer实例

        self.bn = nn.BatchNorm1d(out_channel)  # 创建批归一化层
        self.relu = nn.ReLU()  # 创建ReLU激活函数
        self.leaky_relu = nn.LeakyReLU()  # 创建LeakyReLU激活函数

    def forward(self, x, edge_index, embedding=None, node_num=0):
        """
        Args:
            x: 节点特征 [batch*node_num, feature_dim]
            edge_index: 边索引 [2, edge_num]
            embedding: 节点嵌入
            node_num: 节点数量

        Returns:
            out: 输出特征 [batch*node_num, out_channel]
        """
        # 执行图层前向传播
        out, (new_edge_index, att_weight) = self.gnn(
            x, edge_index, embedding, return_attention_weights=True
        )
        self.att_weight_1 = att_weight  # 保存注意力权重
        self.edge_index_1 = new_edge_index  # 保存边索引

        out = self.bn(out)  # 应用批归一化

        return self.relu(out)  # 应用ReLU激活函数并返回输出


class GDN(nn.Module):
    """
    GDN (Graph Deviation Network) 主模型

    工作流程:
    1. 节点嵌入学习
    2. 基于嵌入的图结构学习（余弦相似度 + TopK）
    3. GNN消息传递
    4. 输出层预测
    """
    def __init__(
            self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256,
            input_dim=10, out_layer_num=1, topk=20
    ):
        # 初始化GDN模型
        super(GDN, self).__init__()  # 调用父类初始化方法

        self.edge_index_sets = edge_index_sets  # 存储边索引集合

        device = get_device()  # 获取设备

        edge_index = edge_index_sets[0]  # 获取第一个边索引

        embed_dim = dim  # 设置嵌入维度
        self.embedding = nn.Embedding(node_num, embed_dim)  # 创建节点嵌入层
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)  # 创建输出层输入批归一化层

        edge_set_num = len(edge_index_sets)  # 获取边集合数量
        self.gnn_layers = nn.ModuleList([
            # 为每个边集合创建GNN层
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1)
            for i in range(edge_set_num)
        ])

        self.node_embedding = None  # 初始化节点嵌入为None
        self.topk = topk  # 存储TopK参数
        self.learned_graph = None  # 初始化学习到的图为None

        # 创建输出层
        self.out_layer = OutLayer(
            dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim
        )

        self.cache_edge_index_sets = [None] * edge_set_num  # 初始化边索引缓存列表
        self.cache_embed_index = None  # 初始化嵌入索引缓存为None

        self.dp = nn.Dropout(0.2)  # 创建Dropout层

        # 用于控制forward日志的计数器
        self._forward_count = 0  # 初始化前向传播计数器

        self.init_params()  # 初始化参数

    def init_params(self):
        """初始化参数: 使用Kaiming初始化节点嵌入"""
        # 使用Kaiming方法初始化嵌入权重
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        """
        前向传播

        Args:
            data: 输入数据 [batch_size, node_num, feature_dim]
            org_edge_index: 原始边索引

        Returns:
            out: 预测输出 [batch_size, node_num]
        """
        # 获取日志器
        logger = get_logger()  # 获取日志器实例

        # 只在第一次forward时打印详细日志
        should_log = logger.debug and logger.debug_forward and self._forward_count == 0  # 判断是否需要记录日志
        self._forward_count += 1  # 增加前向传播计数

        x = data.clone().detach()  # 复制输入数据并分离梯度
        edge_index_sets = self.edge_index_sets  # 获取边索引集合

        device = data.device  # 获取数据设备

        batch_num, node_num, all_feature = x.shape  # 获取批次大小、节点数量和特征维度

        if should_log:  # 如果需要记录日志
            logger.log_subsection("GDN Forward 过程", icon='🧠')  # 记录前向传播过程标题
            # 记录输入数据信息
            logger.log_forward_step(
                f"1. 输入数据", x, f"batch={batch_num}, nodes={node_num}, features={all_feature}"
            )

        x = x.view(-1, all_feature).contiguous()  # 将输入展平为二维张量

        if should_log:  # 如果需要记录日志
            logger.log_forward_step("2. 展平输入", x, f"合并batch和node维度")  # 记录展平操作

        gcn_outs = []  # 初始化GNN输出列表
        for i, edge_index in enumerate(edge_index_sets):  # 遍历边索引集合
            edge_num = edge_index.shape[1]  # 获取边数量
            cache_edge_index = self.cache_edge_index_sets[i]  # 获取缓存的边索引

            # 如果缓存不存在或形状不匹配
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                # 生成批次边索引并缓存
                self.cache_edge_index_sets[i] = get_batch_edge_index(
                    edge_index, batch_num, node_num
                ).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]  # 获取批次边索引

            # 获取节点嵌入
            all_embeddings = self.embedding(torch.arange(node_num).to(device))  # 获取所有节点嵌入

            if should_log:  # 如果需要记录日志
                logger.log_forward_step("3. 节点嵌入", all_embeddings)  # 记录节点嵌入信息

            weights_arr = all_embeddings.detach().clone()  # 分离并复制嵌入权重
            all_embeddings = all_embeddings.repeat(batch_num, 1)  # 为每个批次重复嵌入

            weights = weights_arr.view(node_num, -1)  # 重塑权重矩阵

            # 计算余弦相似度矩阵
            cos_ji_mat = torch.matmul(weights, weights.T)  # 计算权重的矩阵乘积
            # 计算权重的范数乘积
            normed_mat = torch.matmul(
                weights.norm(dim=-1).view(-1, 1),
                weights.norm(dim=-1).view(1, -1)
            )
            cos_ji_mat = cos_ji_mat / normed_mat  # 计算余弦相似度矩阵

            if should_log:  # 如果需要记录日志
                # 记录余弦相似度矩阵信息
                logger.log_forward_step(
                    "4. 余弦相似度矩阵", cos_ji_mat,
                    f"min={cos_ji_mat.min().item():.3f}, max={cos_ji_mat.max().item():.3f}"
                )

            dim = weights.shape[-1]  # 获取权重维度
            topk_num = self.topk  # 获取TopK数量

            # TopK选择邻居
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]  # 选择TopK最相似的节点

            if should_log:  # 如果需要记录日志
                # 记录TopK邻居选择结果
                logger.log_forward_step(
                    f"5. TopK邻居选择", topk_indices_ji, f"每节点{topk_num}个邻居"
                )

            self.learned_graph = topk_indices_ji  # 保存学习到的图结构

            # 创建目标节点索引
            gated_i = torch.arange(0, node_num).unsqueeze(1).repeat(1, topk_num).flatten().to(
                device
            ).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)  # 创建源节点索引
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)  # 构建门控边索引

            if should_log:  # 如果需要记录日志
                # 记录学习到的边信息
                logger.log_forward_step(
                    "6. 学习到的边", gated_edge_index, f"总边数={gated_edge_index.shape[1]}"
                )

            # 获取批次门控边索引
            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num
            ).to(device)
            # 执行GNN层前向传播
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, embedding=all_embeddings)

            if should_log:  # 如果需要记录日志
                logger.log_forward_step(f"7. GNN层{i}输出", gcn_out)  # 记录GNN层输出信息

            gcn_outs.append(gcn_out)  # 将GNN输出添加到列表

        x = torch.cat(gcn_outs, dim=1)  # 拼接所有GNN输出

        if should_log:  # 如果需要记录日志
            logger.log_forward_step("8. 拼接GNN输出", x)  # 记录拼接后的GNN输出

        x = x.view(batch_num, node_num, -1)  # 将输出重塑为三维张量

        indexes = torch.arange(0, node_num).to(device)  # 创建节点索引
        out = torch.mul(x, self.embedding(indexes))  # 将输出与节点嵌入相乘

        if should_log:  # 如果需要记录日志
            logger.log_forward_step("9. 嵌入乘积", out)  # 记录嵌入乘积结果

        out = out.permute(0, 2, 1)  # 调整维度顺序
        out = F.relu(self.bn_outlayer_in(out))  # 应用批归一化和ReLU激活函数
        out = out.permute(0, 2, 1)  # 恢复维度顺序

        if should_log:  # 如果需要记录日志
            logger.log_forward_step("10. BatchNorm + ReLU", out)  # 记录批归一化和ReLU结果

        out = self.dp(out)  # 应用Dropout
        out = self.out_layer(out)  # 通过输出层
        out = out.view(-1, node_num)  # 将输出重塑为二维张量

        if should_log:  # 如果需要记录日志
            logger.log_forward_step("11. 最终输出", out)  # 记录最终输出

        return out  # 返回最终输出