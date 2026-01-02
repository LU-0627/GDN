"""
GDN (Graph Deviation Network) 模型
用于多变量时间序列异常检测的图神经网络

主要组件:
1. OutLayer: 输出层MLP
2. GNNLayer: 图神经网络层（包含GraphLayer + BatchNorm + ReLU）
3. GDN: 主模型，包含节点嵌入、图结构学习、GNN层和输出层
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from util.debug_logger import get_logger
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


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
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    """
    输出层: 多层感知机(MLP)
    将GNN输出映射到最终预测值
    """
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    """
    GNN层: 图注意力层 + BatchNorm + ReLU
    """
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

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
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
  
        out = self.bn(out)
        
        return self.relu(out)


class GDN(nn.Module):
    """
    GDN (Graph Deviation Network) 主模型
    
    工作流程:
    1. 节点嵌入学习
    2. 基于嵌入的图结构学习（余弦相似度 + TopK）
    3. GNN消息传递
    4. 输出层预测
    """
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)
        
        # 用于控制forward日志的计数器
        self._forward_count = 0

        self.init_params()
    
    def init_params(self):
        """初始化参数: 使用Kaiming初始化节点嵌入"""
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
        logger = get_logger()
        
        # 只在第一次forward时打印详细日志
        should_log = logger.debug and logger.debug_forward and self._forward_count == 0
        self._forward_count += 1

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        
        if should_log:
            logger.log_subsection("GDN Forward 过程", icon='🧠')
            logger.log_forward_step(f"1. 输入数据", x, f"batch={batch_num}, nodes={node_num}, features={all_feature}")
        
        x = x.view(-1, all_feature).contiguous()
        
        if should_log:
            logger.log_forward_step("2. 展平输入", x, f"合并batch和node维度")


        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            # 获取节点嵌入
            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            
            if should_log:
                logger.log_forward_step("3. 节点嵌入", all_embeddings)

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            # 计算余弦相似度矩阵
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat
            
            if should_log:
                logger.log_forward_step("4. 余弦相似度矩阵", cos_ji_mat, 
                    f"min={cos_ji_mat.min().item():.3f}, max={cos_ji_mat.max().item():.3f}")

            dim = weights.shape[-1]
            topk_num = self.topk

            # TopK选择邻居
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            
            if should_log:
                logger.log_forward_step(f"5. TopK邻居选择", topk_indices_ji, f"每节点{topk_num}个邻居")

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            
            if should_log:
                logger.log_forward_step("6. 学习到的边", gated_edge_index, f"总边数={gated_edge_index.shape[1]}")

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, embedding=all_embeddings)
            
            if should_log:
                logger.log_forward_step(f"7. GNN层{i}输出", gcn_out)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        
        if should_log:
            logger.log_forward_step("8. 拼接GNN输出", x)
        
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        if should_log:
            logger.log_forward_step("9. 嵌入乘积", out)
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)
        
        if should_log:
            logger.log_forward_step("10. BatchNorm + ReLU", out)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
        
        if should_log:
            logger.log_forward_step("11. 最终输出", out)
   

        return out