import torch  # 导入PyTorch深度学习框架
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU  # 从torch.nn导入参数、线性层、序列容器、批归一化和ReLU激活函数
import torch.nn.functional as F  # 导入PyTorch神经网络函数模块
from torch_geometric.nn.conv import MessagePassing  # 从PyTorch Geometric导入消息传递基类
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax  # 从PyTorch Geometric工具模块导入自环处理和softmax函数

from torch_geometric.nn.inits import glorot, zeros  # 从PyTorch Geometric初始化模块导入glorot和zeros初始化函数
import time  # 导入时间处理模块
import math  # 导入数学运算模块

class GraphLayer(MessagePassing):  # 定义图层类，继承自MessagePassing基类
    def __init__(self, in_channels, out_channels, heads=1, concat=True,  # 初始化方法，定义图层的参数
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):  # 初始化方法的参数设置
        super(GraphLayer, self).__init__(aggr='add', **kwargs)  # 调用父类MessagePassing的初始化方法，设置聚合方式为相加

        self.in_channels = in_channels  # 存储输入通道数
        self.out_channels = out_channels  # 存储输出通道数
        self.heads = heads  # 存储注意力头数
        self.concat = concat  # 存储是否拼接多头结果的标志
        self.negative_slope = negative_slope  # 存储LeakyReLU的负斜率参数
        self.dropout = dropout  # 存储dropout比率

        self.__alpha__ = None  # 初始化注意力权重参数

        self.lin = Linear(in_channels, heads * out_channels, bias=False)  # 定义线性变换层，将输入映射到多头输出维度

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))  # 定义目标节点注意力参数
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))  # 定义源节点注意力参数
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))  # 定义目标节点嵌入注意力参数
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))  # 定义源节点嵌入注意力参数

        if bias and concat:  # 如果需要偏置且拼接多头结果
            self.bias = Parameter(torch.Tensor(heads * out_channels))  # 设置偏置参数为拼接后的维度
        elif bias and not concat:  # 如果需要偏置但不拼接多头结果
            self.bias = Parameter(torch.Tensor(out_channels))  # 设置偏置参数为单头维度
        else:  # 其他情况
            self.register_parameter('bias', None)  # 不设置偏置参数

        self.reset_parameters()  # 重置参数

    def reset_parameters(self):  # 重置参数方法
        glorot(self.lin.weight)  # 使用Glorot方法初始化线性层权重
        glorot(self.att_i)  # 使用Glorot方法初始化目标节点注意力参数
        glorot(self.att_j)  # 使用Glorot方法初始化源节点注意力参数
        
        zeros(self.att_em_i)  # 使用零值初始化目标节点嵌入注意力参数
        zeros(self.att_em_j)  # 使用零值初始化源节点嵌入注意力参数

        zeros(self.bias)  # 使用零值初始化偏置参数


    '''   向前传播'''
    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        # 1.线性变换
        if torch.is_tensor(x):  # 如果输入x是张量
            x = self.lin(x)  # 对x应用线性变换
            x = (x, x)  # 将x转换为元组形式
        else:  # 如果输入x是元组
            x = (self.lin(x[0]), self.lin(x[1]))  # 对元组中的每个元素应用线性变换

        # 2. 图结构处理：移除并重新添加自环，确保节点能关注到自身
        edge_index, _ = remove_self_loops(edge_index)  # 移除边索引中的自环
        edge_index, _ = add_self_loops(edge_index,  # 添加自环到边索引
                                       num_nodes=x[1].size(self.node_dim))  # 根据节点数量添加自环

        # 3. 开启消息传递流程
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,  # 通过边索引传播消息
                             return_attention_weights=return_attention_weights)  # 传播时可选择返回注意力权重


        # 4. 后处理
        if not self.concat:  # 如果不拼接多头结果
            # 如果不拼接，对每个head取平均
            out = out.view(-1, self.heads, self.out_channels).mean(dim=1)  # 将输出重新整形并沿头维度取平均

        if self.bias is not None:  # 如果存在偏置参数
            out = out + self.bias  # 将偏置加到输出上

        if return_attention_weights:  # 如果需要返回注意力权重
            alpha, self.__alpha__ = self.__alpha__, None  # 获取并清除内部存储的注意力权重
            return out, (edge_index, alpha)  # 返回输出和注意力权重
        else:  # 否则
            return out  # 只返回输出

    # 消息传递方法，定义如何沿边传递消息
    def message(self, x_i, x_j, edge_index_i, size_i,  
                embedding,  # 节点嵌入
                edges,  # 边索引
                return_attention_weights):  # 是否返回注意力权重

        x_i = x_i.view(-1, self.heads, self.out_channels)  # 将目标节点特征重塑为[节点数, 头数, 输出通道数]
        x_j = x_j.view(-1, self.heads, self.out_channels)  # 将源节点特征重塑为[节点数, 头数, 输出通道数]

        if embedding is not None:  # 如果提供了嵌入向量
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]  # 获取目标和源节点的嵌入
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)  # 在第1维度扩展嵌入向量，并重复heads次
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)  # 在第1维度扩展嵌入向量，并重复heads次

            key_i = torch.cat((x_i, embedding_i), dim=-1)  # 拼接目标节点特征和嵌入
            key_j = torch.cat((x_j, embedding_j), dim=-1)  # 拼接源节点特征和嵌入


        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)  # 拼接目标节点注意力参数和嵌入注意力参数
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)  # 拼接源节点注意力参数和嵌入注意力参数

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)  # 计算注意力分数

        # 确保alpha的形状正确
        alpha = alpha.view(-1, self.heads)  # 将注意力分数重塑为[边数, 头数]

        alpha = F.leaky_relu(alpha, self.negative_slope)  # 应用LeakyReLU激活函数
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)  # 对注意力分数应用softmax归一化

        if return_attention_weights:  # 如果需要返回注意力权重
            self.__alpha__ = alpha  # 保存注意力权重

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # 对注意力权重应用dropout
        
        out = x_j * alpha.view(-1, self.heads, 1)  # 将源节点特征与注意力权重相乘
        # 将3D张量展平为2D，以兼容新版PyTorch Geometric的scatter聚合
        # 原始形状: [E, heads, out_channels] -> 新形状: [E, heads * out_channels]
        out = out.view(-1, self.heads * self.out_channels)  # 将输出重塑为[边数, 头数*输出通道数]
        return out  # 返回计算后的消息


    def __repr__(self):  # 定义对象的字符串表示方法
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,  # 返回类名、输入维度、输出维度和头数的格式化字符串
                                             self.in_channels,  # 输入维度
                                             self.out_channels, self.heads)  # 输出维度和头数