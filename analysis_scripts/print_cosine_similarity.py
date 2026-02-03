"""
打印GDN模型的节点嵌入余弦相似度矩阵
用于分析传感器/节点之间的学习到的关系
"""

import torch
import numpy as np
import os
import argparse
from models.GDN import GDN
from util.env import get_device
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path, node_num=27, dim=64, input_dim=10, topk=20):
    """
    加载已训练的GDN模型
    
    Args:
        model_path: 模型权重文件路径
        node_num: 节点数量(传感器数量)
        dim: 嵌入维度
        input_dim: 输入特征维度
        topk: TopK参数
        
    Returns:
        model: 加载了权重的模型
    """
    device = get_device()
    
    # 创建一个简单的边索引(用于初始化模型)
    # 实际的边索引可能需要根据你的数据集调整
    edge_index = torch.zeros((2, node_num * topk), dtype=torch.long)
    edge_index_sets = [edge_index]
    
    # 初始化模型
    model = GDN(
        edge_index_sets=edge_index_sets,
        node_num=node_num,
        dim=dim,
        input_dim=input_dim,
        topk=topk
    ).to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ 成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return model


def compute_cosine_similarity(model):
    """
    计算节点嵌入的余弦相似度矩阵
    
    Args:
        model: GDN模型实例
        
    Returns:
        cos_similarity: 余弦相似度矩阵 [node_num, node_num]
    """
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        # 获取节点嵌入 (node_num, embed_dim)
        embeddings = model.embedding.weight
        node_num = embeddings.shape[0]
        embed_dim = embeddings.shape[1]
        
        print(f"\n📊 嵌入矩阵信息:")
        print(f"  - 节点数量: {node_num}")
        print(f"  - 嵌入维度: {embed_dim}")
        
        # 计算余弦相似度矩阵
        # 方法1: 使用矩阵乘法
        weights = embeddings.view(node_num, -1)
        cos_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_similarity = cos_mat / normed_mat  # [node_num, node_num]
        
        # 方法2(可选): 使用PyTorch的cosine_similarity
        # from torch.nn.functional import cosine_similarity
        # cos_similarity_alt = torch.zeros(node_num, node_num)
        # for i in range(node_num):
        #     for j in range(node_num):
        #         cos_similarity_alt[i, j] = cosine_similarity(
        #             embeddings[i].unsqueeze(0), 
        #             embeddings[j].unsqueeze(0)
        #         )
        
        return cos_similarity


def print_similarity_matrix(cos_similarity, top_n=5):
    """
    打印余弦相似度矩阵的详细信息
    
    Args:
        cos_similarity: 余弦相似度矩阵
        top_n: 打印前N个节点的子矩阵
    """
    print("\n" + "="*80)
    print("余弦相似度矩阵统计信息")
    print("="*80)
    
    print(f"\n📐 矩阵形状: {cos_similarity.shape}")
    print(f"📊 数值范围:")
    print(f"  - 最小值: {cos_similarity.min().item():.6f}")
    print(f"  - 最大值: {cos_similarity.max().item():.6f}")
    print(f"  - 平均值: {cos_similarity.mean().item():.6f}")
    print(f"  - 标准差: {cos_similarity.std().item():.6f}")
    
    # 打印对角线(应该都是1.0)
    diag = cos_similarity.diag()
    print(f"\n🔍 对角线值 (节点自己与自己):")
    print(f"  - 最小值: {diag.min().item():.6f}")
    print(f"  - 最大值: {diag.max().item():.6f}")
    print(f"  - 平均值: {diag.mean().item():.6f}")
    
    # 打印非对角线元素的统计
    mask = ~torch.eye(cos_similarity.shape[0], dtype=bool)
    off_diag = cos_similarity[mask]
    print(f"\n🔍 非对角线值 (节点与其他节点):")
    print(f"  - 最小值: {off_diag.min().item():.6f}")
    print(f"  - 最大值: {off_diag.max().item():.6f}")
    print(f"  - 平均值: {off_diag.mean().item():.6f}")
    print(f"  - 标准差: {off_diag.std().item():.6f}")
    
    # 打印前N个传感器的相似度子矩阵
    print(f"\n📋 前{top_n}个传感器的相似度矩阵:")
    print("-" * 80)
    submatrix = cos_similarity[:top_n, :top_n].cpu().numpy()
    
    # 打印表头
    header = "      " + "  ".join([f"Node{i:2d}" for i in range(top_n)])
    print(header)
    
    # 打印每一行
    for i in range(top_n):
        row_str = f"Node{i:2d} " + "  ".join([f"{submatrix[i, j]:7.4f}" for j in range(top_n)])
        print(row_str)
    
    print("\n" + "="*80)


def find_most_similar_pairs(cos_similarity, top_k=10):
    """
    找出相似度最高的节点对(排除对角线)
    
    Args:
        cos_similarity: 余弦相似度矩阵
        top_k: 返回前K个最相似的节点对
    """
    node_num = cos_similarity.shape[0]
    
    # 创建一个掩码,排除对角线
    mask = ~torch.eye(node_num, dtype=bool, device=cos_similarity.device)
    
    # 只取上三角(避免重复)
    triu_mask = torch.triu(torch.ones(node_num, node_num, dtype=bool), diagonal=1)
    combined_mask = mask & triu_mask
    
    # 获取上三角部分的相似度值和索引
    similarities = cos_similarity[combined_mask]
    
    # 找到top_k个最大值的索引
    topk_values, topk_indices = torch.topk(similarities, min(top_k, len(similarities)))
    
    # 将一维索引转换为二维坐标
    coords = torch.nonzero(combined_mask, as_tuple=False)
    
    print(f"\n🏆 相似度最高的{top_k}个节点对:")
    print("-" * 80)
    for idx, (value, pos) in enumerate(zip(topk_values, topk_indices)):
        i, j = coords[pos]
        print(f"{idx+1:2d}. Node {i.item():2d} ↔ Node {j.item():2d}  |  相似度: {value.item():.6f}")


def visualize_similarity_matrix(cos_similarity, save_path='cosine_similarity.png'):
    """
    可视化余弦相似度矩阵
    
    Args:
        cos_similarity: 余弦相似度矩阵
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 10))
    
    # 转换为numpy数组
    similarity_np = cos_similarity.cpu().numpy()
    
    # 使用seaborn绘制热力图
    sns.heatmap(
        similarity_np, 
        cmap='RdYlBu_r',  # 红-黄-蓝配色,红色表示高相似度
        center=0,  # 中心值为0
        square=True,  # 方形网格
        linewidths=0.5,  # 网格线宽度
        cbar_kws={"shrink": 0.8, "label": "余弦相似度"},
        vmin=-1,  # 最小值
        vmax=1,   # 最大值
        annot=similarity_np.shape[0] <= 20,  # 如果节点数<=20,显示数值
        fmt='.2f'  # 数值格式
    )
    
    plt.title('节点嵌入余弦相似度矩阵', fontsize=16, pad=20)
    plt.xlabel('节点索引', fontsize=12)
    plt.ylabel('节点索引', fontsize=12)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 相似度热力图已保存: {save_path}")
    
    # 可选: 显示图片
    # plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='打印GDN模型的余弦相似度矩阵')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='模型权重文件路径,例如: checkpoints/swat_best.pt')
    parser.add_argument('--node_num', type=int, default=27, 
                        help='节点数量(默认: 27, 对应SWaT数据集的传感器数量)')
    parser.add_argument('--dim', type=int, default=64, 
                        help='嵌入维度(默认: 64)')
    parser.add_argument('--input_dim', type=int, default=10, 
                        help='输入特征维度(默认: 10)')
    parser.add_argument('--topk', type=int, default=20, 
                        help='TopK参数(默认: 20)')
    parser.add_argument('--top_n', type=int, default=5, 
                        help='打印前N个节点的子矩阵(默认: 5)')
    parser.add_argument('--save_fig', action='store_true', 
                        help='是否保存相似度热力图')
    parser.add_argument('--fig_path', type=str, default='cosine_similarity.png', 
                        help='热力图保存路径(默认: cosine_similarity.png)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GDN模型 - 节点嵌入余弦相似度分析")
    print("="*80)
    
    # 加载模型
    print(f"\n📂 加载模型: {args.model_path}")
    model = load_model(
        model_path=args.model_path,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk
    )
    
    # 计算余弦相似度
    print("\n🔄 计算余弦相似度矩阵...")
    cos_similarity = compute_cosine_similarity(model)
    
    # 打印统计信息
    print_similarity_matrix(cos_similarity, top_n=args.top_n)
    
    # 找出最相似的节点对
    find_most_similar_pairs(cos_similarity, top_k=10)
    
    # 可视化(可选)
    if args.save_fig:
        visualize_similarity_matrix(cos_similarity, save_path=args.fig_path)
    
    print("\n✓ 分析完成!\n")
    
    return cos_similarity


if __name__ == '__main__':
    main()
