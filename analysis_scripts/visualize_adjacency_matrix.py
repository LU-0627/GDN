# -*- coding: utf-8 -*-
"""
生成Top-K有向图的邻接矩阵并可视化
邻接矩阵A[i,j]表示:节点i选择节点j作为Top-K邻居时的相似度值
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.GDN import GDN
from util.env import get_device

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_topk_adjacency_matrix(model_path, node_num=27, dim=64, input_dim=15, topk=20):
    """
    创建Top-K有向图的邻接矩阵
    
    Args:
        model_path: 模型路径
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
        
    Returns:
        adjacency_matrix: 邻接矩阵 [node_num, node_num]
        topk_indices: Top-K索引
        topk_values: Top-K相似度值
    """
    device = get_device()
    
    print("="*80)
    print("生成Top-K有向图邻接矩阵")
    print("="*80)
    print(f"\n📂 加载模型: {model_path}")
    
    # 加载模型
    edge_index = torch.zeros((2, node_num * topk), dtype=torch.long)
    model = GDN(
        edge_index_sets=[edge_index],
        node_num=node_num,
        dim=dim,
        input_dim=input_dim,
        topk=topk
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ 模型加载成功")
    
    with torch.no_grad():
        # 计算余弦相似度矩阵
        embeddings = model.embedding.weight
        weights = embeddings.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat
        
        print(f"\n📊 完整余弦相似度矩阵: {cos_ji_mat.shape}")
        
        # Top-K选择
        topk_values, topk_indices = torch.topk(cos_ji_mat, k=topk, dim=-1)
        print(f"🎯 Top-K选择: K={topk}")
        print(f"   - Top-K索引矩阵: {topk_indices.shape}")
        print(f"   - Top-K相似度矩阵: {topk_values.shape}")
    
    # 转换为numpy
    topk_indices_np = topk_indices.cpu().numpy()
    topk_values_np = topk_values.cpu().numpy()
    
    # 构建邻接矩阵
    print(f"\n🔄 构建邻接矩阵...")
    adjacency_matrix = np.zeros((node_num, node_num))
    
    for i in range(node_num):
        neighbors = topk_indices_np[i]
        similarities = topk_values_np[i]
        
        for neighbor, sim in zip(neighbors, similarities):
            # A[i, j] = 节点i选择节点j的相似度
            adjacency_matrix[i, neighbor] = sim
    
    print(f"✓ 邻接矩阵构建完成: {adjacency_matrix.shape}")
    
    # 统计信息
    print(f"\n📈 邻接矩阵统计:")
    print(f"   - 非零元素数: {np.count_nonzero(adjacency_matrix)}")
    print(f"   - 稀疏度: {1 - np.count_nonzero(adjacency_matrix) / (node_num * node_num):.2%}")
    print(f"   - 最小值: {adjacency_matrix.min():.6f}")
    print(f"   - 最大值: {adjacency_matrix.max():.6f}")
    print(f"   - 平均值(非零): {adjacency_matrix[adjacency_matrix > 0].mean():.6f}")
    
    # 检查对称性
    is_symmetric = np.allclose(adjacency_matrix, adjacency_matrix.T)
    print(f"   - 是否对称: {'是' if is_symmetric else '否(有向图)'}")
    
    # 对角线(自己到自己)
    diagonal = np.diag(adjacency_matrix)
    print(f"   - 对角线值范围: [{diagonal.min():.4f}, {diagonal.max():.4f}]")
    
    return adjacency_matrix, topk_indices_np, topk_values_np


def visualize_adjacency_matrix(adjacency_matrix, save_path='topk_adjacency_matrix.png'):
    """
    可视化邻接矩阵
    
    Args:
        adjacency_matrix: 邻接矩阵
        save_path: 保存路径
    """
    print(f"\n🎨 生成邻接矩阵热力图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # 子图1: 完整邻接矩阵
    ax1 = axes[0]
    im1 = ax1.imshow(adjacency_matrix, cmap='RdYlBu_r', aspect='auto',
                     interpolation='nearest', vmin=-1, vmax=1)
    
    ax1.set_title('Top-K有向图邻接矩阵\nA[i,j] = 节点i选择节点j的相似度',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('目标节点 j (被选择)', fontsize=12)
    ax1.set_ylabel('源节点 i (选择者)', fontsize=12)
    
    # 设置刻度
    node_num = adjacency_matrix.shape[0]
    ax1.set_xticks(range(node_num))
    ax1.set_yticks(range(node_num))
    ax1.set_xticklabels(range(node_num), fontsize=9)
    ax1.set_yticklabels(range(node_num), fontsize=9)
    
    # 添加网格
    ax1.set_xticks(np.arange(node_num) - 0.5, minor=True)
    ax1.set_yticks(np.arange(node_num) - 0.5, minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # 颜色条
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('相似度', fontsize=11)
    
    # 子图2: 使用seaborn绘制,更清晰
    ax2 = axes[1]
    sns.heatmap(adjacency_matrix, ax=ax2, cmap='RdYlBu_r', center=0,
                square=True, linewidths=0.5, linecolor='lightgray',
                cbar_kws={"shrink": 0.8, "label": "相似度"},
                vmin=-1, vmax=1,
                xticklabels=True, yticklabels=True)
    
    ax2.set_title('Top-K有向图邻接矩阵(带网格)\n对角线 = 自环(相似度=1.0)',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('目标节点 j', fontsize=12)
    ax2.set_ylabel('源节点 i', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 可视化已保存: {save_path}")
    
    plt.close()


def visualize_binary_adjacency(adjacency_matrix, save_path='topk_adjacency_binary.png'):
    """
    可视化二值化邻接矩阵(只显示是否有边)
    
    Args:
        adjacency_matrix: 邻接矩阵
        save_path: 保存路径
    """
    print(f"\n🎨 生成二值化邻接矩阵...")
    
    # 创建二值矩阵(非零 = 1, 零 = 0)
    binary_adj = (adjacency_matrix != 0).astype(int)
    
    fig, ax = plt.subplots(figsize=(12, 11))
    
    im = ax.imshow(binary_adj, cmap='binary', aspect='auto', vmin=0, vmax=1)
    
    ax.set_title('Top-K有向图二值邻接矩阵\n黑色 = 有边, 白色 = 无边',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('目标节点 j', fontsize=12)
    ax.set_ylabel('源节点 i', fontsize=12)
    
    # 设置刻度
    node_num = binary_adj.shape[0]
    ax.set_xticks(range(node_num))
    ax.set_yticks(range(node_num))
    ax.set_xticklabels(range(node_num), fontsize=9)
    ax.set_yticklabels(range(node_num), fontsize=9)
    
    # 网格
    ax.set_xticks(np.arange(node_num) - 0.5, minor=True)
    ax.set_yticks(np.arange(node_num) - 0.5, minor=True)
    ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 二值化邻接矩阵已保存: {save_path}")
    
    plt.close()


def analyze_adjacency_matrix(adjacency_matrix):
    """分析邻接矩阵的性质"""
    
    print("\n" + "="*80)
    print("邻接矩阵详细分析")
    print("="*80)
    
    node_num = adjacency_matrix.shape[0]
    
    # 1. 出度和入度
    print("\n1️⃣ 节点度数分析:")
    out_degree = (adjacency_matrix != 0).sum(axis=1)  # 行和
    in_degree = (adjacency_matrix != 0).sum(axis=0)   # 列和
    
    print(f"\n出度统计(每个节点选择的邻居数):")
    print(f"   - 平均出度: {out_degree.mean():.2f}")
    print(f"   - 最小出度: {out_degree.min()} (节点{out_degree.argmin()})")
    print(f"   - 最大出度: {out_degree.max()} (节点{out_degree.argmax()})")
    
    print(f"\n入度统计(每个节点被其他节点选择的次数):")
    print(f"   - 平均入度: {in_degree.mean():.2f}")
    print(f"   - 最小入度: {in_degree.min()} (节点{in_degree.argmin()})")
    print(f"   - 最大入度: {in_degree.max()} (节点{in_degree.argmax()})")
    
    # Top-10 入度最高的节点
    print(f"\n入度最高的10个节点(最受欢迎):")
    top_in_degree = np.argsort(in_degree)[::-1][:10]
    for rank, node in enumerate(top_in_degree, 1):
        print(f"   {rank:2d}. 节点{node:2d}: 入度={in_degree[node]}, 出度={out_degree[node]}")
    
    # 2. 对称性分析(找出双向连接)
    print("\n2️⃣ 双向连接分析:")
    
    bidirectional_count = 0
    bidirectional_pairs = []
    
    for i in range(node_num):
        for j in range(i+1, node_num):
            if adjacency_matrix[i, j] != 0 and adjacency_matrix[j, i] != 0:
                bidirectional_count += 1
                weight_avg = (adjacency_matrix[i, j] + adjacency_matrix[j, i]) / 2
                bidirectional_pairs.append((i, j, weight_avg))
    
    total_edges = np.count_nonzero(adjacency_matrix)
    bidirectional_edges = bidirectional_count * 2
    
    print(f"   - 总边数: {total_edges}")
    print(f"   - 双向边数: {bidirectional_edges}")
    print(f"   - 双向连接对数: {bidirectional_count}")
    print(f"   - 双向边占比: {bidirectional_edges / total_edges * 100:.1f}%")
    
    # 3. 自环分析
    print("\n3️⃣ 自环分析:")
    self_loops = np.diag(adjacency_matrix)
    self_loop_count = np.count_nonzero(self_loops)
    
    print(f"   - 有自环的节点数: {self_loop_count}/{node_num}")
    print(f"   - 自环相似度: 全部={'是' if np.all(np.abs(self_loops - 1.0) < 1e-6) else '否'} = 1.0")
    
    # 4. 连通性
    print("\n4️⃣ 图连通性(简单分析):")
    binary_adj = (adjacency_matrix != 0).astype(int)
    
    # 可达性(简单检查是否有孤立节点)
    total_connections = binary_adj.sum(axis=0) + binary_adj.sum(axis=1)
    isolated_nodes = np.where(total_connections == 1)[0]  # ==1 因为只有自环
    
    if len(isolated_nodes) == 0:
        print(f"   - 孤立节点: 无")
    else:
        print(f"   - 孤立节点: {isolated_nodes.tolist()}")
    
    print("\n" + "="*80)


def save_adjacency_matrix(adjacency_matrix, filepath='topk_adjacency_matrix.csv'):
    """保存邻接矩阵为CSV文件"""
    
    print(f"\n💾 保存邻接矩阵...")
    
    np.savetxt(filepath, adjacency_matrix, delimiter=',', fmt='%.6f')
    print(f"✓ 邻接矩阵已保存: {filepath}")
    
    # 同时保存二值版本
    binary_adj = (adjacency_matrix != 0).astype(int)
    binary_path = filepath.replace('.csv', '_binary.csv')
    np.savetxt(binary_path, binary_adj, delimiter=',', fmt='%d')
    print(f"✓ 二值邻接矩阵已保存: {binary_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成Top-K有向图邻接矩阵并可视化')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    
    args = parser.parse_args()
    
    # 1. 创建邻接矩阵
    adj_matrix, topk_idx, topk_val = create_topk_adjacency_matrix(
        model_path=args.model_path,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk
    )
    
    # 2. 可视化
    visualize_adjacency_matrix(adj_matrix)
    visualize_binary_adjacency(adj_matrix)
    
    # 3. 分析
    analyze_adjacency_matrix(adj_matrix)
    
    # 4. 保存
    save_adjacency_matrix(adj_matrix)
    
    print("\n" + "="*80)
    print("✓ 所有任务完成!")
    print("="*80)
    print("\n生成的文件:")
    print("   - topk_adjacency_matrix.png (邻接矩阵热力图)")
    print("   - topk_adjacency_binary.png (二值邻接矩阵)")
    print("   - topk_adjacency_matrix.csv (邻接矩阵CSV)")
    print("   - topk_adjacency_matrix_binary.csv (二值邻接矩阵CSV)")
    print("="*80)
