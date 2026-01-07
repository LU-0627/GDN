# -*- coding: utf-8 -*-
"""
生成GDN模型Top-K有向图的邻接矩阵可视化
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


def create_adjacency_matrix(model_path, node_num=27, dim=64, input_dim=15, topk=20):
    """
    创建基于Top-K邻居的有向图邻接矩阵
    
    Args:
        model_path: 模型路径
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
        
    Returns:
        adj_matrix: 邻接矩阵 [node_num, node_num]
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
        # 计算余弦相似度
        embeddings = model.embedding.weight
        weights = embeddings.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat
        
        # Top-K选择
        topk_values, topk_indices = torch.topk(cos_ji_mat, k=topk, dim=-1)
    
    print(f"\n🔄 构建邻接矩阵...")
    print(f"  - 矩阵大小: [{node_num}, {node_num}]")
    print(f"  - Top-K: {topk}")
    
    # 转换为numpy
    topk_indices_np = topk_indices.cpu().numpy()
    topk_values_np = topk_values.cpu().numpy()
    
    # 创建邻接矩阵
    # adj_matrix[i][j] = 从节点i到节点j的边的权重(如果存在)
    adj_matrix = np.zeros((node_num, node_num))
    
    for source in range(node_num):
        neighbors = topk_indices_np[source]
        similarities = topk_values_np[source]
        
        for neighbor, sim in zip(neighbors, similarities):
            # 排除自环(可选)
            # if neighbor != source:
            adj_matrix[source, neighbor] = sim
    
    print(f"✓ 邻接矩阵构建完成")
    
    # 统计信息
    non_zero = np.count_nonzero(adj_matrix)
    total = node_num * node_num
    sparsity = 1 - (non_zero / total)
    
    print(f"\n📊 邻接矩阵统计:")
    print(f"  - 非零元素: {non_zero}/{total} ({non_zero/total*100:.1f}%)")
    print(f"  - 稀疏度: {sparsity*100:.1f}%")
    print(f"  - 权重范围: [{adj_matrix[adj_matrix>0].min():.4f}, {adj_matrix.max():.4f}]")
    print(f"  - 平均权重(非零): {adj_matrix[adj_matrix>0].mean():.4f}")
    
    # 对角线统计(自环)
    diagonal = np.diag(adj_matrix)
    print(f"\n  - 对角线(自环):")
    print(f"    最小值: {diagonal.min():.4f}")
    print(f"    最大值: {diagonal.max():.4f}")
    print(f"    平均值: {diagonal.mean():.4f}")
    
    return adj_matrix, topk_indices_np, topk_values_np


def visualize_adjacency_matrix(adj_matrix, save_path='adjacency_matrix.png',
                               show_values=False, vmin=None, vmax=None):
    """
    可视化邻接矩阵热力图
    
    Args:
        adj_matrix: 邻接矩阵
        save_path: 保存路径
        show_values: 是否显示数值
        vmin, vmax: 颜色范围
    """
    print(f"\n🎨 生成邻接矩阵热力图...")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 创建热力图
    sns.heatmap(
        adj_matrix,
        cmap='YlOrRd',  # 黄-橙-红配色
        center=None,
        square=True,
        linewidths=0.1,
        linecolor='lightgray',
        cbar_kws={"label": "相似度权重", "shrink": 0.8},
        vmin=vmin if vmin is not None else 0,
        vmax=vmax if vmax is not None else 1,
        annot=show_values if adj_matrix.shape[0] <= 15 else False,
        fmt='.2f' if show_values else '',
        annot_kws={'fontsize': 6} if show_values else None,
        xticklabels=range(adj_matrix.shape[0]),
        yticklabels=range(adj_matrix.shape[0]),
        ax=ax
    )
    
    ax.set_title('Top-K有向图邻接矩阵\n(行=源节点, 列=目标节点)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('目标节点 (To)', fontsize=12)
    ax.set_ylabel('源节点 (From)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 热力图已保存: {save_path}")
    
    plt.close()


def visualize_adjacency_comparison(adj_matrix, save_path='adjacency_comparison.png'):
    """
    创建多视图对比可视化
    """
    print(f"\n🎨 生成对比可视化...")
    
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 完整邻接矩阵
    ax1 = plt.subplot(1, 4, 1)
    sns.heatmap(adj_matrix, cmap='YlOrRd', square=True, 
                cbar_kws={"label": "权重"}, ax=ax1,
                xticklabels=5, yticklabels=5)
    ax1.set_title('完整邻接矩阵', fontsize=12, fontweight='bold')
    ax1.set_xlabel('目标节点')
    ax1.set_ylabel('源节点')
    
    # 2. 二值化邻接矩阵(有边=1,无边=0)
    ax2 = plt.subplot(1, 4, 2)
    binary_adj = (adj_matrix > 0).astype(int)
    sns.heatmap(binary_adj, cmap='Greys', square=True,
                cbar_kws={"label": "连接"}, ax=ax2,
                xticklabels=5, yticklabels=5,
                vmin=0, vmax=1)
    ax2.set_title('连接模式\n(1=有连接, 0=无连接)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('目标节点')
    ax2.set_ylabel('源节点')
    
    # 3. 出度分布
    ax3 = plt.subplot(1, 4, 3)
    out_degree = (adj_matrix > 0).sum(axis=1)
    bars = ax3.bar(range(len(out_degree)), out_degree, color='steelblue', alpha=0.7)
    ax3.set_xlabel('节点', fontsize=11)
    ax3.set_ylabel('出度', fontsize=11)
    ax3.set_title('每个节点的出度', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 入度分布
    ax4 = plt.subplot(1, 4, 4)
    in_degree = (adj_matrix > 0).sum(axis=0)
    bars = ax4.bar(range(len(in_degree)), in_degree, color='coral', alpha=0.7)
    ax4.set_xlabel('节点', fontsize=11)
    ax4.set_ylabel('入度', fontsize=11)
    ax4.set_title('每个节点的入度', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {save_path}")
    
    plt.close()


def analyze_adjacency_structure(adj_matrix):
    """分析邻接矩阵的结构特性"""
    
    print("\n" + "="*80)
    print("邻接矩阵结构分析")
    print("="*80)
    
    node_num = adj_matrix.shape[0]
    
    # 1. 度数分析
    out_degree = (adj_matrix > 0).sum(axis=1)  # 每行非零元素=出度
    in_degree = (adj_matrix > 0).sum(axis=0)   # 每列非零元素=入度
    
    print("\n【出度统计】(每个节点指向多少个其他节点)")
    print(f"  最小出度: {out_degree.min():.0f} (节点{out_degree.argmin()})")
    print(f"  最大出度: {out_degree.max():.0f} (节点{out_degree.argmax()})")
    print(f"  平均出度: {out_degree.mean():.2f}")
    print(f"  标准差: {out_degree.std():.2f}")
    
    print("\n【入度统计】(每个节点被多少个其他节点指向)")
    print(f"  最小入度: {in_degree.min():.0f} (节点{in_degree.argmin()})")
    print(f"  最大入度: {in_degree.max():.0f} (节点{in_degree.argmax()})")
    print(f"  平均入度: {in_degree.mean():.2f}")
    print(f"  标准差: {in_degree.std():.2f}")
    
    # 2. 对称性分析(双向连接)
    print("\n【对称性分析】")
    symmetric_edges = 0
    total_edges = np.count_nonzero(adj_matrix)
    
    for i in range(node_num):
        for j in range(i+1, node_num):
            if adj_matrix[i, j] > 0 and adj_matrix[j, i] > 0:
                symmetric_edges += 2  # 双向算2条边
    
    print(f"  双向边: {symmetric_edges}/{total_edges} ({symmetric_edges/total_edges*100:.1f}%)")
    print(f"  单向边: {total_edges - symmetric_edges}/{total_edges} ({(total_edges - symmetric_edges)/total_edges*100:.1f}%)")
    
    # 3. 最强连接
    print("\n【最强的10条边】")
    # 排除对角线
    adj_no_diag = adj_matrix.copy()
    np.fill_diagonal(adj_no_diag, 0)
    
    flat_indices = np.argsort(adj_no_diag.flatten())[::-1][:10]
    positions = np.unravel_index(flat_indices, adj_no_diag.shape)
    
    for rank, (i, j) in enumerate(zip(positions[0], positions[1]), 1):
        weight = adj_matrix[i, j]
        # 检查是否双向
        is_bidirectional = adj_matrix[j, i] > 0
        print(f"  {rank:2d}. 节点{i:2d} → 节点{j:2d}  |  权重: {weight:.6f}  "
              f"{'(双向)' if is_bidirectional else ''}")
    
    # 4. Hub节点(高出度或高入度)
    print("\n【Hub节点分析】")
    print("  高出度节点(Top 5):")
    top_out = np.argsort(out_degree)[::-1][:5]
    for rank, node in enumerate(top_out, 1):
        print(f"    {rank}. 节点{node:2d}: 出度={out_degree[node]:.0f}")
    
    print("  高入度节点(Top 5):")
    top_in = np.argsort(in_degree)[::-1][:5]
    for rank, node in enumerate(top_in, 1):
        print(f"    {rank}. 节点{node:2d}: 入度={in_degree[node]:.0f}")


def save_adjacency_matrix_to_file(adj_matrix, filepath='adjacency_matrix.csv'):
    """保存邻接矩阵到CSV文件"""
    np.savetxt(filepath, adj_matrix, fmt='%.6f', delimiter=',')
    print(f"\n✓ 邻接矩阵已保存到: {filepath}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成Top-K有向图邻接矩阵可视化')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    parser.add_argument('--show_values', action='store_true',
                        help='在热力图上显示数值(仅适用于小矩阵)')
    parser.add_argument('--save_csv', action='store_true',
                        help='保存邻接矩阵为CSV文件')
    
    args = parser.parse_args()
    
    # 创建邻接矩阵
    adj_matrix, topk_idx, topk_val = create_adjacency_matrix(
        model_path=args.model_path,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk
    )
    
    # 可视化
    visualize_adjacency_matrix(
        adj_matrix,
        save_path='adjacency_matrix_heatmap.png',
        show_values=args.show_values
    )
    
    # 对比可视化
    visualize_adjacency_comparison(
        adj_matrix,
        save_path='adjacency_matrix_analysis.png'
    )
    
    # 结构分析
    analyze_adjacency_structure(adj_matrix)
    
    # 保存CSV
    if args.save_csv:
        save_adjacency_matrix_to_file(adj_matrix)
    
    print("\n" + "="*80)
    print("✓ 所有可视化和分析完成!")
    print("="*80)
    print("\n生成的文件:")
    print("  1. adjacency_matrix_heatmap.png - 邻接矩阵热力图")
    print("  2. adjacency_matrix_analysis.png - 多视图分析")
    if args.save_csv:
        print("  3. adjacency_matrix.csv - 邻接矩阵CSV文件")
    print("="*80)
