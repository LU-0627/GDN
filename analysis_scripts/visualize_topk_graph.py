# -*- coding: utf-8 -*-
"""
基于Top-K邻居生成有向图可视化
展示GDN模型学习到的图结构
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from models.GDN import GDN
from util.env import get_device

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_topk_directed_graph(model_path, node_num=27, dim=64, input_dim=15, topk=20,
                                similarity_threshold=0.0, top_edges=None):
    """
    创建基于Top-K邻居的有向图
    
    Args:
        model_path: 模型路径
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
        similarity_threshold: 相似度阈值,低于此值的边不显示
        top_edges: 如果指定,只显示相似度最高的N条边
        
    Returns:
        G: NetworkX有向图对象
        pos: 节点位置
        topk_indices: Top-K索引
        topk_values: Top-K相似度值
    """
    device = get_device()
    
    print("="*80)
    print("生成GDN模型的Top-K有向图")
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
        # 获取嵌入向量并计算相似度
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
        
        print(f"\n🎯 Top-K参数: K={topk}")
        print(f"📊 总共可能的边数: {node_num * topk}")
        
    # 转换为numpy
    topk_indices_np = topk_indices.cpu().numpy()
    topk_values_np = topk_values.cpu().numpy()
    
    # 创建有向图
    print(f"\n🔄 创建有向图...")
    G = nx.DiGraph()
    
    # 添加所有节点
    for i in range(node_num):
        G.add_node(i)
    
    # 添加边(基于Top-K邻居)
    edge_list = []
    for source in range(node_num):
        neighbors = topk_indices_np[source]
        similarities = topk_values_np[source]
        
        for neighbor, sim in zip(neighbors, similarities):
            # 跳过自环
            if neighbor == source:
                continue
            
            # 应用相似度阈值
            if sim < similarity_threshold:
                continue
            
            edge_list.append((source, neighbor, sim))
            G.add_edge(source, int(neighbor), weight=sim)
    
    # 如果指定了top_edges,只保留相似度最高的边
    if top_edges is not None and top_edges < len(edge_list):
        print(f"⚠️ 只显示相似度最高的{top_edges}条边")
        edge_list.sort(key=lambda x: x[2], reverse=True)
        edge_list = edge_list[:top_edges]
        
        # 重新创建图
        G = nx.DiGraph()
        for i in range(node_num):
            G.add_node(i)
        for source, target, weight in edge_list:
            G.add_edge(source, int(target), weight=weight)
    
    print(f"✓ 图创建完成")
    print(f"  - 节点数: {G.number_of_nodes()}")
    print(f"  - 边数: {G.number_of_edges()}")
    print(f"  - 平均出度: {sum(dict(G.out_degree()).values()) / node_num:.2f}")
    print(f"  - 平均入度: {sum(dict(G.in_degree()).values()) / node_num:.2f}")
    
    return G, topk_indices_np, topk_values_np


def visualize_directed_graph(G, save_path='topk_directed_graph.png', 
                             layout='spring', figsize=(16, 16),
                             show_edge_labels=False, node_size_by_degree=True):
    """
    可视化有向图
    
    Args:
        G: NetworkX有向图
        save_path: 保存路径
        layout: 布局算法 ('spring', 'circular', 'kamada_kawai', 'shell')
        figsize: 图片大小
        show_edge_labels: 是否显示边权重
        node_size_by_degree: 节点大小是否由度数决定
    """
    print(f"\n🎨 生成可视化...")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算布局
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # 节点大小(根据入度)
    if node_size_by_degree:
        in_degrees = dict(G.in_degree())
        node_sizes = [300 + in_degrees[node] * 50 for node in G.nodes()]
    else:
        node_sizes = 500
    
    # 节点颜色(根据出度)
    out_degrees = dict(G.out_degree())
    node_colors = [out_degrees[node] for node in G.nodes()]
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        cmap='YlOrRd',
        alpha=0.9,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )
    
    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold',
        font_color='white',
        ax=ax
    )
    
    # 绘制边
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # 边的颜色和宽度由权重决定
    edge_colors = weights
    edge_widths = [w * 2 for w in weights]
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowstyle='->',
        arrowsize=15,
        connectionstyle='arc3,rad=0.1',  # 弧形边避免重叠
        ax=ax
    )
    
    # 显示边权重(可选)
    if show_edge_labels and G.number_of_edges() <= 100:
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" 
                      for u, v in G.edges()}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels,
            font_size=7,
            ax=ax
        )
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                               norm=plt.Normalize(vmin=min(node_colors), 
                                                 vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('节点出度', fontsize=12)
    
    ax.set_title(f'GDN Top-K有向图 ({G.number_of_nodes()}个节点, {G.number_of_edges()}条边)\n'
                f'节点大小=入度, 节点颜色=出度, 边宽度=相似度',
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 可视化已保存: {save_path}")
    
    plt.close()


def create_multiple_visualizations(model_path, node_num=27, dim=64, input_dim=15, topk=20):
    """创建多种不同的可视化"""
    
    # 1. 完整的Top-K图
    print("\n" + "="*80)
    print("1️⃣ 生成完整Top-K有向图")
    print("="*80)
    G_full, topk_idx, topk_val = create_topk_directed_graph(
        model_path, node_num, dim, input_dim, topk, similarity_threshold=0.0
    )
    visualize_directed_graph(
        G_full, 
        save_path='topk_graph_full.png',
        layout='spring',
        figsize=(16, 16)
    )
    
    # 2. 只显示高相似度的边(阈值0.5)
    print("\n" + "="*80)
    print("2️⃣ 生成高相似度边图(阈值>0.5)")
    print("="*80)
    G_high_sim, _, _ = create_topk_directed_graph(
        model_path, node_num, dim, input_dim, topk, similarity_threshold=0.5
    )
    visualize_directed_graph(
        G_high_sim,
        save_path='topk_graph_high_similarity.png',
        layout='spring',
        figsize=(14, 14),
        show_edge_labels=True
    )
    
    # 3. 只显示Top-100最强的边
    print("\n" + "="*80)
    print("3️⃣ 生成Top-100最强边图")
    print("="*80)
    G_top100, _, _ = create_topk_directed_graph(
        model_path, node_num, dim, input_dim, topk, top_edges=100
    )
    visualize_directed_graph(
        G_top100,
        save_path='topk_graph_top100_edges.png',
        layout='kamada_kawai',
        figsize=(14, 14),
        show_edge_labels=False
    )
    
    # 4. 环形布局
    print("\n" + "="*80)
    print("4️⃣ 生成环形布局图")
    print("="*80)
    visualize_directed_graph(
        G_high_sim,
        save_path='topk_graph_circular.png',
        layout='circular',
        figsize=(14, 14)
    )
    
    # 分析图结构
    print("\n" + "="*80)
    print("📊 图结构分析")
    print("="*80)
    
    # 计算中心性
    print("\n节点重要性排名(基于入度中心性):")
    in_degree_centrality = nx.in_degree_centrality(G_full)
    sorted_nodes = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (node, centrality) in enumerate(sorted_nodes[:10], 1):
        in_deg = G_full.in_degree(node)
        out_deg = G_full.out_degree(node)
        print(f"  {rank:2d}. 节点{node:2d}: 入度中心性={centrality:.4f} "
              f"(入度={in_deg}, 出度={out_deg})")
    
    # 找出强连接的节点对(互相选择对方)
    print("\n互为Top-K邻居的节点对(双向箭头):")
    mutual_edges = []
    for u, v in G_full.edges():
        if G_full.has_edge(v, u):
            if u < v:  # 避免重复
                weight_uv = G_full[u][v]['weight']
                weight_vu = G_full[v][u]['weight']
                mutual_edges.append((u, v, weight_uv, weight_vu))
    
    mutual_edges.sort(key=lambda x: (x[2] + x[3])/2, reverse=True)
    
    print(f"  找到{len(mutual_edges)}对双向连接")
    print(f"  前10个最强双向连接:")
    for rank, (u, v, w_uv, w_vu) in enumerate(mutual_edges[:10], 1):
        avg_weight = (w_uv + w_vu) / 2
        print(f"  {rank:2d}. 节点{u:2d} ↔ 节点{v:2d}  |  "
              f"权重: {w_uv:.4f}/{w_vu:.4f} (平均:{avg_weight:.4f})")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成GDN Top-K有向图可视化')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'full', 'high_sim', 'top_edges', 'circular'],
                        help='可视化模式')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='相似度阈值(仅用于high_sim模式)')
    parser.add_argument('--top_edges', type=int, default=100,
                        help='显示的最强边数量(仅用于top_edges模式)')
    
    args = parser.parse_args()
    
    if args.mode == 'all':
        # 生成所有可视化
        create_multiple_visualizations(
            args.model_path, args.node_num, args.dim, args.input_dim, args.topk
        )
    else:
        # 生成单个可视化
        G, _, _ = create_topk_directed_graph(
            args.model_path, args.node_num, args.dim, args.input_dim, args.topk,
            similarity_threshold=args.threshold if args.mode == 'high_sim' else 0.0,
            top_edges=args.top_edges if args.mode == 'top_edges' else None
        )
        
        layout = 'circular' if args.mode == 'circular' else 'spring'
        visualize_directed_graph(
            G,
            save_path=f'topk_graph_{args.mode}.png',
            layout=layout,
            show_edge_labels=(G.number_of_edges() <= 100)
        )
    
    print("\n✓ 所有可视化已完成!")
