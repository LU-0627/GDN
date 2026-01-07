# -*- coding: utf-8 -*-
"""
可视化单个节点的Top-K邻居有向图
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


def visualize_single_node_graph(model_path, target_node, node_num=27, dim=64, 
                                input_dim=15, topk=20, show_bidirectional=True):
    """
    可视化单个节点的Top-K邻居图
    
    Args:
        model_path: 模型路径
        target_node: 目标节点ID
        node_num: 总节点数
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
        show_bidirectional: 是否显示双向连接(邻居也选择了目标节点)
    """
    device = get_device()
    
    print("="*80)
    print(f"生成节点{target_node}的Top-K邻居有向图")
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
    
    topk_indices_np = topk_indices.cpu().numpy()
    topk_values_np = topk_values.cpu().numpy()
    
    # 获取目标节点的邻居
    target_neighbors = topk_indices_np[target_node]
    target_similarities = topk_values_np[target_node]
    
    print(f"\n🎯 节点{target_node}的Top-{topk}邻居:")
    print(f"  邻居列表: {target_neighbors.tolist()}")
    
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加中心节点
    G.add_node(target_node)
    
    # 添加邻居节点和边
    outgoing_edges = []
    for neighbor, sim in zip(target_neighbors, target_similarities):
        neighbor = int(neighbor)
        if neighbor == target_node:
            continue
        
        G.add_node(neighbor)
        G.add_edge(target_node, neighbor, weight=sim, edge_type='outgoing')
        outgoing_edges.append((target_node, neighbor, sim))
    
    print(f"  出边数: {len(outgoing_edges)}")
    
    # 检查双向连接
    incoming_edges = []
    bidirectional_nodes = set()
    
    if show_bidirectional:
        for neighbor in target_neighbors:
            neighbor = int(neighbor)
            if neighbor == target_node:
                continue
            
            # 检查邻居是否也选择了目标节点
            neighbor_topk = topk_indices_np[neighbor]
            if target_node in neighbor_topk:
                # 找到相似度
                idx = np.where(neighbor_topk == target_node)[0][0]
                sim = topk_values_np[neighbor][idx]
                
                G.add_edge(neighbor, target_node, weight=sim, edge_type='incoming')
                incoming_edges.append((neighbor, target_node, sim))
                bidirectional_nodes.add(neighbor)
        
        print(f"  入边数(双向连接): {len(incoming_edges)}")
        print(f"  双向连接的节点: {sorted(list(bidirectional_nodes))}")
    
    # 可视化
    print(f"\n🎨 生成可视化...")
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # 使用环形布局,但将目标节点放在中心
    # 其他节点围绕中心排列
    pos = {}
    pos[target_node] = (0, 0)  # 中心
    
    # 邻居节点环形排列
    neighbors = [n for n in G.nodes() if n != target_node]
    n_neighbors = len(neighbors)
    
    import math
    for i, neighbor in enumerate(neighbors):
        angle = 2 * math.pi * i / n_neighbors
        radius = 2
        pos[neighbor] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # 绘制节点
    # 中心节点(红色,大)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[target_node],
        node_size=2000,
        node_color='red',
        alpha=0.9,
        edgecolors='black',
        linewidths=3,
        ax=ax,
        label='目标节点'
    )
    
    # 双向连接的节点(绿色)
    if bidirectional_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(bidirectional_nodes),
            node_size=1200,
            node_color='lightgreen',
            alpha=0.8,
            edgecolors='darkgreen',
            linewidths=2,
            ax=ax,
            label='双向连接'
        )
    
    # 单向连接的节点(蓝色)
    unidirectional_nodes = [n for n in neighbors if n not in bidirectional_nodes]
    if unidirectional_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=unidirectional_nodes,
            node_size=1000,
            node_color='lightblue',
            alpha=0.7,
            edgecolors='darkblue',
            linewidths=2,
            ax=ax,
            label='单向连接'
        )
    
    # 绘制节点标签
    nx.draw_networkx_labels(
        G, pos,
        font_size=11,
        font_weight='bold',
        font_color='white',
        ax=ax
    )
    
    # 绘制出边(从中心到邻居)
    outgoing_edge_list = [(u, v) for u, v, d in G.edges(data=True) 
                         if d.get('edge_type') == 'outgoing']
    outgoing_weights = [G[u][v]['weight'] for u, v in outgoing_edge_list]
    
    nx.draw_networkx_edges(
        G, pos,
        edgelist=outgoing_edge_list,
        edge_color=outgoing_weights,
        edge_cmap=plt.cm.Reds,
        width=[w*3 for w in outgoing_weights],
        alpha=0.7,
        arrows=True,
        arrowstyle='->',
        arrowsize=25,
        connectionstyle='arc3,rad=0.1',
        ax=ax,
        label='出边'
    )
    
    # 绘制入边(从邻居到中心)
    if show_bidirectional:
        incoming_edge_list = [(u, v) for u, v, d in G.edges(data=True) 
                             if d.get('edge_type') == 'incoming']
        incoming_weights = [G[u][v]['weight'] for u, v in incoming_edge_list]
        
        if incoming_edge_list:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=incoming_edge_list,
                edge_color=incoming_weights,
                edge_cmap=plt.cm.Greens,
                width=[w*3 for w in incoming_weights],
                alpha=0.7,
                arrows=True,
                arrowstyle='->',
                arrowsize=25,
                connectionstyle='arc3,rad=-0.1',
                ax=ax,
                label='入边'
            )
    
    # 添加边标签(显示相似度)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        edge_labels[(u, v)] = f"{d['weight']:.3f}"
    
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels,
        font_size=8,
        font_color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        ax=ax
    )
    
    ax.set_title(f'节点{target_node}的Top-{topk}邻居关系图\n'
                f'出边:{len(outgoing_edges)}条, 双向连接:{len(incoming_edges)}对',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.axis('off')
    
    plt.tight_layout()
    save_path = f'node_{target_node}_topk_graph.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ 可视化已保存: {save_path}")
    
    # 打印详细信息
    print("\n" + "="*80)
    print(f"节点{target_node}的详细邻居信息")
    print("="*80)
    
    print(f"\n出边(节点{target_node}选择的Top-{topk}邻居):")
    print(f"  {'排名':<6} {'邻居':<6} {'相似度':<12} {'双向':<8} {'相似度条形图'}")
    print(f"  {'-'*70}")
    
    for rank, (neighbor, sim) in enumerate(zip(target_neighbors, target_similarities), 1):
        if neighbor == target_node:
            print(f"  {rank:<6} {neighbor:<6} {sim:<12.6f} {'(自己)':<8}")
            continue
        
        is_bidirectional = neighbor in bidirectional_nodes
        bar_length = int(sim * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        
        print(f"  {rank:<6} {neighbor:<6} {sim:<12.6f} {'✓' if is_bidirectional else '✗':<8} {bar}")
    
    if bidirectional_nodes:
        print(f"\n双向连接详情:")
        print(f"  {'邻居':<6} {'出边权重':<12} {'入边权重':<12} {'平均':<12}")
        print(f"  {'-'*50}")
        
        for neighbor in sorted(list(bidirectional_nodes)):
            out_weight = G[target_node][neighbor]['weight']
            in_weight = G[neighbor][target_node]['weight']
            avg_weight = (out_weight + in_weight) / 2
            print(f"  {neighbor:<6} {out_weight:<12.6f} {in_weight:<12.6f} {avg_weight:<12.6f}")
    
    print("\n" + "="*80)
    print("✓ 完成!")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化单个节点的Top-K邻居图')
    parser.add_argument('--node', type=int, required=True,
                        help='目标节点ID')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点总数')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    parser.add_argument('--no_bidirectional', action='store_true',
                        help='不显示双向连接')
    
    args = parser.parse_args()
    
    visualize_single_node_graph(
        model_path=args.model_path,
        target_node=args.node,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk,
        show_bidirectional=not args.no_bidirectional
    )
