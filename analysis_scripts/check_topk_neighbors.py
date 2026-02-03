# -*- coding: utf-8 -*-
"""
检查GDN模型中每个节点的Top-K邻居选择
展示模型如何基于嵌入相似度选择邻居节点
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from models.GDN import GDN
from util.env import get_device


def check_topk_neighbors(model_path, node_num=27, dim=64, input_dim=15, topk=20, 
                         show_all=False, specific_nodes=None):
    """
    检查Top-K邻居选择
    
    Args:
        model_path: 模型路径
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
        show_all: 是否显示所有节点
        specific_nodes: 指定要查看的节点列表
    """
    device = get_device()
    
    # 加载模型
    print("="*80)
    print("检查GDN模型的Top-K邻居选择")
    print("="*80)
    print(f"\n📂 加载模型: {model_path}")
    
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
        # 获取嵌入向量
        embeddings = model.embedding.weight  # [27, 64]
        
        print(f"\n📊 嵌入向量信息:")
        print(f"  - 形状: {embeddings.shape}")
        print(f"  - 节点数: {node_num}")
        print(f"  - 嵌入维度: {dim}")
        print(f"  - Top-K值: {topk}")
        
        # 计算余弦相似度矩阵
        print(f"\n🔄 计算节点间余弦相似度...")
        
        # 方法与GDN中完全一致
        weights = embeddings.view(node_num, -1)  # [27, 64]
        cos_ji_mat = torch.matmul(weights, weights.T)  # [27, 27]
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat  # [27, 27]
        
        print(f"✓ 余弦相似度矩阵形状: {cos_ji_mat.shape}")
        print(f"  - 数值范围: [{cos_ji_mat.min().item():.4f}, {cos_ji_mat.max().item():.4f}]")
        
        # Top-K 选择邻居
        print(f"\n🎯 执行Top-{topk}邻居选择...")
        topk_values, topk_indices = torch.topk(cos_ji_mat, k=topk, dim=-1)
        # topk_indices: [27, 20] - 每个节点的Top-20个最相似邻居的索引
        # topk_values: [27, 20] - 对应的相似度值
        
        print(f"✓ Top-K索引形状: {topk_indices.shape}")
        print(f"✓ Top-K相似度形状: {topk_values.shape}")
        
        # 转换为CPU numpy便于显示
        topk_indices_np = topk_indices.cpu().numpy()
        topk_values_np = topk_values.cpu().numpy()
        
        # 显示结果
        print("\n" + "="*80)
        print(f"每个节点的Top-{topk}邻居")
        print("="*80)
        
        # 决定显示哪些节点
        if specific_nodes is not None:
            nodes_to_show = specific_nodes
        elif show_all:
            nodes_to_show = range(node_num)
        else:
            nodes_to_show = range(min(10, node_num))  # 默认显示前10个
        
        for node_id in nodes_to_show:
            if node_id >= node_num:
                print(f"⚠️ 节点{node_id}越界(最大:{node_num-1})")
                continue
                
            neighbors = topk_indices_np[node_id]
            similarities = topk_values_np[node_id]
            
            print(f"\n{'='*80}")
            print(f"节点 {node_id:2d} 的Top-{topk}邻居:")
            print(f"{'='*80}")
            
            # 检查第一个邻居是否是自己
            if neighbors[0] == node_id:
                print("  ✓ 第1个邻居是自己(相似度=1.0)")
                print(f"\n  最相似的{topk-1}个其他节点:")
                start_idx = 1
            else:
                print(f"  ⚠️ 注意:第1个邻居不是自己,而是节点{neighbors[0]}")
                print(f"\n  最相似的{topk}个节点:")
                start_idx = 0
            
            print(f"  {'排名':<6} {'节点ID':<8} {'相似度':<12} {'相似度条形图'}")
            print(f"  {'-'*60}")
            
            for rank, (neighbor, sim) in enumerate(zip(neighbors[start_idx:], 
                                                       similarities[start_idx:]), 
                                                   start=1):
                # 创建相似度条形图
                bar_length = int(sim * 30)  # 最大30个字符
                bar = '█' * bar_length + '░' * (30 - bar_length)
                
                # 标记特别高的相似度
                if sim > 0.8:
                    marker = "🔥"
                elif sim > 0.5:
                    marker = "✓"
                else:
                    marker = " "
                
                print(f"  {rank:<6} {neighbor:<8} {sim:<12.6f} {bar} {marker}")
        
        # 统计分析
        print("\n" + "="*80)
        print("Top-K邻居统计分析")
        print("="*80)
        
        # 1. 每个节点Top-K邻居的平均相似度
        avg_topk_sim = topk_values_np.mean(axis=1)
        print(f"\n各节点Top-{topk}邻居的平均相似度:")
        print(f"  - 最高: {avg_topk_sim.max():.6f} (节点{avg_topk_sim.argmax()})")
        print(f"  - 最低: {avg_topk_sim.min():.6f} (节点{avg_topk_sim.argmin()})")
        print(f"  - 平均: {avg_topk_sim.mean():.6f}")
        
        # 2. 最受欢迎的节点(被选为邻居最多的节点)
        neighbor_counts = np.zeros(node_num)
        for i in range(node_num):
            neighbors = topk_indices_np[i]
            for neighbor in neighbors:
                if neighbor != i:  # 排除自己
                    neighbor_counts[neighbor] += 1
        
        print(f"\n最受欢迎的节点(被选为Top-{topk}邻居最多的节点):")
        popular_nodes = np.argsort(neighbor_counts)[::-1][:10]
        for rank, node_id in enumerate(popular_nodes, 1):
            count = int(neighbor_counts[node_id])
            print(f"  {rank:2d}. 节点{node_id:2d}: 被选择{count:2d}次 "
                  f"({'█' * (count // 2)})")
        
        # 3. 孤立的节点(很少被选为邻居)
        print(f"\n较孤立的节点(很少被选为邻居):")
        isolated_nodes = np.argsort(neighbor_counts)[:5]
        for rank, node_id in enumerate(isolated_nodes, 1):
            count = int(neighbor_counts[node_id])
            print(f"  {rank}. 节点{node_id:2d}: 仅被选择{count:2d}次")
        
        # 4. 互为邻居的节点对(双向选择)
        print(f"\n互为Top-{topk}邻居的节点对:")
        mutual_pairs = []
        for i in range(node_num):
            neighbors_i = set(topk_indices_np[i].tolist())
            for j in range(i+1, node_num):
                neighbors_j = set(topk_indices_np[j].tolist())
                if i in neighbors_j and j in neighbors_i:
                    # 找到相似度
                    sim_i_to_j = topk_values_np[i][topk_indices_np[i].tolist().index(j)]
                    sim_j_to_i = topk_values_np[j][topk_indices_np[j].tolist().index(i)]
                    mutual_pairs.append((i, j, sim_i_to_j, sim_j_to_i))
        
        mutual_pairs.sort(key=lambda x: (x[2] + x[3]) / 2, reverse=True)
        
        print(f"  找到 {len(mutual_pairs)} 对互为邻居的节点")
        print(f"  前10个最强的双向关系:")
        for rank, (i, j, sim_ij, sim_ji) in enumerate(mutual_pairs[:10], 1):
            avg_sim = (sim_ij + sim_ji) / 2
            print(f"  {rank:2d}. 节点{i:2d} ↔ 节点{j:2d}  |  "
                  f"相似度: {sim_ij:.4f}/{sim_ji:.4f} (平均:{avg_sim:.4f})")
        
        print("\n" + "="*80)
        print("✓ 检查完成!")
        print("="*80)
        
        return topk_indices, topk_values, cos_ji_mat


def save_topk_to_file(topk_indices, topk_values, filepath='topk_neighbors.txt'):
    """保存Top-K邻居到文件"""
    topk_indices_np = topk_indices.cpu().numpy()
    topk_values_np = topk_values.cpu().numpy()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("GDN模型 - Top-K邻居详细列表\n")
        f.write("="*80 + "\n\n")
        
        for node_id in range(len(topk_indices_np)):
            f.write(f"节点 {node_id:2d}:\n")
            f.write(f"  邻居: {topk_indices_np[node_id].tolist()}\n")
            f.write(f"  相似度: {[f'{v:.4f}' for v in topk_values_np[node_id]]}\n")
            f.write("\n")
    
    print(f"✓ Top-K邻居已保存到: {filepath}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='检查GDN模型的Top-K邻居选择')
    parser.add_argument('--model_path', type=str, 
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    parser.add_argument('--show_all', action='store_true', 
                        help='显示所有节点的邻居')
    parser.add_argument('--nodes', type=int, nargs='+',
                        help='指定要查看的节点,例如: --nodes 0 1 5 10')
    parser.add_argument('--save', action='store_true',
                        help='保存结果到文件')
    
    args = parser.parse_args()
    
    # 执行检查
    topk_indices, topk_values, cos_mat = check_topk_neighbors(
        model_path=args.model_path,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk,
        show_all=args.show_all,
        specific_nodes=args.nodes
    )
    
    # 保存结果
    if args.save:
        save_topk_to_file(topk_indices, topk_values)
