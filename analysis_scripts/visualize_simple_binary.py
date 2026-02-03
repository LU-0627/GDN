# -*- coding: utf-8 -*-
"""
生成简洁的文本格式邻接矩阵
1 = 有边, 空格 = 无边
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from models.GDN import GDN
from util.env import get_device


def create_text_adjacency_matrix(model_path, node_num=27, dim=64, input_dim=15, topk=20):
    """
    创建文本格式的邻接矩阵
    """
    device = get_device()
    
    print("="*80)
    print("生成Top-K邻接矩阵(文本格式)")
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
    
    with torch.no_grad():
        # 计算相似度和Top-K
        embeddings = model.embedding.weight
        weights = embeddings.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat
        
        topk_values, topk_indices = torch.topk(cos_ji_mat, k=topk, dim=-1)
    
    topk_indices_np = topk_indices.cpu().numpy()
    
    # 构建二值邻接矩阵
    binary_adj = np.zeros((node_num, node_num), dtype=int)
    
    for i in range(node_num):
        neighbors = topk_indices_np[i]
        for neighbor in neighbors:
            binary_adj[i, neighbor] = 1
    
    return binary_adj


def print_text_adjacency_matrix(binary_adj, output_file='adjacency_matrix_text.txt'):
    """
    打印文本格式的邻接矩阵
    1 = 有边, 空格 = 无边
    """
    node_num = binary_adj.shape[0]
    
    # 打印到控制台
    print("\n" + "="*100)
    print("Top-K邻接矩阵 (1=有边, 空格=无边)")
    print("="*100)
    
    # 打印表头
    header = "    "  # 行号占位
    for j in range(node_num):
        header += f" {j:2d}"
    print(header)
    print("    " + "-" * (node_num * 3))
    
    # 打印每一行
    for i in range(node_num):
        row_str = f"{i:2d} |"
        for j in range(node_num):
            if binary_adj[i, j] == 1:
                row_str += "  1"
            else:
                row_str += "   "  # 三个空格
        print(row_str)
    
    print("="*100)
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Top-K有向图邻接矩阵\n")
        f.write("格式说明: 1 = 有边(节点i选择节点j作为Top-K邻居), 空格 = 无边\n")
        f.write("行 = 源节点(选择者), 列 = 目标节点(被选择)\n")
        f.write("="*100 + "\n\n")
        
        # 表头
        header = "    "
        for j in range(node_num):
            header += f" {j:2d}"
        f.write(header + "\n")
        f.write("    " + "-" * (node_num * 3) + "\n")
        
        # 每一行
        for i in range(node_num):
            row_str = f"{i:2d} |"
            for j in range(node_num):
                if binary_adj[i, j] == 1:
                    row_str += "  1"
                else:
                    row_str += "   "
            f.write(row_str + "\n")
        
        f.write("\n" + "="*100 + "\n")
        
        # 添加统计信息
        f.write("\n统计信息:\n")
        f.write(f"  - 节点数: {node_num}\n")
        f.write(f"  - 总边数: {binary_adj.sum()}\n")
        f.write(f"  - 每个节点平均出度: {binary_adj.sum(axis=1).mean():.2f}\n")
        f.write(f"  - 每个节点平均入度: {binary_adj.sum(axis=0).mean():.2f}\n")
        
        # 双向连接统计
        bidirectional = 0
        for i in range(node_num):
            for j in range(i+1, node_num):
                if binary_adj[i, j] == 1 and binary_adj[j, i] == 1:
                    bidirectional += 1
        
        f.write(f"  - 双向连接对数: {bidirectional}\n")
    
    print(f"\n✓ 文本格式邻接矩阵已保存: {output_file}")


def print_compact_matrix(binary_adj, output_file='adjacency_matrix_compact.txt'):
    """
    更紧凑的格式,每个节点一行列出其邻居
    """
    node_num = binary_adj.shape[0]
    
    print("\n" + "="*80)
    print("紧凑格式: 每个节点的Top-K邻居列表")
    print("="*80)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Top-K邻居列表(紧凑格式)\n")
        f.write("="*80 + "\n\n")
        
        for i in range(node_num):
            neighbors = np.where(binary_adj[i] == 1)[0].tolist()
            neighbors_str = ' '.join([f"{n:2d}" for n in neighbors])
            
            line = f"节点 {i:2d} → [{neighbors_str}]"
            print(line)
            f.write(line + "\n")
    
    print(f"\n✓ 紧凑格式已保存: {output_file}")


def create_visual_grid(binary_adj, output_file='adjacency_matrix_grid.txt'):
    """
    创建网格可视化,更清晰
    """
    node_num = binary_adj.shape[0]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Top-K邻接矩阵网格可视化\n")
        f.write("█ = 有边, ░ = 无边\n")
        f.write("="*100 + "\n\n")
        
        # 表头
        header = "     "
        for j in range(node_num):
            header += f"{j:2d} "
        f.write(header + "\n")
        f.write("    +" + "---" * node_num + "+\n")
        
        # 每一行
        for i in range(node_num):
            row_str = f"{i:2d}  |"
            for j in range(node_num):
                if binary_adj[i, j] == 1:
                    row_str += " █ "
                else:
                    row_str += " ░ "
            row_str += "|"
            f.write(row_str + "\n")
        
        f.write("    +" + "---" * node_num + "+\n")
    
    print(f"\n✓ 网格可视化已保存: {output_file}")
    
    # 同时显示在控制台
    print("\n" + "="*100)
    print("网格可视化 (█=有边, ░=无边)")
    print("="*100)
    
    # 表头
    header = "     "
    for j in range(node_num):
        header += f"{j:2d} "
    print(header)
    print("    +" + "---" * node_num + "+")
    
    for i in range(node_num):
        row_str = f"{i:2d}  |"
        for j in range(node_num):
            if binary_adj[i, j] == 1:
                row_str += " █ "
            else:
                row_str += " ░ "
        row_str += "|"
        print(row_str)
    
    print("    +" + "---" * node_num + "+")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='生成文本格式Top-K邻接矩阵')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    parser.add_argument('--format', type=str, default='all',
                        choices=['all', 'text', 'compact', 'grid'],
                        help='输出格式')
    
    args = parser.parse_args()
    
    # 创建邻接矩阵
    binary_adj = create_text_adjacency_matrix(
        model_path=args.model_path,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk
    )
    
    # 根据选择生成不同格式
    if args.format == 'all' or args.format == 'text':
        print_text_adjacency_matrix(binary_adj)
    
    if args.format == 'all' or args.format == 'compact':
        print_compact_matrix(binary_adj)
    
    if args.format == 'all' or args.format == 'grid':
        create_visual_grid(binary_adj)
    
    print("\n" + "="*80)
    print("✓ 所有格式生成完成!")
    print("="*80)
    print("\n生成的文件:")
    if args.format == 'all':
        print("   - adjacency_matrix_text.txt (1和空格格式)")
        print("   - adjacency_matrix_compact.txt (紧凑列表格式)")
        print("   - adjacency_matrix_grid.txt (网格符号格式)")
    else:
        print(f"   - adjacency_matrix_{args.format}.txt")
    print("="*80)
