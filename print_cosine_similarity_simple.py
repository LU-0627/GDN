"""
简单版本: 在代码中直接调用的余弦相似度打印函数
可以在训练过程中或训练后使用
"""

import torch
import numpy as np


def print_cosine_similarity_simple(model, top_n=5):
    """
    打印模型的节点嵌入余弦相似度矩阵(简化版)
    
    Args:
        model: GDN模型实例(必须已经训练或加载了权重)
        top_n: 打印前N个节点的子矩阵
        
    示例用法:
        # 在训练后
        from print_cosine_similarity_simple import print_cosine_similarity_simple
        print_cosine_similarity_simple(model)
        
        # 或者加载模型后
        model = GDN(...)
        model.load_state_dict(torch.load('checkpoints/swat_best.pt'))
        print_cosine_similarity_simple(model)
    """
    model.eval()  # 设置为评估模式
    
    # 获取节点嵌入 (node_num个传感器, embed_dim维)
    with torch.no_grad():
        embeddings = model.embedding.weight  # [node_num, embed_dim]
        node_num = embeddings.shape[0]
        
        # 计算余弦相似度矩阵
        weights = embeddings.view(node_num, -1)
        cos_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_similarity = cos_mat / normed_mat  # [node_num, node_num]
        
        # 打印基本信息
        print("\n" + "="*80)
        print("余弦相似度矩阵:")
        print("="*80)
        print(f"Shape: {cos_similarity.shape}")
        print(f"Min: {cos_similarity.min().item():.4f}")
        print(f"Max: {cos_similarity.max().item():.4f}")
        print(f"Mean: {cos_similarity.mean().item():.4f}")
        
        # 打印对角线(应该都是1.0)
        diag_values = cos_similarity.diag()
        print(f"\n对角线 (自己与自己):")
        print(f"  Min: {diag_values.min().item():.4f}")
        print(f"  Max: {diag_values.max().item():.4f}")
        print(f"  示例: {diag_values[:5].cpu().numpy()}")  # 打印前5个
        
        # 打印前N个传感器的相似度矩阵
        print(f"\n前{top_n}个传感器的相似度矩阵:")
        print("-"*80)
        submatrix = cos_similarity[:top_n, :top_n].cpu().numpy()
        
        # 格式化打印
        header = "      " + "  ".join([f"Node{i:2d}" for i in range(top_n)])
        print(header)
        for i in range(top_n):
            row_str = f"Node{i:2d} " + "  ".join([f"{submatrix[i, j]:7.4f}" for j in range(top_n)])
            print(row_str)
        
        # 打印完整矩阵(如果节点数不多)
        if node_num <= 15:
            print(f"\n完整余弦相似度矩阵 ({node_num}x{node_num}):")
            print("-"*80)
            full_matrix = cos_similarity.cpu().numpy()
            header = "      " + "  ".join([f"N{i:2d}" for i in range(node_num)])
            print(header)
            for i in range(node_num):
                row_str = f"N{i:2d} " + "  ".join([f"{full_matrix[i, j]:6.3f}" for j in range(node_num)])
                print(row_str)
        
        print("="*80 + "\n")
        
        return cos_similarity


def save_cosine_similarity_to_file(model, filepath='cosine_similarity.txt'):
    """
    将余弦相似度矩阵保存到文件
    
    Args:
        model: GDN模型实例
        filepath: 保存文件路径
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight
        node_num = embeddings.shape[0]
        
        # 计算余弦相似度矩阵
        weights = embeddings.view(node_num, -1)
        cos_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_similarity = cos_mat / normed_mat
        
        # 保存到文件
        cos_np = cos_similarity.cpu().numpy()
        np.savetxt(filepath, cos_np, fmt='%.6f', delimiter=',')
        print(f"✓ 余弦相似度矩阵已保存到: {filepath}")
        
        return cos_similarity


# 示例: 直接使用的版本(不需要命令行参数)
if __name__ == '__main__':
    """
    如果你想直接运行这个脚本,请修改下面的参数
    """
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(__file__))
    
    from models.GDN import GDN
    from util.env import get_device
    
    # === 修改这里的参数 ===
    MODEL_PATH = 'pretrained/msl/best_01|07-15:42:50.pt'  # 使用通配符
    NODE_NUM = 27
    DIM = 64
# =====================

    INPUT_DIM = 15  # 输入特征维度
    TOPK = 20  # TopK参数
    # =====================
    
    print("加载模型...")
    device = get_device()
    
    # 创建简单的边索引
    edge_index = torch.zeros((2, NODE_NUM * TOPK), dtype=torch.long)
    edge_index_sets = [edge_index]
    
    # 初始化模型
    model = GDN(
        edge_index_sets=edge_index_sets,
        node_num=NODE_NUM,
        dim=DIM,
        input_dim=INPUT_DIM,
        topk=TOPK
    ).to(device)
    
    # 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"✓ 成功加载模型: {MODEL_PATH}\n")
        
        # 打印余弦相似度矩阵
        cos_sim = print_cosine_similarity_simple(model, top_n=5)
        
        # 保存到文件(可选)
        save_cosine_similarity_to_file(model, 'cosine_similarity.csv')
        
    else:
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请修改脚本中的 MODEL_PATH 变量")

