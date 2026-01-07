"""
直接查看GDN模型的原始嵌入向量
支持打印、保存、可视化等多种方式
"""

import torch
import numpy as np
import os
import argparse
from models.GDN import GDN
from util.env import get_device


def load_model(model_path, node_num=27, dim=64, input_dim=15, topk=20):
    """
    加载已训练的GDN模型
    
    Args:
        model_path: 模型权重文件路径
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入特征维度
        topk: TopK参数
        
    Returns:
        model: 加载了权重的模型
    """
    device = get_device()
    
    # 创建简单的边索引
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


def view_embeddings_basic(model):
    """
    基本方式:打印嵌入向量的统计信息
    
    Args:
        model: GDN模型实例
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight  # [节点数, 嵌入维度]
        
        print("\n" + "="*80)
        print("📊 嵌入向量基本信息")
        print("="*80)
        
        print(f"\n形状: {embeddings.shape}")
        print(f"  - 节点数量: {embeddings.shape[0]}")
        print(f"  - 嵌入维度: {embeddings.shape[1]}")
        print(f"  - 总参数量: {embeddings.numel()}")
        
        print(f"\n数值统计:")
        print(f"  - 最小值: {embeddings.min().item():.6f}")
        print(f"  - 最大值: {embeddings.max().item():.6f}")
        print(f"  - 平均值: {embeddings.mean().item():.6f}")
        print(f"  - 标准差: {embeddings.std().item():.6f}")
        
        # 计算每个节点的L2范数
        norms = embeddings.norm(dim=1)
        print(f"\nL2范数统计:")
        print(f"  - 最小L2范数: {norms.min().item():.6f}")
        print(f"  - 最大L2范数: {norms.max().item():.6f}")
        print(f"  - 平均L2范数: {norms.mean().item():.6f}")
        
        print("="*80)


def view_embeddings_detailed(model, num_nodes=5, num_dims=10):
    """
    详细方式:打印具体的嵌入向量值
    
    Args:
        model: GDN模型实例
        num_nodes: 显示前N个节点
        num_dims: 显示前M个维度
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
        
        print("\n" + "="*80)
        print(f"📋 前{num_nodes}个节点的嵌入向量(前{num_dims}维)")
        print("="*80)
        
        for i in range(min(num_nodes, embeddings.shape[0])):
            print(f"\nNode {i:2d}:")
            print(f"  完整向量形状: {embeddings[i].shape}")
            print(f"  前{num_dims}维: ", end="")
            
            # 打印前num_dims个值
            dims_to_show = min(num_dims, embeddings.shape[1])
            for j in range(dims_to_show):
                print(f"{embeddings[i, j]:8.4f}", end=" ")
            
            if embeddings.shape[1] > num_dims:
                print("...")
            else:
                print()
            
            print(f"  L2范数: {np.linalg.norm(embeddings[i]):.6f}")
        
        print("="*80)


def view_embeddings_full_matrix(model, num_nodes=None):
    """
    完整矩阵方式:打印完整的嵌入矩阵
    
    Args:
        model: GDN模型实例
        num_nodes: 显示前N个节点,None表示全部
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
        
        if num_nodes is None:
            num_nodes = embeddings.shape[0]
        
        print("\n" + "="*80)
        print(f"📊 完整嵌入矩阵 (前{num_nodes}个节点)")
        print("="*80)
        
        # 设置numpy打印选项
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        
        print(f"\n形状: [{num_nodes}, {embeddings.shape[1]}]")
        print("\n嵌入矩阵:")
        print(embeddings[:num_nodes])
        
        # 恢复默认打印选项
        np.set_printoptions()
        
        print("\n" + "="*80)


def save_embeddings(model, save_path='embeddings.npy', format='npy'):
    """
    保存嵌入向量到文件
    
    Args:
        model: GDN模型实例
        save_path: 保存路径
        format: 保存格式 ('npy', 'txt', 'csv')
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
        
        if format == 'npy':
            np.save(save_path, embeddings)
            print(f"✓ 嵌入向量已保存为numpy格式: {save_path}")
            
        elif format == 'txt':
            np.savetxt(save_path, embeddings, fmt='%.6f', delimiter=' ')
            print(f"✓ 嵌入向量已保存为文本格式: {save_path}")
            
        elif format == 'csv':
            np.savetxt(save_path, embeddings, fmt='%.6f', delimiter=',')
            print(f"✓ 嵌入向量已保存为CSV格式: {save_path}")
            
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        print(f"  - 形状: {embeddings.shape}")
        print(f"  - 文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")


def visualize_embeddings_heatmap(model, save_path='embeddings_heatmap.png'):
    """
    可视化嵌入向量热力图
    
    Args:
        model: GDN模型实例
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("❌ 需要安装 matplotlib 和 seaborn 才能可视化")
        print("   运行: pip install matplotlib seaborn")
        return
    
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
        
        plt.figure(figsize=(14, 8))
        
        # 绘制热力图
        sns.heatmap(
            embeddings,
            cmap='RdBu_r',
            center=0,
            cbar_kws={"label": "嵌入值"},
            xticklabels=10,  # 每10个维度显示一个标签
            yticklabels=True
        )
        
        plt.title(f'节点嵌入向量热力图\n形状: {embeddings.shape}', fontsize=14)
        plt.xlabel('嵌入维度', fontsize=12)
        plt.ylabel('节点索引', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n💾 嵌入向量热力图已保存: {save_path}")
        
        plt.close()


def analyze_embedding_dimensions(model, top_k=10):
    """
    分析嵌入向量的重要维度
    
    Args:
        model: GDN模型实例
        top_k: 显示前K个最重要的维度
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu().numpy()
        
        print("\n" + "="*80)
        print("🔍 嵌入维度重要性分析")
        print("="*80)
        
        # 计算每个维度的方差(方差越大,该维度越重要)
        dim_variance = np.var(embeddings, axis=0)
        
        # 找到方差最大的维度
        top_dims = np.argsort(dim_variance)[::-1][:top_k]
        
        print(f"\n前{top_k}个最重要的维度(按方差排序):")
        print("-" * 80)
        for rank, dim in enumerate(top_dims):
            print(f"{rank+1:2d}. 维度 {dim:3d}  |  方差: {dim_variance[dim]:.6f}  |  "
                  f"范围: [{embeddings[:, dim].min():.4f}, {embeddings[:, dim].max():.4f}]")
        
        # 计算每个维度的平均绝对值
        dim_mean_abs = np.mean(np.abs(embeddings), axis=0)
        top_dims_abs = np.argsort(dim_mean_abs)[::-1][:top_k]
        
        print(f"\n前{top_k}个平均绝对值最大的维度:")
        print("-" * 80)
        for rank, dim in enumerate(top_dims_abs):
            print(f"{rank+1:2d}. 维度 {dim:3d}  |  平均绝对值: {dim_mean_abs[dim]:.6f}")
        
        print("="*80)


def compare_node_embeddings(model, node_ids):
    """
    比较特定节点的嵌入向量
    
    Args:
        model: GDN模型实例
        node_ids: 要比较的节点ID列表
    """
    model.eval()
    
    with torch.no_grad():
        embeddings = model.embedding.weight.cpu()
        
        print("\n" + "="*80)
        print(f"🔄 比较节点嵌入向量: {node_ids}")
        print("="*80)
        
        for i, node_id in enumerate(node_ids):
            if node_id >= embeddings.shape[0]:
                print(f"⚠️ 节点 {node_id} 超出范围(最大: {embeddings.shape[0]-1})")
                continue
            
            vec = embeddings[node_id]
            print(f"\nNode {node_id}:")
            print(f"  L2范数: {vec.norm().item():.6f}")
            print(f"  平均值: {vec.mean().item():.6f}")
            print(f"  标准差: {vec.std().item():.6f}")
        
        # 计算节点之间的余弦相似度
        if len(node_ids) >= 2:
            print("\n节点间的余弦相似度:")
            print("-" * 60)
            from torch.nn.functional import cosine_similarity
            
            for i in range(len(node_ids)):
                for j in range(i+1, len(node_ids)):
                    if node_ids[i] < embeddings.shape[0] and node_ids[j] < embeddings.shape[0]:
                        sim = cosine_similarity(
                            embeddings[node_ids[i]].unsqueeze(0),
                            embeddings[node_ids[j]].unsqueeze(0)
                        )
                        print(f"  Node {node_ids[i]:2d} ↔ Node {node_ids[j]:2d}: {sim.item():.6f}")
        
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='查看GDN模型的原始嵌入向量')
    parser.add_argument('--model_path', type=str, 
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型权重文件路径')
    parser.add_argument('--node_num', type=int, default=27,
                        help='节点数量(默认: 27)')
    parser.add_argument('--dim', type=int, default=64,
                        help='嵌入维度(默认: 64)')
    parser.add_argument('--input_dim', type=int, default=15,
                        help='输入特征维度(默认: 15)')
    parser.add_argument('--topk', type=int, default=20,
                        help='TopK参数(默认: 20)')
    
    # 显示选项
    parser.add_argument('--basic', action='store_true', default=True,
                        help='显示基本统计信息')
    parser.add_argument('--detailed', action='store_true',
                        help='显示详细的嵌入值')
    parser.add_argument('--full', action='store_true',
                        help='显示完整矩阵')
    parser.add_argument('--num_nodes', type=int, default=5,
                        help='详细模式下显示的节点数(默认: 5)')
    parser.add_argument('--num_dims', type=int, default=10,
                        help='详细模式下显示的维度数(默认: 10)')
    
    # 分析选项
    parser.add_argument('--analyze_dims', action='store_true',
                        help='分析维度重要性')
    parser.add_argument('--compare_nodes', type=int, nargs='+',
                        help='比较特定节点,例如: --compare_nodes 0 1 5')
    
    # 保存选项
    parser.add_argument('--save', action='store_true',
                        help='保存嵌入向量')
    parser.add_argument('--save_path', type=str, default='embeddings.npy',
                        help='保存路径(默认: embeddings.npy)')
    parser.add_argument('--format', type=str, default='npy',
                        choices=['npy', 'txt', 'csv'],
                        help='保存格式')
    
    # 可视化选项
    parser.add_argument('--visualize', action='store_true',
                        help='生成热力图')
    parser.add_argument('--fig_path', type=str, default='embeddings_heatmap.png',
                        help='热力图保存路径')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("GDN模型 - 原始嵌入向量查看器")
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
    
    # 基本信息
    if args.basic or not any([args.detailed, args.full, args.analyze_dims, args.compare_nodes]):
        view_embeddings_basic(model)
    
    # 详细信息
    if args.detailed:
        view_embeddings_detailed(model, args.num_nodes, args.num_dims)
    
    # 完整矩阵
    if args.full:
        view_embeddings_full_matrix(model, args.num_nodes)
    
    # 维度分析
    if args.analyze_dims:
        analyze_embedding_dimensions(model)
    
    # 节点比较
    if args.compare_nodes:
        compare_node_embeddings(model, args.compare_nodes)
    
    # 保存
    if args.save:
        save_embeddings(model, args.save_path, args.format)
    
    # 可视化
    if args.visualize:
        visualize_embeddings_heatmap(model, args.fig_path)
    
    print("\n✓ 查看完成!\n")


if __name__ == '__main__':
    main()
