# -*- coding: utf-8 -*-
"""
可视化GDN的加权聚合邻居特征过程
展示如何使用Top-K邻居的相似度作为权重来聚合特征
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


def demonstrate_weighted_aggregation(model_path, target_node=2, node_num=27, 
                                     dim=64, input_dim=15, topk=20):
    """
    演示加权聚合邻居特征的过程
    
    Args:
        model_path: 模型路径
        target_node: 演示的目标节点
        node_num: 节点数量
        dim: 嵌入维度
        input_dim: 输入维度
        topk: K值
    """
    device = get_device()
    
    print("="*80)
    print(f"演示节点{target_node}的加权聚合邻居特征过程")
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
        # 1. 获取嵌入向量
        embeddings = model.embedding.weight  # [27, 64]
        
        # 2. 计算余弦相似度矩阵
        weights = embeddings.view(node_num, -1)
        cos_ji_mat = torch.matmul(weights, weights.T)
        normed_mat = torch.matmul(
            weights.norm(dim=-1).view(-1, 1),
            weights.norm(dim=-1).view(1, -1)
        )
        cos_ji_mat = cos_ji_mat / normed_mat  # [27, 27]
        
        # 3. Top-K选择
        topk_values, topk_indices = torch.topk(cos_ji_mat, k=topk, dim=-1)
        # topk_indices[i]: 节点i的Top-K邻居索引
        # topk_values[i]: 对应的相似度值(即权重)
        
        print(f"\n✓ 模型加载完成")
        print(f"  - 嵌入向量: {embeddings.shape}")
        print(f"  - 余弦相似度矩阵: {cos_ji_mat.shape}")
        print(f"  - Top-K索引: {topk_indices.shape}")
        print(f"  - Top-K权重: {topk_values.shape}")
        
        # 转换为numpy便于展示(在with块内完成)
        embeddings_np = embeddings.detach().cpu().numpy()
        topk_indices_np = topk_indices.detach().cpu().numpy()
        topk_values_np = topk_values.detach().cpu().numpy()
    
    # 获取目标节点的信息
    target_neighbors = topk_indices_np[target_node]  # [20]
    target_weights = topk_values_np[target_node]     # [20]
    
    print("\n" + "="*80)
    print(f"节点{target_node}的加权聚合详情")
    print("="*80)
    
    print(f"\n🎯 步骤1: 选择Top-{topk}邻居")
    print(f"   节点{target_node}的Top-{topk}邻居: {target_neighbors.tolist()}")
    
    print(f"\n📊 步骤2: 获取邻居的相似度权重")
    print(f"   {'邻居':<6} {'相似度权重':<15} {'归一化前':<15} {'权重可视化'}")
    print(f"   {'-'*70}")
    
    for i, (neighbor, weight) in enumerate(zip(target_neighbors, target_weights)):
        if neighbor == target_node:
            marker = "(自己)"
        else:
            marker = ""
        
        bar_length = int(abs(weight) * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        
        print(f"   {neighbor:<6} {weight:<15.6f} {weight:<15.6f} {bar} {marker}")
    
    # 步骤3: 归一化权重(如果需要)
    print(f"\n⚖️ 步骤3: 归一化权重(可选)")
    print(f"   原始权重和: {target_weights.sum():.6f}")
    
    # 使用softmax归一化
    normalized_weights = np.exp(target_weights) / np.exp(target_weights).sum()
    print(f"   Softmax归一化后的权重和: {normalized_weights.sum():.6f}")
    
    print(f"\n   归一化对比:")
    print(f"   {'邻居':<6} {'原始权重':<15} {'Softmax权重':<15} {'百分比'}")
    print(f"   {'-'*60}")
    
    for neighbor, orig_w, norm_w in zip(target_neighbors[:10], 
                                        target_weights[:10], 
                                        normalized_weights[:10]):
        percentage = norm_w * 100
        print(f"   {neighbor:<6} {orig_w:<15.6f} {norm_w:<15.6f} {percentage:6.2f}%")
    
    print(f"   ...")
    
    # 步骤4: 模拟特征聚合
    print(f"\n🔄 步骤4: 加权聚合邻居特征")
    print(f"\n   假设我们有输入特征 X ∈ R^{{27×{input_dim}}} (每个节点{input_dim}维特征)")
    
    # 创建模拟的输入特征
    np.random.seed(42)
    X = np.random.randn(node_num, input_dim)
    
    print(f"\n   聚合公式:")
    print(f"   h_{target_node} = Σ w_{{j}} × X_{{j}}  (j是节点{target_node}的Top-K邻居)")
    print(f"         = w_{{neighbor1}} × X_{{neighbor1}} + w_{{neighbor2}} × X_{{neighbor2}} + ...")
    
    # 执行聚合
    aggregated_feature = np.zeros(input_dim)
    
    print(f"\n   详细计算过程(前5个邻居):")
    for i, (neighbor, weight) in enumerate(zip(target_neighbors[:5], 
                                               normalized_weights[:5])):
        neighbor_feature = X[neighbor]
        contribution = weight * neighbor_feature
        aggregated_feature += contribution
        
        print(f"\n   邻居{neighbor} (权重={weight:.4f}):")
        print(f"     特征向量X_{neighbor}[:5] = [{neighbor_feature[:5]}]")
        print(f"     贡献 = {weight:.4f} × X_{neighbor}")
        print(f"     贡献[:5] = [{contribution[:5]}]")
    
    # 完整聚合
    for neighbor, weight in zip(target_neighbors[5:], normalized_weights[5:]):
        aggregated_feature += weight * X[neighbor]
    
    print(f"\n   ...")
    print(f"\n   聚合后的特征 h_{target_node}[:5] = [{aggregated_feature[:5]}]")
    print(f"   聚合特征维度: {aggregated_feature.shape}")
    
    # 可视化
    visualize_aggregation_process(target_node, target_neighbors, normalized_weights, 
                                   X, aggregated_feature)
    
    return target_neighbors, normalized_weights, X, aggregated_feature


def visualize_aggregation_process(target_node, neighbors, weights, features, 
                                   aggregated_feature):
    """
    可视化聚合过程
    """
    print(f"\n🎨 生成可视化...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # 子图1: 权重分布
    ax1 = plt.subplot(2, 3, 1)
    colors = ['red' if n == target_node else 'skyblue' for n in neighbors]
    bars = ax1.bar(range(len(weights)), weights, color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('邻居索引(按相似度排序)', fontsize=11)
    ax1.set_ylabel('归一化权重', fontsize=11)
    ax1.set_title(f'节点{target_node}的Top-K邻居权重分布', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 标注节点ID
    for i, (neighbor, weight) in enumerate(zip(neighbors, weights)):
        if i % 2 == 0 or weight > 0.1:  # 只标注部分以免拥挤
            ax1.text(i, weight, f'{neighbor}', ha='center', va='bottom', fontsize=8)
    
    # 子图2: 权重饼图(Top-10)
    ax2 = plt.subplot(2, 3, 2)
    top_n = 10
    top_weights = weights[:top_n]
    top_neighbors = neighbors[:top_n]
    other_weight = weights[top_n:].sum()
    
    pie_weights = list(top_weights) + [other_weight]
    pie_labels = [f'N{n}' for n in top_neighbors] + ['其他']
    
    colors_pie = ['red' if n == target_node else plt.cm.Set3(i) 
                  for i, n in enumerate(list(top_neighbors) + [-1])]
    
    ax2.pie(pie_weights, labels=pie_labels, autopct='%1.1f%%', startangle=90,
            colors=colors_pie)
    ax2.set_title(f'前{top_n}个邻居的权重占比', fontsize=12, fontweight='bold')
    
    # 子图3: 邻居特征热力图
    ax3 = plt.subplot(2, 3, 3)
    neighbor_features = features[neighbors[:15], :10]  # 前15个邻居的前10维
    
    im = ax3.imshow(neighbor_features, cmap='RdBu_r', aspect='auto')
    ax3.set_xlabel('特征维度', fontsize=11)
    ax3.set_ylabel('邻居节点', fontsize=11)
    ax3.set_title('邻居特征矩阵(前15邻居,前10维)', fontsize=12, fontweight='bold')
    ax3.set_yticks(range(min(15, len(neighbors))))
    ax3.set_yticklabels([f'N{n}' for n in neighbors[:15]], fontsize=8)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    
    # 子图4: 加权特征贡献
    ax4 = plt.subplot(2, 3, 4)
    weighted_contributions = []
    for i, (neighbor, weight) in enumerate(zip(neighbors[:15], weights[:15])):
        contribution = weight * features[neighbor, :10]
        weighted_contributions.append(contribution)
    
    weighted_contributions = np.array(weighted_contributions)
    
    im2 = ax4.imshow(weighted_contributions, cmap='RdBu_r', aspect='auto')
    ax4.set_xlabel('特征维度', fontsize=11)
    ax4.set_ylabel('邻居节点', fontsize=11)
    ax4.set_title('加权后的特征贡献(前15,前10维)', fontsize=12, fontweight='bold')
    ax4.set_yticks(range(min(15, len(neighbors))))
    ax4.set_yticklabels([f'N{n}({weights[i]:.2f})' for i, n in enumerate(neighbors[:15])], 
                        fontsize=8)
    plt.colorbar(im2, ax=ax4, fraction=0.046, pad=0.04)
    
    # 子图5: 聚合特征对比
    ax5 = plt.subplot(2, 3, 5)
    original_feature = features[target_node, :10]
    aggregated_part = aggregated_feature[:10]
    
    x = np.arange(10)
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, original_feature, width, label=f'原始特征(节点{target_node})',
                    alpha=0.7, color='orange', edgecolor='black')
    bars2 = ax5.bar(x + width/2, aggregated_part, width, label='聚合后特征',
                    alpha=0.7, color='green', edgecolor='black')
    
    ax5.set_xlabel('特征维度', fontsize=11)
    ax5.set_ylabel('特征值', fontsize=11)
    ax5.set_title('原始特征 vs 聚合特征(前10维)', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 子图6: 聚合流程图
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 绘制流程图文本(移除所有特殊符号)
    flow_text = f"""加权聚合流程总结
    
[1] 选择Top-K邻居
   节点{target_node} -> Top-{len(neighbors)}邻居
   
[2] 计算相似度权重
   基于嵌入向量的余弦相似度
   
[3] 归一化权重
   使用Softmax: 权重和 = {weights.sum():.4f} ≈ 1.0
   
[4] 加权聚合特征
   h_{target_node} = Σ w_j × X_j
   
[5] 输出聚合特征
   用于后续的图卷积层

关键参数:
- 邻居数K = {len(neighbors)}
- 特征维度 = {features.shape[1]}
- 权重范围 = [{weights.min():.4f}, {weights.max():.4f}]
- 最大贡献邻居 = 节点{neighbors[weights.argmax()]}
"""
    
    ax6.text(0.1, 0.5, flow_text, fontsize=10, 
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'node_{target_node}_weighted_aggregation.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    print(f"✓ 可视化已保存: node_{target_node}_weighted_aggregation.png")
    
    plt.close()


def save_aggregation_details(target_node, neighbors, weights, features, 
                             aggregated_feature, filepath='aggregation_details.txt'):
    """
    保存聚合详情到文本文件
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"节点{target_node}的加权聚合邻居特征详细过程\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Top-K邻居选择\n")
        f.write("-" * 80 + "\n")
        f.write(f"邻居列表: {neighbors.tolist()}\n\n")
        
        f.write("2. 相似度权重\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'邻居':<8} {'权重':<15} {'百分比':<10} {'累积百分比'}\n")
        f.write("-" * 80 + "\n")
        
        cumsum = 0
        for neighbor, weight in zip(neighbors, weights):
            percentage = weight * 100
            cumsum += percentage
            f.write(f"{neighbor:<8} {weight:<15.6f} {percentage:<10.2f}% {cumsum:>6.2f}%\n")
        
        f.write(f"\n权重和: {weights.sum():.6f}\n\n")
        
        f.write("3. 聚合公式\n")
        f.write("-" * 80 + "\n")
        f.write(f"h_{target_node} = ")
        formula_parts = [f"w_{n}·X_{n}" for n in neighbors[:5]]
        f.write(" + ".join(formula_parts) + " + ...\n\n")
        
        f.write("4. 聚合结果\n")
        f.write("-" * 80 + "\n")
        f.write(f"聚合特征维度: {aggregated_feature.shape}\n")
        f.write(f"聚合特征统计:\n")
        f.write(f"  - 最小值: {aggregated_feature.min():.6f}\n")
        f.write(f"  - 最大值: {aggregated_feature.max():.6f}\n")
        f.write(f"  - 平均值: {aggregated_feature.mean():.6f}\n")
        f.write(f"  - 标准差: {aggregated_feature.std():.6f}\n")
    
    print(f"✓ 聚合详情已保存: {filepath}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='演示GDN的加权聚合邻居特征')
    parser.add_argument('--node', type=int, default=2, help='目标节点')
    parser.add_argument('--model_path', type=str,
                        default='pretrained/msl/best_01_07-154250.pt',
                        help='模型路径')
    parser.add_argument('--node_num', type=int, default=27, help='节点数量')
    parser.add_argument('--dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--input_dim', type=int, default=15, help='输入维度')
    parser.add_argument('--topk', type=int, default=20, help='K值')
    
    args = parser.parse_args()
    
    neighbors, weights, features, aggregated = demonstrate_weighted_aggregation(
        model_path=args.model_path,
        target_node=args.node,
        node_num=args.node_num,
        dim=args.dim,
        input_dim=args.input_dim,
        topk=args.topk
    )
    
    save_aggregation_details(args.node, neighbors, weights, features, aggregated)
    
    print("\n" + "="*80)
    print("✓ 所有演示完成!")
    print("="*80)
