# -*- coding: utf-8 -*-
"""
快速可视化嵌入向量
使用PCA和t-SNE降维,生成2D可视化图
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("嵌入向量降维可视化")
print("="*80)

# 加载嵌入向量
print("\n📂 加载嵌入向量...")
embeddings = np.load('embeddings.npy')
print(f"✓ 已加载形状: {embeddings.shape}")

# 1. PCA降维
print("\n🔄 执行PCA降维...")
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)
print(f"✓ PCA完成 - 解释方差比例: {pca.explained_variance_ratio_.sum():.2%}")

# 2. t-SNE降维
print("\n🔄 执行t-SNE降维(可能需要几秒钟)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
embeddings_2d_tsne = tsne.fit_transform(embeddings)
print("✓ t-SNE完成")

# 创建可视化
print("\n🎨 生成可视化图表...")
fig = plt.figure(figsize=(16, 7))

# 子图1: PCA
ax1 = plt.subplot(1, 2, 1)
scatter1 = ax1.scatter(embeddings_2d_pca[:, 0], embeddings_2d_pca[:, 1], 
                       s=150, alpha=0.7, c=range(len(embeddings)), 
                       cmap='tab20', edgecolors='black', linewidths=1.5)

# 添加节点标签
for i in range(len(embeddings)):
    ax1.annotate(f'{i}', 
                (embeddings_2d_pca[i, 0], embeddings_2d_pca[i, 1]),
                fontsize=9, ha='center', va='center', fontweight='bold')

ax1.set_title(f'PCA降维可视化 ({len(embeddings)}个节点)\n解释方差: {pca.explained_variance_ratio_.sum():.1%}', 
             fontsize=13, fontweight='bold')
ax1.set_xlabel(f'第1主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax1.set_ylabel(f'第2主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax1.grid(True, alpha=0.3, linestyle='--')

# 子图2: t-SNE
ax2 = plt.subplot(1, 2, 2)
scatter2 = ax2.scatter(embeddings_2d_tsne[:, 0], embeddings_2d_tsne[:, 1],
                       s=150, alpha=0.7, c=range(len(embeddings)), 
                       cmap='tab20', edgecolors='black', linewidths=1.5)

# 添加节点标签
for i in range(len(embeddings)):
    ax2.annotate(f'{i}',
                (embeddings_2d_tsne[i, 0], embeddings_2d_tsne[i, 1]),
                fontsize=9, ha='center', va='center', fontweight='bold')

ax2.set_title(f't-SNE降维可视化 ({len(embeddings)}个节点)', 
             fontsize=13, fontweight='bold')
ax2.set_xlabel('维度1', fontsize=11)
ax2.set_ylabel('维度2', fontsize=11)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('embeddings_2d_visualization.png', dpi=300, bbox_inches='tight')
print("✓ 可视化已保存: embeddings_2d_visualization.png")

# 打印一些有用的信息
print("\n📊 可视化结果分析:")
print("-" * 80)

# 找出在2D空间中距离最近的节点对
from scipy.spatial.distance import pdist, squareform

dist_pca = squareform(pdist(embeddings_2d_pca))
np.fill_diagonal(dist_pca, np.inf)

print("\nPCA空间中最接近的5个节点对:")
for i in range(5):
    min_idx = np.unravel_index(np.argmin(dist_pca), dist_pca.shape)
    node1, node2 = min_idx
    dist = dist_pca[node1, node2]
    print(f"  {i+1}. 节点{node1:2d} ↔ 节点{node2:2d}  |  距离: {dist:.4f}")
    dist_pca[node1, node2] = np.inf
    dist_pca[node2, node1] = np.inf

print("\n" + "="*80)
print("✓ 完成!请查看生成的图片: embeddings_2d_visualization.png")
print("="*80)
