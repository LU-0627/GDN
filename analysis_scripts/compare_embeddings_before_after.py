# -*- coding: utf-8 -*-
"""
对比训练前后的嵌入向量
展示模型学习的效果
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from models.GDN import GDN
from util.env import get_device

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("对比训练前后的嵌入向量")
print("="*80)

# 参数设置
MODEL_PATH = 'pretrained/msl/best_01_07-154250.pt'
NODE_NUM = 27
DIM = 64
TOPK = 20

device = get_device()

# 1. 创建未训练的模型(随机初始化)
print("\n📂 创建随机初始化的模型(训练前)...")
edge_index = torch.zeros((2, NODE_NUM * TOPK), dtype=torch.long)
model_before = GDN(
    edge_index_sets=[edge_index],
    node_num=NODE_NUM,
    dim=DIM,
    input_dim=15,
    topk=TOPK
).to(device)

model_before.eval()
with torch.no_grad():
    embeddings_before = model_before.embedding.weight.cpu().numpy()

print(f"✓ 训练前嵌入向量: {embeddings_before.shape}")

# 2. 加载训练后的模型
print("\n📂 加载训练后的模型...")
model_after = GDN(
    edge_index_sets=[edge_index],
    node_num=NODE_NUM,
    dim=DIM,
    input_dim=15,
    topk=TOPK
).to(device)

model_after.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_after.eval()

with torch.no_grad():
    embeddings_after = model_after.embedding.weight.cpu().numpy()

print(f"✓ 训练后嵌入向量: {embeddings_after.shape}")

# 3. 统计对比
print("\n" + "="*80)
print("统计对比")
print("="*80)

print("\n【训练前 - 随机初始化】")
print(f"  最小值: {embeddings_before.min():.6f}")
print(f"  最大值: {embeddings_before.max():.6f}")
print(f"  平均值: {embeddings_before.mean():.6f}")
print(f"  标准差: {embeddings_before.std():.6f}")

norms_before = np.linalg.norm(embeddings_before, axis=1)
print(f"  平均L2范数: {norms_before.mean():.6f}")

print("\n【训练后 - 学习得到】")
print(f"  最小值: {embeddings_after.min():.6f}")
print(f"  最大值: {embeddings_after.max():.6f}")
print(f"  平均值: {embeddings_after.mean():.6f}")
print(f"  标准差: {embeddings_after.std():.6f}")

norms_after = np.linalg.norm(embeddings_after, axis=1)
print(f"  平均L2范数: {norms_after.mean():.6f}")

# 4. 相似度对比
print("\n" + "="*80)
print("节点相似度结构对比")
print("="*80)

cos_sim_before = cosine_similarity(embeddings_before)
cos_sim_after = cosine_similarity(embeddings_after)

# 排除对角线
np.fill_diagonal(cos_sim_before, 0)
np.fill_diagonal(cos_sim_after, 0)

print("\n【训练前】节点间余弦相似度:")
print(f"  平均相似度: {cos_sim_before.mean():.6f}")
print(f"  最大相似度: {cos_sim_before.max():.6f}")
print(f"  最小相似度: {cos_sim_before.min():.6f}")
print(f"  标准差: {cos_sim_before.std():.6f}")

print("\n【训练后】节点间余弦相似度:")
print(f"  平均相似度: {cos_sim_after.mean():.6f}")
print(f"  最大相似度: {cos_sim_after.max():.6f}")
print(f"  最小相似度: {cos_sim_after.min():.6f}")
print(f"  标准差: {cos_sim_after.std():.6f}")

print(f"\n💡 相似度标准差变化: {cos_sim_before.std():.6f} → {cos_sim_after.std():.6f}")
if cos_sim_after.std() > cos_sim_before.std():
    print("   ✓ 训练后节点间的差异性增强,模型学习到了更明确的节点关系!")
else:
    print("   相似度分布变化较小")

# 5. 可视化对比
print("\n🎨 生成对比可视化...")

fig = plt.figure(figsize=(18, 7))

# 训练前 - PCA
pca_before = PCA(n_components=2)
emb_2d_before = pca_before.fit_transform(embeddings_before)

ax1 = plt.subplot(1, 3, 1)
scatter1 = ax1.scatter(emb_2d_before[:, 0], emb_2d_before[:, 1],
                       s=150, alpha=0.7, c=range(NODE_NUM),
                       cmap='tab20', edgecolors='black', linewidths=1.5)
for i in range(NODE_NUM):
    ax1.annotate(f'{i}', (emb_2d_before[i, 0], emb_2d_before[i, 1]),
                fontsize=9, ha='center', va='center', fontweight='bold')
ax1.set_title('训练前 (随机初始化)\n节点分布较均匀', fontsize=12, fontweight='bold')
ax1.set_xlabel('PCA维度1', fontsize=10)
ax1.set_ylabel('PCA维度2', fontsize=10)
ax1.grid(True, alpha=0.3)

# 训练后 - PCA
pca_after = PCA(n_components=2)
emb_2d_after = pca_after.fit_transform(embeddings_after)

ax2 = plt.subplot(1, 3, 2)
scatter2 = ax2.scatter(emb_2d_after[:, 0], emb_2d_after[:, 1],
                       s=150, alpha=0.7, c=range(NODE_NUM),
                       cmap='tab20', edgecolors='black', linewidths=1.5)
for i in range(NODE_NUM):
    ax2.annotate(f'{i}', (emb_2d_after[i, 0], emb_2d_after[i, 1]),
                fontsize=9, ha='center', va='center', fontweight='bold')
ax2.set_title('训练后 (学习得到)\n节点形成明显的聚类', fontsize=12, fontweight='bold', color='green')
ax2.set_xlabel('PCA维度1', fontsize=10)
ax2.set_ylabel('PCA维度2', fontsize=10)
ax2.grid(True, alpha=0.3)

# 相似度矩阵对比
ax3 = plt.subplot(1, 3, 3)
diff = cos_sim_after - cos_sim_before
im = ax3.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax3.set_title('相似度变化\n(训练后 - 训练前)', fontsize=12, fontweight='bold')
ax3.set_xlabel('节点', fontsize=10)
ax3.set_ylabel('节点', fontsize=10)
plt.colorbar(im, ax=ax3, label='相似度变化')

plt.tight_layout()
plt.savefig('embeddings_before_after_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 对比图已保存: embeddings_before_after_comparison.png")

# 6. 找出学习到的最强关系
print("\n" + "="*80)
print("模型学习到的最强节点关系(训练后)")
print("="*80)

# 恢复对角线为0
np.fill_diagonal(cos_sim_after, 0)

# 找出Top10最相似的节点对
triu_indices = np.triu_indices(NODE_NUM, k=1)
similarities = cos_sim_after[triu_indices]
top_indices = np.argsort(similarities)[::-1][:10]

print("\n前10个最相似的节点对:")
for rank, idx in enumerate(top_indices):
    i, j = triu_indices[0][idx], triu_indices[1][idx]
    sim_after = cos_sim_after[i, j]
    sim_before = cosine_similarity(embeddings_before[i:i+1], embeddings_before[j:j+1])[0, 0]
    change = sim_after - sim_before
    
    print(f"{rank+1:2d}. 节点{i:2d} ↔ 节点{j:2d}  |  "
          f"相似度: {sim_after:7.4f}  |  变化: {change:+.4f}")

print("\n" + "="*80)
print("✓ 分析完成!")
print("="*80)
print("\n💡 总结:")
print("   - 训练前的嵌入是随机初始化的,节点间关系不明确")
print("   - 训练后的嵌入是模型学习到的,反映了真实的节点相关性")
print("   - 可视化图中靠近的节点=模型认为它们在异常检测任务中相关")
print("="*80)
