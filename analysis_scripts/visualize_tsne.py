"""
使用t-SNE对GDN模型的嵌入向量进行降维可视化
支持多种可视化方式：按节点着色、按时间步着色、按异常标签着色
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import argparse
from pathlib import Path

from models.GDN import GDN
from util.env import get_device
from datasets.TimeDataset import TimeDataset
from torch.utils.data import DataLoader
import pandas as pd


class TSNEVisualizer:
    """t-SNE可视化工具类"""
    
    def __init__(self, model, device, perplexity=30, n_iter=1000, random_state=42):
        """
        初始化t-SNE可视化器
        
        Args:
            model: GDN模型
            device: 设备(cpu/cuda)
            perplexity: t-SNE困惑度参数
            n_iter: t-SNE迭代次数
            random_state: 随机种子
        """
        self.model = model
        self.device = device
        self.tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def extract_embeddings(self, data_loader):
        """
        从模型提取嵌入向量
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            embeddings: 提取的嵌入向量 [N, dim]
            labels: 样本标签 [N]
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_x, batch_y, batch_labels in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # 前向传播
                try:
                    # 尝试获取返回嵌入向量
                    result = self.model(batch_x, return_embeddings=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        out, hidden_features, embeddings_dict = result
                        # 使用隐层特征作为嵌入向量
                        feat = hidden_features.cpu().numpy()  # [batch, node_num, dim]
                    else:
                        raise ValueError("模型未返回嵌入向量")
                except TypeError:
                    # 如果模型不支持return_embeddings参数
                    out = self.model(batch_x)
                    # 直接从模型的嵌入层提取
                    node_num = batch_x.shape[1]
                    feat = self.model.embedding.weight.detach().cpu().numpy()  # [node_num, dim]
                    feat = np.tile(feat, (len(batch_labels), 1))  # 重复以匹配批次大小
                
                # 如果是3D张量，进行处理
                if len(feat.shape) == 3:
                    batch_size, node_num, dim = feat.shape
                    feat = feat.reshape(batch_size * node_num, dim)
                elif len(feat.shape) == 2:
                    # 已经是2D，直接使用
                    pass
                
                embeddings_list.append(feat)
                labels_list.extend(batch_labels.cpu().numpy())
        
        embeddings = np.vstack(embeddings_list) if embeddings_list else np.array([])
        labels = np.array(labels_list)
        
        return embeddings, labels
    
    def fit_transform(self, embeddings):
        """
        对嵌入向量进行标准化和t-SNE降维
        
        Args:
            embeddings: 原始嵌入向量 [N, dim]
        
        Returns:
            tsne_result: t-SNE降维结果 [N, 2]
        """
        print("正在进行数据标准化...")
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        print("正在进行t-SNE降维... 这可能需要几分钟")
        tsne_result = self.tsne.fit_transform(embeddings_scaled)
        
        return tsne_result
    
    def visualize_by_label(self, tsne_result, labels, title="t-SNE Visualization - Anomaly Label"):
        """
        按异常标签着色可视化
        
        Args:
            tsne_result: t-SNE降维结果 [N, 2]
            labels: 异常标签 [N]
            title: 图表标题
        """
        plt.figure(figsize=(12, 10))
        
        # 分离正常和异常样本
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        if np.sum(normal_mask) > 0:
            plt.scatter(tsne_result[normal_mask, 0], tsne_result[normal_mask, 1],
                       c='blue', label='Normal', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        if np.sum(anomaly_mask) > 0:
            plt.scatter(tsne_result[anomaly_mask, 0], tsne_result[anomaly_mask, 1],
                       c='red', label='Anomaly', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_by_density(self, tsne_result, title="t-SNE Visualization - Local Density"):
        """
        按局部密度着色可视化
        
        Args:
            tsne_result: t-SNE降维结果 [N, 2]
            title: 图表标题
        """
        # 计算每个点周围的密度（使用距离）
        from scipy.spatial.distance import pdist, squareform
        
        print("计算样本密度...")
        distances = squareform(pdist(tsne_result, metric='euclidean'))
        # 使用k-NN的平均距离作为密度的度量
        k = min(10, len(tsne_result) - 1)
        density = np.mean(np.partition(distances, k, axis=1)[:, :k], axis=1)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                             c=density, cmap='viridis', 
                             alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, label='Local Density')
        plt.xlabel('t-SNE Component 1', fontsize=12)
        plt.ylabel('t-SNE Component 2', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def visualize_distribution(self, embeddings, title="嵌入向量分布"):
        """
        可视化原始嵌入向量的分布
        
        Args:
            embeddings: 原始嵌入向量
            title: 标题
        """
        # 计算统计量
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 直方图
        axes[0, 0].hist(embeddings.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('嵌入向量分布直方图', fontsize=11)
        axes[0, 0].set_xlabel('值')
        axes[0, 0].set_ylabel('频率')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 按维度的平均值
        mean_vals = np.mean(embeddings, axis=0)
        axes[0, 1].plot(mean_vals, marker='o', linestyle='-', linewidth=2)
        axes[0, 1].set_title('各维度平均值', fontsize=11)
        axes[0, 1].set_xlabel('维度')
        axes[0, 1].set_ylabel('平均值')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 按维度的标准差
        std_vals = np.std(embeddings, axis=0)
        axes[1, 0].plot(std_vals, marker='s', linestyle='-', linewidth=2, color='orange')
        axes[1, 0].set_title('各维度标准差', fontsize=11)
        axes[1, 0].set_xlabel('维度')
        axes[1, 0].set_ylabel('标准差')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 相关矩阵热力图（采样）
        if embeddings.shape[1] > 1:
            sample_indices = np.random.choice(len(embeddings), min(500, len(embeddings)), replace=False)
            corr_matrix = np.corrcoef(embeddings[sample_indices].T)
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[1, 1].set_title('嵌入向量相关性（采样）', fontsize=11)
            plt.colorbar(im, ax=axes[1, 1], label='相关系数')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        return fig


def load_model(model_path, config, dataset_name='swat'):
    """
    加载已训练的GDN模型
    
    Args:
        model_path: 模型权重文件路径
        config: 配置字典
        dataset_name: 数据集名称
    
    Returns:
        model: 加载的模型
        device: 设备
    """
    device = get_device()
    
    # 从配置获取参数
    node_num = config.get('node_num', 51)
    dim = config.get('dim', 64)
    input_dim = config.get('input_dim', 51)
    topk = config.get('topk', 20)
    
    # 创建边索引
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
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ 成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    return model, device


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用t-SNE可视化GDN模型嵌入向量')
    
    parser.add_argument('--dataset', type=str, default='swat',
                       choices=['swat', 'msl'],
                       help='数据集名称')
    parser.add_argument('--model_path', type=str,
                       help='模型权重文件路径')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE困惑度参数')
    parser.add_argument('--n_iter', type=int, default=1000,
                       help='t-SNE迭代次数')
    parser.add_argument('--output_dir', type=str, default='./tsne_results',
                       help='输出目录')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='用于t-SNE的样本数（加速）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 配置
    config = {
        'swat': {'node_num': 51, 'dim': 64, 'input_dim': 51, 'topk': 20},
        'msl': {'node_num': 55, 'dim': 64, 'input_dim': 55, 'topk': 20},
    }
    dataset_config = config[args.dataset]
    
    # 自动寻找模型文件
    if not args.model_path:
        model_dir = Path(f'./pretrained/{args.dataset}')
        if model_dir.exists():
            model_files = sorted(list(model_dir.glob('*.pt')))
            if model_files:
                args.model_path = str(model_files[-1])  # 使用最新的模型
                print(f"自动选择模型: {args.model_path}")
    
    if not args.model_path:
        raise ValueError(f"未找到 {args.dataset} 的模型文件")
    
    # 加载模型
    print(f"\n{'='*60}")
    print(f"正在加载模型: {args.model_path}")
    print(f"{'='*60}")
    model, device = load_model(args.model_path, dataset_config, args.dataset)
    
    # 加载数据
    print(f"\n正在加载 {args.dataset} 数据集...")
    test_dataset = TimeDataset(
        root=f'./data/{args.dataset}',
        mode='test',
        config=dataset_config
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建可视化工具
    print("初始化t-SNE可视化工具...")
    visualizer = TSNEVisualizer(model, device, 
                               perplexity=args.perplexity,
                               n_iter=args.n_iter)
    
    # 提取嵌入向量
    print("\n提取嵌入向量...")
    embeddings, labels = visualizer.extract_embeddings(test_dataloader)
    
    print(f"✓ 嵌入向量形状: {embeddings.shape}")
    print(f"✓ 标签形状: {labels.shape}")
    print(f"  - 正常样本: {np.sum(labels == 0)}")
    print(f"  - 异常样本: {np.sum(labels == 1)}")
    
    # 可选：采样以加速t-SNE
    sample_size = args.sample_size or min(5000, len(embeddings))
    if len(embeddings) > sample_size:
        print(f"\n采样 {sample_size} 个样本以加速t-SNE计算...")
        sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings_sample = embeddings[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        embeddings_sample = embeddings
        labels_sample = labels
    
    # 执行t-SNE降维
    print("\n执行t-SNE降维...")
    tsne_result = visualizer.fit_transform(embeddings_sample)
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 1. 按异常标签着色
    fig1 = visualizer.visualize_by_label(
        tsne_result, labels_sample,
        title=f't-SNE Visualization - {args.dataset.upper()} Dataset'
    )
    fig1.savefig(output_dir / f'{args.dataset}_tsne_by_label.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_dir / f'{args.dataset}_tsne_by_label.png'}")
    plt.close(fig1)
    
    # 2. 按密度着色
    fig2 = visualizer.visualize_by_density(
        tsne_result,
        title=f't-SNE Visualization - {args.dataset.upper()} (Density)'
    )
    fig2.savefig(output_dir / f'{args.dataset}_tsne_by_density.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_dir / f'{args.dataset}_tsne_by_density.png'}")
    plt.close(fig2)
    
    # 3. 嵌入向量分布
    fig3 = visualizer.visualize_distribution(embeddings, '嵌入向量分布分析')
    fig3.savefig(output_dir / f'{args.dataset}_embedding_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ 已保存: {output_dir / f'{args.dataset}_embedding_distribution.png'}")
    plt.close(fig3)
    
    # 保存t-SNE结果到CSV
    results_df = pd.DataFrame({
        'tsne_1': tsne_result[:, 0],
        'tsne_2': tsne_result[:, 1],
        'label': labels_sample,
    })
    results_df.to_csv(output_dir / f'{args.dataset}_tsne_results.csv', index=False)
    print(f"✓ 已保存t-SNE结果: {output_dir / f'{args.dataset}_tsne_results.csv'}")
    
    # 保存嵌入向量到CSV（采样）
    embeddings_df = pd.DataFrame(embeddings_sample[:100])  # 保存前100个样本的完整嵌入
    embeddings_df.to_csv(output_dir / f'{args.dataset}_embeddings_sample.csv', index=False)
    print(f"✓ 已保存嵌入向量样本: {output_dir / f'{args.dataset}_embeddings_sample.csv'}")
    
    print(f"\n{'='*60}")
    print(f"可视化完成！所有结果保存到: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
