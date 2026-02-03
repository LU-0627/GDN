# GDN 模型 t-SNE 可视化工具

这个工具用于对训练好的GDN（图深度网络）模型的嵌入向量进行t-SNE降维可视化。

## 功能概述

- **t-SNE降维**：将高维嵌入向量降至2维用于可视化
- **多种着色方案**：
  - 按异常标签着色（区分正常/异常样本）
  - 按局部密度着色（识别聚类结构）
  - 嵌入向量分布分析
- **批量可视化**：自动生成多张图表进行对比分析
- **结果导出**：将t-SNE结果和嵌入向量导出为CSV

## 快速开始

### 方式1：使用批处理文件（Windows）

```bash
# 使用默认参数（SWAT数据集）
run_tsne.bat

# 指定数据集
run_tsne.bat msl

# 指定参数
run_tsne.bat swat 30 1000
```

### 方式2：使用Shell脚本（Linux/Mac）

```bash
# 使用默认参数
bash run_tsne.sh

# 指定数据集
bash run_tsne.sh msl
```

### 方式3：直接运行Python脚本

```bash
# 基础用法
python visualize_tsne.py --dataset swat

# 完整参数
python visualize_tsne.py \
    --dataset swat \
    --perplexity 30 \
    --n_iter 1000 \
    --output_dir ./tsne_results \
    --batch_size 256 \
    --sample_size 5000
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | swat | 数据集名称：`swat` 或 `msl` |
| `--model_path` | 自动检测 | 模型权重文件路径 |
| `--perplexity` | 30 | t-SNE困惑度参数（越小越快） |
| `--n_iter` | 1000 | t-SNE迭代次数（越多质量越好） |
| `--output_dir` | ./tsne_results | 输出目录 |
| `--batch_size` | 256 | 数据加载批次大小 |
| `--sample_size` | 自动 | 用于t-SNE的样本数（加速） |

## 参数调优建议

### 速度 vs 质量权衡

**快速模式**（2-5分钟）：
```bash
python visualize_tsne.py --dataset swat --perplexity 15 --n_iter 500 --sample_size 2000
```

**平衡模式**（10-15分钟）：
```bash
python visualize_tsne.py --dataset swat --perplexity 30 --n_iter 1000 --sample_size 5000
```

**高质量模式**（30+分钟）：
```bash
python visualize_tsne.py --dataset swat --perplexity 50 --n_iter 1500 --sample_size 10000
```

### 参数影响

- **perplexity**：
  - 值越小：计算越快，但可能失去全局结构
  - 值越大：显示更多全局结构，但计算慢
  - 建议范围：5-50（样本数据的5%-30%）

- **n_iter**：
  - 值越小：收敛越快
  - 值越大：结果越稳定
  - 建议最小值：300，推荐值：1000+

- **sample_size**：
  - 如果全数据集太大，采样可加速计算
  - 建议值：5000-10000

## 输出说明

脚本会在 `tsne_results/` 目录生成以下文件：

### 可视化图表

1. **{dataset}_tsne_by_label.png**
   - 按异常标签着色（蓝色=正常，红色=异常）
   - 用于评估异常检测性能

2. **{dataset}_tsne_by_density.png**
   - 按局部密度着色（暖色=密集，冷色=稀疏）
   - 用于识别聚类和离群点

3. **{dataset}_embedding_distribution.png**
   - 嵌入向量分布统计分析（4张子图）
   - 包含直方图、各维度统计等

### 数据文件

1. **{dataset}_tsne_results.csv**
   - t-SNE降维后的2D坐标和标签
   - 可用于后续分析或自定义可视化

2. **{dataset}_embeddings_sample.csv**
   - 嵌入向量样本（前100个）
   - 用于分析嵌入空间的特征

## 结果解释

### t-SNE by Label图
- **蓝色聚集区域**：模型将其视为正常数据
- **红色聚集区域**：模型检测到的异常数据
- **颜色混杂区域**：难以分类的边界样本
- **离群点**：可能是真实异常或误检

### t-SNE by Density图
- **高密度区域（暖色）**：样本集中的主要模式
- **低密度区域（冷色）**：可能是异常或边界样本
- **密度突变处**：可能存在决策边界

### Embedding Distribution图
- **直方图**：了解嵌入值的整体分布
- **各维度平均值**：识别哪些维度更重要
- **各维度标准差**：哪些维度信息量大
- **相关性热力图**：维度间的关系

## 常见问题

### Q: t-SNE运行太慢怎么办？

A: 尝试以下方法：
1. 减少采样数量：`--sample_size 2000`
2. 降低困惑度：`--perplexity 15`
3. 减少迭代次数：`--n_iter 500`

### Q: 模型文件找不到怎么办？

A: 确保：
1. 模型文件在 `pretrained/{dataset}/` 目录
2. 使用 `--model_path` 显式指定路径
3. 文件格式为 `.pt`

### Q: 异常样本无法清晰分离怎么办？

A: 这可能表明：
1. 模型需要微调或重新训练
2. 数据集中异常定义不清
3. 需要调整模型超参数

### Q: 如何保存高分辨率图表？

A: 修改 `visualize_tsne.py` 中的 `dpi` 参数：
```python
fig.savefig(..., dpi=600, bbox_inches='tight')
```

## 扩展应用

### 自定义着色
在 `visualize_tsne.py` 中，可以添加自定义着色方式：

```python
def visualize_custom(self, tsne_result, color_values, title="Custom Visualization"):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                         c=color_values, cmap='plasma', s=50)
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()
```

### 与其他工具配合
导出的CSV文件可用于：
- **Plotly/bokeh**：交互式可视化
- **seaborn**：高级统计可视化
- **pandas**：进一步数据分析

## 相关文件

- `visualize_tsne.py` - 主可视化脚本
- `models/GDN.py` - GDN模型（已修改以支持返回嵌入向量）
- `run_tsne.sh` - Linux/Mac运行脚本
- `run_tsne.bat` - Windows运行脚本

## 模型修改说明

为了支持嵌入向量导出，对 `models/GDN.py` 的forward方法进行了修改：

```python
def forward(self, data, org_edge_index=None, return_embeddings=False):
    # ...原代码...
    if return_embeddings:
        return out, x.view(batch_num, node_num, -1), embeddings_dict
    return out
```

现在模型可以返回：
1. **out**: 预测输出
2. **hidden_features**: 隐层特征 [batch, node_num, feature_dim]
3. **embeddings_dict**: 嵌入向量字典（包含node_embedding）

## 性能指标参考

在标准配置下（perplexity=30, n_iter=1000）：
- **SWAT数据集**：约15-20分钟
- **MSL数据集**：约10-15分钟
- **内存占用**：约2-4GB（取决于样本数量）

## 版本历史

- v1.0 (2024-01)
  - 初始版本
  - 支持基础t-SNE可视化
  - 多种着色方案

## 许可证

遵循项目主许可证
