# GDN 分析工具说明

本目录包含用于分析GDN模型的各种脚本和结果文件。

## 📁 目录结构

```
GDN/
├── analysis_scripts/          # 分析脚本
│   ├── view_embeddings.py                    # 查看嵌入向量
│   ├── visualize_embeddings_quick.py         # 快速可视化嵌入向量
│   ├── compare_embeddings_before_after.py    # 对比训练前后嵌入
│   ├── print_cosine_similarity.py            # 打印余弦相似度(完整版)
│   ├── print_cosine_similarity_simple.py     # 打印余弦相似度(简化版)
│   ├── check_topk_neighbors.py               # 检查Top-K邻居
│   ├── visualize_topk_graph.py               # 可视化Top-K有向图
│   ├── visualize_single_node.py              # 可视化单个节点
│   ├── visualize_adjacency_matrix.py         # 可视化邻接矩阵
│   ├── visualize_simple_binary.py            # 文本格式邻接矩阵
│   └── demonstrate_weighted_aggregation.py   # 演示加权聚合
│
├── analysis_results/          # 分析结果
│   ├── embeddings/           # 嵌入向量分析结果
│   ├── adjacency/            # 邻接矩阵结果
│   ├── graphs/               # 图可视化结果
│   └── topk/                 # Top-K邻居结果
│
├── docs/                      # 文档
│   ├── 余弦相似度分析使用指南.md
│   ├── 嵌入向量分析示例.md
│   └── 嵌入向量学习效果说明.md
│
└── (原有GDN核心文件)
    ├── main.py
    ├── train.py
    ├── test.py
    ├── evaluate.py
    ├── models/
    ├── util/
    └── ...
```

## 🚀 快速开始

### 1. 查看嵌入向量
```bash
# 基本查看
python analysis_scripts/view_embeddings.py --basic

# 详细查看
python analysis_scripts/view_embeddings.py --detailed --num_nodes 10

# 生成可视化
python analysis_scripts/visualize_embeddings_quick.py
```

### 2. 分析余弦相似度
```bash
# 打印相似度矩阵
python analysis_scripts/print_cosine_similarity.py --model_path pretrained/msl/best_01_07-154250.pt

# 简化版本
python analysis_scripts/print_cosine_similarity_simple.py
```

### 3. 检查Top-K邻居
```bash
# 查看所有节点
python analysis_scripts/check_topk_neighbors.py --show_all

# 查看特定节点
python analysis_scripts/check_topk_neighbors.py --nodes 0 1 2 5
```

### 4. 可视化图结构
```bash
# 生成所有图
python analysis_scripts/visualize_topk_graph.py --mode all

# 生成单个节点的图
python analysis_scripts/visualize_single_node.py --node 2
```

### 5. 分析邻接矩阵
```bash
# 生成邻接矩阵可视化
python analysis_scripts/visualize_adjacency_matrix.py

# 生成文本格式
python analysis_scripts/visualize_simple_binary.py
```

### 6. 演示加权聚合
```bash
# 演示节点2的加权聚合过程
python analysis_scripts/demonstrate_weighted_aggregation.py --node 2
```

## 📊 结果文件说明

### embeddings/ (嵌入向量结果)
- `embeddings.npy` - 原始64维嵌入向量
- `embeddings_2d_visualization.png` - PCA/t-SNE降维可视化
- `cosine_similarity.csv` - 余弦相似度矩阵

### adjacency/ (邻接矩阵结果)
- `topk_adjacency_matrix.csv` - Top-K邻接矩阵(数值)
- `topk_adjacency_matrix_binary.csv` - 二值邻接矩阵
- `*.png` - 各种可视化图
- `*.txt` - 文本格式邻接矩阵

### graphs/ (图可视化结果)
- `topk_graph_full.png` - 完整Top-K有向图
- `topk_graph_high_similarity.png` - 高相似度边图
- `topk_graph_circular.png` - 环形布局图

### topk/ (Top-K邻居结果)
- `topk_neighbors.txt` - 所有节点的Top-K邻居列表
- `node_*_topk_graph.png` - 单个节点的邻居图
- `aggregation_details.txt` - 加权聚合详情

## 📚 相关文档

详细使用说明请参考 `docs/` 目录下的文档:
- **余弦相似度分析使用指南.md** - 如何分析节点相似度
- **嵌入向量分析示例.md** - 各种分析方法示例
- **嵌入向量学习效果说明.md** - 理解模型学习到的嵌入

## 💡 常用分析流程

1. **基础分析**
   ```bash
   python analysis_scripts/view_embeddings.py --basic
   python analysis_scripts/visualize_embeddings_quick.py
   ```

2. **深入分析**
   ```bash
   python analysis_scripts/check_topk_neighbors.py --show_all
   python analysis_scripts/visualize_topk_graph.py --mode all
   ```

3. **特定节点分析**
   ```bash
   python analysis_scripts/visualize_single_node.py --node 2
   python analysis_scripts/demonstrate_weighted_aggregation.py --node 2
   ```

## ⚙️ 参数说明

所有脚本都支持以下通用参数:
- `--model_path` - 模型路径(默认: pretrained/msl/best_01_07-154250.pt)
- `--node_num` - 节点数量(默认: 27)
- `--dim` - 嵌入维度(默认: 64)
- `--topk` - K值(默认: 20)

具体脚本的详细参数请使用 `--help` 查看。
