# GDN 模型 t-SNE 可视化工具 - 快速入门

## 📋 概述

已为您创建了完整的t-SNE可视化工具包，用于可视化GDN模型的嵌入向量。该工具可以帮助您：

✅ 降维可视化高维嵌入向量  
✅ 区分异常和正常样本的聚类情况  
✅ 分析嵌入向量的分布特性  
✅ 评估模型的学习效果  

---

## 🚀 快速开始（3种方式）

### 方式1️⃣：使用交互式演示（推荐）

```bash
python demo_tsne.py
```

这将打开一个交互式菜单，让您：
- 选择快速/平衡/高质量模式
- 选择数据集（SWAT/MSL）
- 自定义运行参数

### 方式2️⃣：直接运行批处理脚本

**Windows用户：**
```bash
run_tsne.bat
# 或指定数据集：
run_tsne.bat msl
```

**Linux/Mac用户：**
```bash
bash run_tsne.sh
# 或指定数据集：
bash run_tsne.sh msl
```

### 方式3️⃣：完全自定义命令行

```bash
python visualize_tsne.py \
    --dataset swat \
    --perplexity 30 \
    --n_iter 1000 \
    --sample_size 5000 \
    --output_dir ./tsne_results
```

---

## ⚙️ 运行模式对比

| 模式 | 时间 | 质量 | 用途 | 参数 |
|------|------|------|------|------|
| **快速** | 2-5分钟 | ⭐⭐ | 快速测试 | perplexity=15, iter=500 |
| **平衡** | 10-15分钟 | ⭐⭐⭐ | **日常使用** | perplexity=30, iter=1000 |
| **高质量** | 30+分钟 | ⭐⭐⭐⭐ | 发表论文 | perplexity=50, iter=1500 |

---

## 📊 生成的可视化结果

脚本会生成以下图表到 `tsne_results/` 目录：

### 1. t-SNE by Label（按异常标签）
- **蓝点** = 正常样本
- **红点** = 异常样本
- **用途**：评估异常检测性能

### 2. t-SNE by Density（按局部密度）
- **暖色** = 高密度区域
- **冷色** = 低密度区域
- **用途**：识别聚类和离群点

### 3. Embedding Distribution（嵌入分布）
- 直方图、各维度统计、相关性分析
- **用途**：理解嵌入向量的性质

### 4. CSV结果文件
- `{dataset}_tsne_results.csv` - 降维坐标
- `{dataset}_embeddings_sample.csv` - 嵌入向量样本

---

## 📁 文件清单

新增文件：

```
visualize_tsne.py              ← 主可视化脚本（核心）
run_tsne.sh                    ← Linux/Mac启动脚本
run_tsne.bat                   ← Windows启动脚本
demo_tsne.py                   ← 交互式演示工具
TSNE_VISUALIZATION_GUIDE.md    ← 详细使用指南
TSNE_QUICK_START.md           ← 本文件
```

修改文件：
```
models/GDN.py                  ← 已修改forward方法以支持返回嵌入向量
```

---

## 💡 使用示例

### 示例1：SWAT数据集平衡模式（推荐）

```bash
python visualize_tsne.py --dataset swat
```

预计时间：10-15分钟
结果质量：很好

### 示例2：MSL数据集快速测试

```bash
python visualize_tsne.py --dataset msl --perplexity 15 --n_iter 500
```

预计时间：2-5分钟
结果质量：可接受（用于快速验证）

### 示例3：自定义高精度参数

```bash
python visualize_tsne.py \
    --dataset swat \
    --perplexity 40 \
    --n_iter 1500 \
    --sample_size 8000
```

预计时间：25-30分钟
结果质量：优秀（用于论文发表）

---

## 🔍 如何解释结果

### 理想情况
```
✓ 红蓝点明显分离 → 异常检测效果好
✓ 聚集成几个明显的簇 → 嵌入空间结构好
✓ 无大量孤立点 → 模型学习稳定
```

### 需要改进的情况
```
✗ 红蓝点完全混淆 → 模型区分能力弱，可能需要：
  - 调整模型参数
  - 增加训练数据
  - 检查数据质量

✗ 所有点聚成一个团 → 嵌入维度可能不够
  - 考虑增加 dim 参数
  - 增加网络深度

✗ 大量孤立点 → 可能存在数据异常
  - 检查异常标签
  - 审查数据预处理
```

---

## ⚡ 性能优化建议

### 如果运行太慢：

1. **减少采样**
   ```bash
   python visualize_tsne.py --sample_size 2000 --perplexity 15
   ```

2. **使用GPU加速**
   - 确保CUDA可用
   - 增加 `--batch_size 512`

3. **降低质量要求**
   ```bash
   python visualize_tsne.py --n_iter 500 --perplexity 20
   ```

### 如果结果不清晰：

1. **增加困惑度**
   ```bash
   python visualize_tsne.py --perplexity 50
   ```

2. **增加迭代次数**
   ```bash
   python visualize_tsne.py --n_iter 1500
   ```

3. **使用全部数据**
   ```bash
   python visualize_tsne.py --sample_size 20000
   ```

---

## 🐛 常见问题排查

### Q: "模型文件不存在"错误

**原因**：模型文件位置不对

**解决**：
1. 确认 `pretrained/swat/` 目录存在模型文件
2. 或使用 `--model_path` 指定完整路径：
   ```bash
   python visualize_tsne.py --model_path /path/to/model.pt
   ```

### Q: "TimeDataset not found"错误

**原因**：缺少数据文件

**解决**：
1. 确保 `data/swat/` 或 `data/msl/` 目录存在数据文件
2. 或修改脚本中的数据加载路径

### Q: 运行卡顿或内存不足

**原因**：样本量太大

**解决**：
```bash
python visualize_tsne.py --batch_size 128 --sample_size 2000
```

### Q: 图表无法打开或显示不完整

**原因**：文件保存失败或格式错误

**解决**：
1. 检查输出目录权限：`chmod 777 tsne_results/`
2. 尝试其他dpi值
3. 更新matplotlib：`pip install --upgrade matplotlib`

---

## 📈 下一步

### 进阶使用

1. **修改着色方案**
   - 编辑 `visualize_tsne.py` 中的 `visualize_*` 方法
   - 添加自己的着色逻辑

2. **集成到研究流程**
   - 在论文中展示t-SNE可视化
   - 对比训练前后的结果

3. **与其他工具结合**
   - 使用Plotly创建交互式版本
   - 用pandas进行进一步数据分析

### 相关资源

- 详细文档：[TSNE_VISUALIZATION_GUIDE.md](TSNE_VISUALIZATION_GUIDE.md)
- 完整参数说明：在脚本中运行 `python visualize_tsne.py --help`
- t-SNE论文：[van der Maaten & Hinton, 2008]

---

## 📞 技术支持

遇到问题？检查以下内容：

1. ✅ Python版本 >= 3.8
2. ✅ 所需包已安装：torch, sklearn, matplotlib
3. ✅ GPU可用（可选）
4. ✅ 数据和模型文件完整
5. ✅ 输出目录有写入权限

运行诊断：
```bash
python -c "import torch; print('PyTorch OK:', torch.__version__)"
python -c "from sklearn.manifold import TSNE; print('Scikit-learn OK')"
python -c "import matplotlib; print('Matplotlib OK:', matplotlib.__version__)"
```

---

## 版本信息

- **创建日期**：2024年1月
- **工具版本**：1.0
- **GDN模型**：已修改支持嵌入向量导出
- **Python兼容性**：3.8+

---

祝您使用愉快！🎉
