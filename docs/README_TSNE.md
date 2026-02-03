# GDN t-SNE 可视化工具 - 总索引

## 🎯 快速导航

### 🚀 我想立即开始
```
1. python check_tsne_deps.py          ← 检查依赖
2. python demo_tsne.py                 ← 交互式演示
3. 按菜单选择选项运行
4. 查看 tsne_results/ 下的图表
```

### 📖 我想学习使用方法
```
阅读: TSNE_QUICK_START.md
- 快速开始 (3种方式)
- 运行模式对比
- 常见问题解答
```

### 🔧 我想深入掌握
```
阅读: TSNE_VISUALIZATION_GUIDE.md
- 完整功能说明
- 所有参数参考
- 参数优化建议
- 结果解释指南
```

### 📋 我想了解项目详情
```
阅读: IMPLEMENTATION_SUMMARY.md
- 完成工作总结
- 功能特性列表
- 系统需求说明
- 故障排除方案
```

### 📁 我想了解文件结构
```
阅读: FILE_MANIFEST.md
- 新增文件清单
- 修改文件说明
- 文件依赖关系
- 快速命令参考
```

---

## 📂 文件导航表

### 脚本文件

| 文件 | 用途 | 调用方式 |
|------|------|--------|
| **visualize_tsne.py** | t-SNE可视化核心脚本 | `python visualize_tsne.py --help` |
| **demo_tsne.py** | 交互式演示工具 | `python demo_tsne.py` |
| **run_tsne.bat** | Windows启动脚本 | `run_tsne.bat [选项]` |
| **run_tsne.sh** | Linux/Mac启动脚本 | `bash run_tsne.sh [选项]` |
| **check_tsne_deps.py** | 依赖检查工具 | `python check_tsne_deps.py` |

### 文档文件

| 文件 | 对象 | 首选阅读场景 |
|------|------|-----------|
| **TSNE_QUICK_START.md** | 初级/中级用户 | ⭐ 首先阅读 |
| **TSNE_VISUALIZATION_GUIDE.md** | 进阶用户/开发者 | 深入学习 |
| **IMPLEMENTATION_SUMMARY.md** | 项目管理/复审 | 全面了解 |
| **FILE_MANIFEST.md** | 项目维护者 | 参考文档 |
| **README.md** (本文件) | 所有用户 | 快速导航 |

---

## 🏃 快速命令参考

### 最简单的方式
```bash
python demo_tsne.py                    # 交互式菜单
```

### 标准使用
```bash
# SWAT数据集，平衡模式
python visualize_tsne.py --dataset swat

# MSL数据集，快速模式
python visualize_tsne.py --dataset msl --perplexity 15 --n_iter 500
```

### 完全自定义
```bash
python visualize_tsne.py \
    --dataset swat \
    --perplexity 30 \
    --n_iter 1000 \
    --sample_size 5000 \
    --output_dir ./results
```

### 查看帮助
```bash
python visualize_tsne.py --help
```

---

## 🎓 学习路径

### 路径1: 快速上手 (30分钟)
```
1. 阅读 TSNE_QUICK_START.md (10分钟)
   └─ 快速开始、3种运行方式
   
2. 运行 python demo_tsne.py (15分钟)
   └─ 选择平衡模式，耐心等待
   
3. 查看生成的图表 (5分钟)
   └─ 理解 label 和 density 着色的含义
```

### 路径2: 完全掌握 (2小时)
```
1. 阅读 TSNE_QUICK_START.md (15分钟)
   └─ 快速概览
   
2. 阅读 TSNE_VISUALIZATION_GUIDE.md (30分钟)
   └─ 深入理解各功能
   
3. 运行演示和实验 (45分钟)
   └─ 尝试不同模式和参数
   
4. 阅读源代码 visualize_tsne.py (30分钟)
   └─ 理解实现细节
```

### 路径3: 开发者深度 (半天)
```
1. 阅读 IMPLEMENTATION_SUMMARY.md (20分钟)
   └─ 了解实现细节
   
2. 查看 models/GDN.py 修改部分 (15分钟)
   └─ 理解嵌入向量导出逻辑
   
3. 运行源代码分析 (30分钟)
   └─ 追踪数据流
   
4. 设计扩展方案 (剩余时间)
   └─ 计划自己的功能增强
```

---

## 🎯 按任务查找

### "我想快速测试模型"
```bash
python visualize_tsne.py --dataset swat \
    --perplexity 15 --n_iter 500 --sample_size 2000
```
→ 参考: QUICK_START.md 的"快速模式"

### "我想生成论文用的高质量图表"
```bash
python visualize_tsne.py --dataset swat \
    --perplexity 50 --n_iter 1500 --sample_size 10000
```
→ 参考: GUIDE.md 的"参数优化建议"

### "我想自定义着色方案"
```python
# 编辑 visualize_tsne.py
# 参考 visualize_by_label 方法
# 创建自己的 visualize_custom 方法
```
→ 参考: GUIDE.md 的"扩展应用"

### "模型运行太慢"
```bash
python visualize_tsne.py --dataset swat \
    --sample_size 2000 --batch_size 128 --perplexity 15
```
→ 参考: QUICK_START.md 的"性能优化"

### "我看不懂生成的图表"
```
→ 参考: QUICK_START.md 的"如何解释结果"
→ 参考: GUIDE.md 的"结果解释"
```

### "程序报错了"
```
→ 参考: QUICK_START.md 的"常见问题排查"
→ 参考: IMPLEMENTATION_SUMMARY.md 的"故障排除"
```

---

## ❓ 常见问题快速答案

**Q: 第一次运行应该用什么参数?**
```
A: 使用 python demo_tsne.py 选择"平衡模式"
   或 python visualize_tsne.py --dataset swat
```

**Q: 运行需要多长时间?**
```
A: 快速模式 2-5分钟
   平衡模式 10-15分钟 (推荐)
   高质量模式 30分钟+
```

**Q: 需要什么硬件要求?**
```
A: CPU 多核推荐
   内存 4GB最少，8GB+更佳
   GPU 可选（会自动检测）
```

**Q: 生成的图表在哪里?**
```
A: 全部保存在 tsne_results/ 目录
   包含 PNG 图表和 CSV 数据
```

**Q: 可以对比不同模型吗?**
```
A: 可以，分别运行不同模型的脚本
   或修改脚本支持模型路径参数
```

**Q: 如何获得最佳可视化效果?**
```
A: 尝试以下参数:
   --perplexity 40-50
   --n_iter 1500+
   --sample_size 10000
```

---

## 🛠️ 常用命令速查

```bash
# 检查依赖 (首次必做)
python check_tsne_deps.py

# 交互式运行 (推荐新手)
python demo_tsne.py

# SWAT数据集，默认参数
python visualize_tsne.py --dataset swat

# MSL数据集，默认参数
python visualize_tsne.py --dataset msl

# 快速模式 (2-5分钟)
python visualize_tsne.py --perplexity 15 --n_iter 500 --sample_size 2000

# 高质量模式 (30+分钟)
python visualize_tsne.py --perplexity 50 --n_iter 1500 --sample_size 10000

# 自定义输出目录
python visualize_tsne.py --output_dir ./my_results

# 显示所有可用参数
python visualize_tsne.py --help

# Windows快速启动
run_tsne.bat

# Linux/Mac快速启动
bash run_tsne.sh
```

---

## 📊 系统信息检查

运行以下命令检查您的系统:

```bash
# 检查Python版本
python --version

# 检查PyTorch安装
python -c "import torch; print('PyTorch:', torch.__version__)"

# 检查CUDA可用性
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 检查scikit-learn
python -c "import sklearn; print('scikit-learn:', sklearn.__version__)"

# 一键检查所有依赖
python check_tsne_deps.py
```

---

## 📈 性能基准 (参考值)

在标准配置下的预期运行时间:

| 数据集 | 模式 | 时间 | 内存占用 |
|------|------|------|--------|
| SWAT | 快速 | 2-3分钟 | ~1.5GB |
| SWAT | 平衡 | 12-15分钟 | ~2.5GB |
| SWAT | 高质量 | 30-40分钟 | ~3.5GB |
| MSL | 快速 | 1-2分钟 | ~1GB |
| MSL | 平衡 | 8-12分钟 | ~2GB |
| MSL | 高质量 | 25-35分钟 | ~3GB |

*注: 实际时间取决于您的硬件配置*

---

## 🎁 特色功能一览

✨ **t-SNE降维** - 高维→2维可视化  
✨ **多重着色** - 按标签/密度着色  
✨ **分布分析** - 嵌入向量统计  
✨ **数据导出** - CSV格式保存  
✨ **自动配置** - 智能参数推荐  
✨ **模式选择** - 快速/平衡/高质量  
✨ **完整文档** - 详细使用指南  
✨ **依赖检查** - 自动诊断修复  

---

## 🔗 相关资源

### 论文和理论
- t-SNE 原论文: [van der Maaten & Hinton, 2008]
- 参考文献在 GUIDE.md 中有详细说明

### Python库文档
- scikit-learn TSNE: [scikit-learn.org/modules/manifold]
- matplotlib: [matplotlib.org]
- pandas: [pandas.pydata.org]

### 类似工具
- UMAP (更快的替代品)
- PCA (简单快速的降维)
- Seaborn (更多可视化选项)

---

## 📞 获取帮助

### 快速帮助
1. 查看 QUICK_START.md 的 FAQ
2. 运行 `python visualize_tsne.py --help`
3. 检查 check_tsne_deps.py 的诊断输出

### 详细帮助
1. 阅读 VISUALIZATION_GUIDE.md
2. 查看 visualize_tsne.py 的代码注释
3. 参考 IMPLEMENTATION_SUMMARY.md

### 故障排除
1. 参考 QUICK_START.md 的"常见问题排查"
2. 参考 IMPLEMENTATION_SUMMARY.md 的"故障排除"
3. 尝试运行 check_tsne_deps.py 诊断

---

## ✅ 检查清单

初次使用前，请确保:

- [ ] Python >= 3.8 已安装
- [ ] 运行了 `python check_tsne_deps.py`
- [ ] 所有依赖包已安装
- [ ] 已阅读 TSNE_QUICK_START.md
- [ ] 数据文件在 data/{dataset}/ 目录
- [ ] 模型文件在 pretrained/{dataset}/ 目录
- [ ] 输出目录有写入权限

---

## 🎯 现在就开始吧!

### 最简单的3步:
```bash
1. python check_tsne_deps.py          # 检查依赖
2. python demo_tsne.py                 # 运行演示
3. 选择"平衡模式"，按Enter继续        # 完成!
```

### 预期结果:
- 10-15分钟后完成
- 在 tsne_results/ 看到3张图表
- 理解模型的嵌入效果

---

## 📅 版本信息

- 版本: 1.0
- 发布日期: 2024年1月29日
- Python: 3.8+
- 主要依赖: PyTorch, scikit-learn, matplotlib

---

## 🙏 感谢

感谢您使用GDN t-SNE可视化工具！

有问题或建议? 欢迎反馈!

---

**祝您使用愉快！** 🎉

*最后更新: 2024-01-29*
