#!/bin/bash
# t-SNE可视化脚本 - 快速运行

echo "==============================================="
echo "GDN 模型 t-SNE 可视化工具"
echo "==============================================="

# 默认参数
DATASET=${1:-"swat"}
PERPLEXITY=${2:-30}
N_ITER=${3:-1000}

echo ""
echo "参数设置:"
echo "  数据集: $DATASET"
echo "  困惑度(perplexity): $PERPLEXITY"
echo "  迭代次数(n_iter): $N_ITER"
echo ""

# 运行可视化
python visualize_tsne.py \
    --dataset $DATASET \
    --perplexity $PERPLEXITY \
    --n_iter $N_ITER \
    --output_dir "./tsne_results" \
    --batch_size 256 \
    --sample_size 5000

echo ""
echo "可视化完成！"
echo "结果保存在: ./tsne_results/"
