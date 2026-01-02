gpu_n=$1
DATASET=$2

seed=5
# 原始参数（针对低内存设备）
# BATCH_SIZE=32
# SLIDE_WIN=5
# topk=5
# out_layer_inter_dim=128

# 针对NVIDIA Tesla T4 16GB GPU和32GB内存优化的参数（适合16GB+显存和32GB内存配置）
BATCH_SIZE=128  # 增加批次大小，充分利用GPU内存
SLIDE_WIN=15    # 增加滑动窗口大小，捕捉更长时间依赖
dim=64
out_layer_num=1
SLIDE_STRIDE=1
topk=20         # 增加邻居数量，提高模型表示能力
out_layer_inter_dim=256  # 增加输出层维度，提高模型容量
val_ratio=0.2
decay=0

# 针对NVIDIA RTX 4090 24GB GPU优化的参数（适合24GB+显存配置，已注释）
# BATCH_SIZE=256  # 进一步增加批次大小，充分利用RTX 4090的24GB内存
# SLIDE_WIN=20    # 增加滑动窗口大小，捕捉更长时间依赖
# topk=30         # 增加邻居数量，提高模型表示能力
# out_layer_inter_dim=512  # 增加输出层维度，提高模型容量


path_pattern="${DATASET}"
COMMENT="${DATASET}"

# 原始参数
# EPOCH=30
# 针对NVIDIA Tesla T4 16GB GPU优化的参数
EPOCH=100  # 增加epoch数量，提高模型性能
report='best'

if [[ "$gpu_n" == "cpu" ]]; then
    python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        -device 'cpu' \
        --debug
else
    CUDA_VISIBLE_DEVICES=$gpu_n  python main.py \
        -dataset $DATASET \
        -save_path_pattern $path_pattern \
        -slide_stride $SLIDE_STRIDE \
        -slide_win $SLIDE_WIN \
        -batch $BATCH_SIZE \
        -epoch $EPOCH \
        -comment $COMMENT \
        -random_seed $seed \
        -decay $decay \
        -dim $dim \
        -out_layer_num $out_layer_num \
        -out_layer_inter_dim $out_layer_inter_dim \
        -val_ratio $val_ratio \
        -report $report \
        -topk $topk \
        --debug
fi