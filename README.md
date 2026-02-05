# GDN

代码实现：[Graph Neural Network-Based Anomaly Detection in Multivariate Time Series(AAAI'21)](https://arxiv.org/pdf/2106.06947.pdf)


# 安装
### 环境要求
* Python >= 3.6
* cuda == 10.2
* [Pytorch==1.5.1](https://pytorch.org/)
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### 安装包
```bash
    pip install -r requirements.txt
```

### 快速开始
运行以下命令检查环境是否准备就绪
    # 在 CPU 上运行
    python main.py -dataset msl -save_path_pattern msl -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment msl -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -topk 5 -device cpu

    # 在 GPU 上运行 (例如 GPU 0)
    # Linux (设置 CUDA_VISIBLE_DEVICES 环境变量): 
    # CUDA_VISIBLE_DEVICES=0 python main.py ...
    
    # Windows (如果需要，在运行前通过 set 命令设置，或依赖内部设备选择（如果支持）):
    # set CUDA_VISIBLE_DEVICES=0
    # python main.py ...



# 使用方法
我们使用部分 msl 数据集(参考 [telemanom](https://github.com/khundman/telemanom)) 作为演示示例。

## 数据准备
```
# 将你的数据集放在 data/ 目录下，结构与 data/msl/ 相同

data
 |-msl
 | |-list.txt    # 特征名称，每行一个特征
 | |-train.csv   # 训练数据
 | |-test.csv    # 测试数据
 |-your_dataset
 | |-list.txt
 | |-train.csv
 | |-test.csv
 | ...

```

### 注意事项:
* .csv 中的第一列将被视为索引列。
* .csv 中的列顺序不需要与 list.txt 中的顺序匹配，我们将根据 list.txt 中的顺序重新排列数据列。
* test.csv 应该有一个名为 "attack" 的列，其中包含被攻击或未被攻击的真实标签(0/1) (0: 正常, 1: 被攻击)

## 运行
```bash
    python main.py -dataset <dataset> -save_path_pattern <dataset> -slide_stride 1 -slide_win 5 -batch 32 -epoch 30 -comment <dataset> -random_seed 5 -decay 0 -dim 64 -out_layer_num 1 -out_layer_inter_dim 128 -val_ratio 0.2 -report best -topk 5
```
你可以在上面的命令中更改运行参数。

# 其他
SWaT 和 WADI 数据集可以从 [iTrust](https://itrust.sutd.edu.sg/) 申请。


# 引用
如果你发现这个仓库或我们的工作对你的研究有用，请考虑引用该论文
```bibtex
@inproceedings{deng2021graph,
  title={Graph neural network-based anomaly detection in multivariate time series},
  author={Deng, Ailin and Hooi, Bryan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={5},
  pages={4027--4035},
  year={2021}
}
```
