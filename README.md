# GDN
论文《基于图神经网络的多元时间序列异常检测》（AAAI'21）的代码实现  
原文链接：[https://arxiv.org/pdf/2106.06947.pdf](https://arxiv.org/pdf/2106.06947.pdf)

# 安装说明
## 环境要求
* Python ≥ 3.6
* CUDA == 10.2
* [PyTorch==1.5.1](https://pytorch.org/)（深度学习框架）
* [PyG: torch-geometric==1.5.0](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)（PyTorch几何图神经网络库）

## 安装依赖包
```bash
# 先安装对应版本的PyTorch，再执行以下命令
bash install.sh
```

## 快速验证
运行以下命令检查环境是否配置成功
```bash
bash run.sh cpu msl
# 或使用GPU运行
bash run.sh <gpu_id> msl    # 示例：bash run.sh 1 msl（使用1号GPU）
```

# 使用说明
我们采用MSL数据集的部分数据作为演示示例（该数据集参考自[telemanom](https://github.com/khundman/telemanom)）。

> **⚠️ 数据集说明**  
> 由于数据集文件较大，`data/` 文件夹未包含在本仓库中。请按照以下说明获取数据集：
> - **MSL数据集**：已包含示例数据（请自行从 [telemanom](https://github.com/khundman/telemanom) 下载完整数据集）
> - **SWaT/WADI数据集**：需从 [iTrust官网](https://itrust.sutd.edu.sg/) 申请获取
> - 下载后请将数据集放置在 `data/` 目录下，参考下方的目录结构说明


## 数据准备
```
# 将你的数据集放在data/目录下，目录结构需与data/msl/保持一致
data
 |-msl                  # 示例数据集
 | |-list.txt           # 特征名称文件，每行一个特征
 | |-train.csv          # 训练数据
 | |-test.csv           # 测试数据
 |-your_dataset         # 你的自定义数据集
 | |-list.txt           # 特征名称文件
 | |-train.csv          # 训练数据
 | |-test.csv           # 测试数据
 | ...
```

### 注意事项
1. CSV文件的第一列将被视为索引列，不参与模型训练。
2. CSV文件中的列顺序无需与list.txt中的特征顺序一致，程序会根据list.txt自动重新排列列顺序。
3. 测试数据文件test.csv必须包含名为“attack”的列，用于存储异常标签（0表示正常，1表示异常）。

## 运行模型
```bash
# 使用GPU运行（指定GPU编号）
bash run.sh <gpu_id> <dataset>

# 或使用CPU运行
bash run.sh cpu <dataset>
```
可在run.sh文件中修改模型运行参数（如学习率、迭代次数等）。

# 其他说明
SWaT和WADI数据集需从[iTrust官网](https://itrust.sutd.edu.sg/)申请获取。

# 引用说明
如果本代码仓库或相关研究成果对您的研究有帮助，请引用以下论文：
```
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
