该目录包含SWaT和WADI的主要预处理代码。

### SWaT
1. 将 SWaT_Dataset_Normal_v0/SWaT_Dataset_Attack_v0.xlsx 转换为 csv，将 'Normal' / 'Normal/Attack' 列重命名为 'attack'，并将标签设为 0/1
2. 将它们重命名为 'swat_train.csv' 和 'swat_test.csv'，作为 process_swat.py 的输入文件
3. 运行脚本 `python process_swat.py`

### WADI
1. 根据攻击描述文档，添加带有 0/1 'attack' 列的 WADI_attackdata.csv，并将文件重命名为 'WADI_attackdata_labelled.csv'
2. 运行脚本 `python process_wadi.py`

### 其他
你可以通过 [此链接](https://drive.google.com/drive/folders/1_4TlatKh-f7QhstaaY7YTSCs8D4ywbWc?usp=sharing) 获取部分处理后的数据。