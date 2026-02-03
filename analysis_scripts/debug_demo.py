# -*- coding: utf-8 -*-
"""
快速调试演示脚本
演示如何在GDN训练中使用调试断点查看数据变化
"""

# 在 train.py 中的第112行后添加以下代码（训练循环开始处）

"""
===========================================
方法1: 使用 pdb 调试器（最简单）
===========================================
"""

# 添加在 train.py 第112行之后：
'''
    for i_epoch in range(epoch):
        epoch_start_time = time.time()
        
        # 🔥 添加断点：只在第1个epoch的第1个batch暂停
        if i_epoch == 0:
            import pdb; pdb.set_trace()
        
        logger.log_epoch_start(i_epoch, epoch)
'''

# 然后运行： python main.py -dataset swat -epoch 5
# 程序会在断点处暂停，你可以输入：
#   p x.shape          # 查看输入shape
#   p out.shape        # 查看输出shape  
#   p loss.item()      # 查看损失值
#   c                  # 继续运行


"""
===========================================
方法2: 添加详细打印（推荐）
===========================================
"""

# 在 train.py 第125-130行（模型前向传播处）添加：
'''
            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            # 🔥 添加详细打印
            if batch_idx == 0 and i_epoch % 5 == 0:  # 每5个epoch打印一次
                print("\n" + "="*70)
                print(f"🔍 调试信息 [Epoch {i_epoch}, Batch {batch_idx}]")
                print("="*70)
                print(f"  📥 输入 x:")
                print(f"     Shape: {x.shape}")
                print(f"     Range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"     Mean:  {x.mean():.4f} ± {x.std():.4f}")
                print(f"\n  📤 输出 out:")
                print(f"     Shape: {out.shape}")
                print(f"     Range: [{out.min():.4f}, {out.max():.4f}]")
                print(f"     Mean:  {out.mean():.4f} ± {out.std():.4f}")
                print(f"\n  🎯 标签 labels:")
                print(f"     Shape: {labels.shape}")
                print(f"     Range: [{labels.min():.4f}, {labels.max():.4f}]")
                print(f"     Mean:  {labels.mean():.4f} ± {labels.std():.4f}")
                print(f"\n  📊 损失:")
                print(f"     MSE Loss: {loss.item():.6f}")
                
                # 计算每个传感器的预测误差
                errors = (out - labels).abs().mean(dim=0)
                print(f"\n  📈 各传感器预测误差:")
                print(f"     平均误差: {errors.mean():.4f}")
                print(f"     最大误差: {errors.max():.4f} (传感器 {errors.argmax()})")
                print(f"     最小误差: {errors.min():.4f} (传感器 {errors.argmin()})")
                print("="*70 + "\n")
            
            loss.backward()
'''


"""
===========================================
方法3: 使用debug_utils工具（功能最强）
===========================================
"""

# 在 train.py 顶部导入：
'''
from util.debug_utils import print_tensor_stats, plot_batch_distribution, check_gradients
'''

# 在 train.py 第125-140行添加：
'''
            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            # 🔥 使用调试工具
            if batch_idx == 0 and i_epoch == 0:  # 第一个epoch的第一个batch
                from util.debug_utils import print_tensor_stats, plot_batch_distribution
                
                # 打印统计信息
                print_tensor_stats(x, "输入x", show_values=False)
                print_tensor_stats(out, "模型输出", show_values=False)
                print_tensor_stats(labels, "真实标签", show_values=False)
                
                # 可视化分布
                plot_batch_distribution(x, "第0个epoch输入分布")
                plot_batch_distribution(out.unsqueeze(-1), "第0个epoch输出分布")
            
            loss.backward()
            
            # 🔥 检查梯度
            if batch_idx == 0 and i_epoch == 0:
                from util.debug_utils import check_gradients
                grad_info = check_gradients(model, threshold=10.0)
                if grad_info['has_nan']:
                    print("⚠️ 发现NaN梯度，请检查！")
            
            optimizer.step()
'''


"""
===========================================
方法4: 使用VS Code调试器（可视化最好）
===========================================
"""

# 步骤：
# 1. 在VS Code中打开 train.py
# 2. 在第125行（out = model(x, edge_index)）左侧点击，添加红色断点
# 3. 按F5，选择 "🐛 调试GDN (小数据集)"
# 4. 程序运行到断点处会暂停
# 5. 在左侧"变量"面板查看所有变量
# 6. 在"调试控制台"输入表达式：
#    - x.shape
#    - out.mean()
#    - loss.item()
# 7. 按F5继续，或F10单步执行


"""
===========================================
推荐使用流程
===========================================
"""

print("""
📖 推荐的调试流程：

1️⃣ 第一次运行（了解整体流程）：
   python main.py -dataset swat -epoch 2 -batch 32 --debug --debug_batch 1
   → 查看日志文件了解数据流向

2️⃣ 发现问题（深入调试）：
   - 在train.py中需要检查的位置添加打印语句
   - 或使用VS Code断点调试
   
3️⃣ 排查数据异常：
   - 使用 debug_utils.py 中的工具函数
   - 打印统计信息
   - 生成可视化图片

4️⃣ 检查梯度问题：
   - 使用 check_gradients() 函数
   - 查看是否有梯度爆炸/消失

💡 提示：
   - 对于快速验证：直接用 print()
   - 对于深入分析：使用 VS Code 断点
   - 对于持续监控：启用 --debug 日志
""")


"""
===========================================
实战例子：追踪第一个batch的数据流
===========================================
"""

# 完整示例代码（添加到 train.py）:
'''
def train(...):
    # ... 前面的代码 ...
    
    for i_epoch in range(epoch):
        for batch_idx, (x, labels, attack_labels, edge_index) in enumerate(dataloader):
            
            # 移动到GPU
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]
            
            # 🔥🔥🔥 调试代码开始 🔥🔥🔥
            if batch_idx == 0 and i_epoch == 0:
                print("\n" + "🔍"*40)
                print("【数据流追踪】第0个epoch，第0个batch")
                print("🔍"*40)
                
                # 1. 查看原始输入
                print(f"\n1️⃣ 原始输入数据:")
                print(f"   x.shape = {x.shape}  # [batch=32, sensors=38, time_steps=15]")
                print(f"   x的取值范围: [{x.min():.3f}, {x.max():.3f}]")
                print(f"   第一个样本的第一个传感器的时间序列: {x[0, 0, :]}")
                
                # 2. 模型前向传播
                print(f"\n2️⃣ 模型前向传播...")
                
            # 前向传播
            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            
            if batch_idx == 0 and i_epoch == 0:
                # 3. 查看模型输出
                print(f"\n3️⃣ 模型输出:")
                print(f"   out.shape = {out.shape}  # [batch=32, sensors=38]")
                print(f"   out的取值范围: [{out.min():.3f}, {out.max():.3f}]")
                print(f"   第一个样本的预测: {out[0, :5]}  # 前5个传感器")
                
                # 4. 查看真实标签
                print(f"\n4️⃣ 真实标签:")
                print(f"   labels.shape = {labels.shape}  # [batch=32, sensors=38]")
                print(f"   第一个样本的真值: {labels[0, :5]}  # 前5个传感器")
                
            # 计算损失
            loss = loss_func(out, labels)
            
            if batch_idx == 0 and i_epoch == 0:
                # 5. 查看损失
                print(f"\n5️⃣ 损失计算:")
                print(f"   MSE Loss = {loss.item():.6f}")
                
                # 6. 手动计算第一个样本的MSE验证
                sample_mse = ((out[0] - labels[0]) ** 2).mean().item()
                print(f"   第一个样本的MSE (验证) = {sample_mse:.6f}")
                
            # 反向传播
            loss.backward()
            
            if batch_idx == 0 and i_epoch == 0:
                # 7. 查看梯度
                print(f"\n6️⃣ 梯度信息:")
                first_param = next(model.parameters())
                if first_param.grad is not None:
                    print(f"   第一层参数的梯度范围: [{first_param.grad.min():.6f}, {first_param.grad.max():.6f}]")
                    print(f"   梯度的平均值: {first_param.grad.mean():.6f}")
                else:
                    print(f"   ⚠️ 梯度为None")
                
                print("\n" + "🔍"*40 + "\n")
                
                # 可选：在这里设置断点暂停
                # import pdb; pdb.set_trace()
            
            # 🔥🔥🔥 调试代码结束 🔥🔥🔥
            
            optimizer.step()
            # ... 后续代码 ...
'''

print("\n✅ 调试演示脚本准备完成！")
print("📝 请根据上面的示例代码修改您的 train.py 文件")
