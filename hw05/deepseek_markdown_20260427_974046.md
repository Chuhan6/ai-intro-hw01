# 调试记录

## 1. 下载 MNIST 数据集超时
- **现象**：运行脚本后长时间卡在 Downloading...
- **原因**：网络至 torchvision 官方链接缓慢或不可访问。
- **修改**：手动下载四个 .gz 文件至 `data/MNIST/raw/` 目录后正常运行；或在代码中添加镜像源。

## 2. CUDA out of memory
- **现象**：训练时显存溢出报错
- **原因**：较大 batch_size + 其他进程占用
- **修改**：将 batch_size 改为 32 或添加 `torch.cuda.empty_cache()`，并检查显存占用情况。

## 3. 梯度形状不匹配
- **现象**：RuntimeError: size mismatch
- **原因**：最初设计极简 CNN 时全连接层输入维度计算错误（未考虑 padding）
- **修改**：重新计算各层输出尺寸，修正 `fc1` 输入尺寸为 `32*7*7`。

## 4. LeNet-5 准确率低（<80%）
- **现象**：初期训练准确率无法提升
- **原因**：忘记对输出进行 softmax 转换，但损失函数 CrossEntropyLoss 已内置 log_softmax，故不影响；真正原因是激活函数错用为 ReLU 与论文 tanh 不一致，学习率过大导致震荡。
- **修改**：改用 tanh 激活，学习率从 0.1 调至 0.01，准确率恢复正常。