# utils/data_loader.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import BATCH_SIZE, NUM_WORKERS, VALID_SPLIT

def get_data_loaders():
    """获取训练集、验证集、测试集的 DataLoader"""
    # 数据预处理：归一化到 [0,1] 并标准化（均值0.1307，标准差0.3081 为 MNIST 统计值）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练集
    full_train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 划分训练集和验证集
    val_size = int(len(full_train_set) * VALID_SPLIT)
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # 测试集
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader