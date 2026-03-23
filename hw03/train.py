# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.model import CNN
from utils.data_loader import get_data_loaders
from utils.visualize import plot_curves, plot_confusion_matrix
from config import DEVICE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, LOSS_CURVE_PATH, ACC_CURVE_PATH, CONFUSION_MATRIX_PATH
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch，返回平均损失和准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    """评估模型，返回平均损失和准确率"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def main():
    # 创建保存结果的目录
    os.makedirs('./results', exist_ok=True)

    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # 初始化模型、损失函数、优化器
    model = CNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录指标
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Best model saved with val acc {val_acc:.2f}%")

    print("\nTraining finished. Best validation accuracy: {:.2f}%".format(best_val_acc))

    # 绘制曲线
    plot_curves(train_losses, val_losses, train_accs, val_accs, LOSS_CURVE_PATH, ACC_CURVE_PATH)

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"\nTest set performance using best model: Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, DEVICE, CONFUSION_MATRIX_PATH)

if __name__ == '__main__':
    main()