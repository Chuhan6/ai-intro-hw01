# utils/visualize.py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch

def plot_curves(train_losses, val_losses, train_accs, val_accs, loss_path, acc_path):
    """绘制训练/验证损失曲线和准确率曲线"""
    epochs = range(1, len(train_losses) + 1)

    # 损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(loss_path)   # 实际上 loss_path 和 acc_path 会保存同一张图，这里修改为保存总图
    # 修改：保存一张图包含两个子图，更符合习惯
    plt.savefig(loss_path.replace('loss_curve.png', 'curves.png'))  # 可选
    plt.savefig(acc_path.replace('acc_curve.png', 'curves.png'))     # 与上一行重复，简单处理：保存为 curves.png
    # 简化：直接保存为一个文件
    plt.savefig('./results/curves.png')
    plt.close()

def plot_confusion_matrix(model, test_loader, device, save_path):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix on Test Set')
    plt.savefig(save_path)
    plt.close()