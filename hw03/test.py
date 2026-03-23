# test.py
import torch
from models.model import CNN
from utils.data_loader import get_data_loaders
from utils.visualize import plot_confusion_matrix
from config import DEVICE, MODEL_SAVE_PATH, CONFUSION_MATRIX_PATH

def main():
    # 加载数据
    _, _, test_loader = get_data_loaders()

    # 加载模型
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 评估测试集准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100.0 * correct / total:.2f}%")

    # 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, DEVICE, CONFUSION_MATRIX_PATH)

if __name__ == '__main__':
    main()