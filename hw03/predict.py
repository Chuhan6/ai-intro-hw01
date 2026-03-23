# predict.py
import torch
from PIL import Image
import numpy as np
from models.model import CNN
from config import DEVICE, MODEL_SAVE_PATH
import argparse

def preprocess_image(image_path):
    """将单张图片预处理为模型输入格式（MNIST 风格）"""
    img = Image.open(image_path).convert('L')          # 转为灰度图
    img = img.resize((28, 28))                         # 调整大小
    img = np.array(img, dtype=np.float32)
    img = img / 255.0                                  # 归一化到 [0,1]
    # 标准化（与训练时一致）
    img = (img - 0.1307) / 0.3081
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度
    return img

def main():
    parser = argparse.ArgumentParser(description='Predict digit from an image')
    parser.add_argument('image_path', type=str, help='Path to input image')
    args = parser.parse_args()

    # 加载模型
    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # 预处理图像
    input_tensor = preprocess_image(args.image_path).to(DEVICE)

    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prob = torch.softmax(output, dim=1)[0][pred].item()

    print(f"Predicted digit: {pred.item()} (confidence: {prob:.4f})")

if __name__ == '__main__':
    main()