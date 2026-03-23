# config.py
import torch

# 数据集相关
BATCH_SIZE = 64
NUM_WORKERS = 2
VALID_SPLIT = 0.1          # 从训练集中划分 10% 作为验证集

# 训练相关
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径
MODEL_SAVE_PATH = './results/best_model.pth'
LOSS_CURVE_PATH = './results/loss_curve.png'
ACC_CURVE_PATH = './results/acc_curve.png'
CONFUSION_MATRIX_PATH = './results/confusion_matrix.png'