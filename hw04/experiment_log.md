# 实验记录：Whisper 音频识别测试

## 环境说明
- 操作系统：Windows 11
- Python版本：3.10.6
- 硬件：Intel i7-1165G7，16GB RAM，无GPU
- 依赖：torch (CPU版)，openai-whisper

## 测试音频
1. 使用任务二导出的 `ai_future_voice.mp3`（时长约50秒，内容为人工智能科普）。
2. 使用自录一段中文语音（约15秒，内容：“今天的天气很好，适合出门散步。”）。

## 识别命令及结果
```bash
whisper ai_future_voice.mp3 --model tiny --language zh