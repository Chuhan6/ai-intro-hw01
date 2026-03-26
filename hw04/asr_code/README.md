markdown
# ASR 代码运行说明

## 依赖安装
```bash
pip install -r requirements.txt
注意：torch 需根据你的系统选择CPU或GPU版本。若使用CPU，可直接安装 torch 默认版本；若需GPU，请参考 PyTorch官网。

运行识别
bash
python recognize.py 音频文件路径
示例：

bash
python recognize.py ../ai_future_voice.mp3
脚本将使用 Whisper tiny 模型进行识别，输出识别文本及耗时。

可选参数
--model：选择模型大小（tiny, base, small, medium, large），默认 tiny。

--language：指定语言（如 zh, en），默认自动检测。

修改脚本中的默认参数或通过命令行传递。

text
