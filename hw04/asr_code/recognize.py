#!/usr/bin/env python3
"""
语音识别脚本：基于 OpenAI Whisper 识别音频文件。
用法：python recognize.py <音频文件路径> [--model <模型大小>] [--language <语言>]
"""

import argparse
import time
import whisper

def main():
    parser = argparse.ArgumentParser(description="语音识别")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 模型大小")
    parser.add_argument("--language", default=None, help="语言代码，如 zh, en，不指定则自动检测")
    args = parser.parse_args()

    print(f"加载模型 {args.model} ...")
    model = whisper.load_model(args.model)
    print("模型加载完成，开始识别...")
    start = time.time()
    result = model.transcribe(args.audio, language=args.language)
    elapsed = time.time() - start

    print("\n识别结果：")
    print(result["text"])
    print(f"\n识别耗时：{elapsed:.2f} 秒")

if __name__ == "__main__":
    main()