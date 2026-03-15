# 《人工智能导论》HW02 论文导读与DeepSeek Chatbot实践
人工智能专业课程作业，提交截止2026年3月19日12:00

## 任务一：论文导读说明
1. **论文来源**
   - 论文标题：Hybrid CNN-Mamba Model for Multi-Scale Retinal Image Enhancement
   - 出处：2025年 International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI)
   - 发表年份：2025年
2. **导读生成方式**：使用DeepSeek R1大模型基于论文原文生成核心正文，人工润色调整结构与表述
3. **配图方式**：所有配图均从论文原文中手动截取，插入文档并补充对应图注，无AI自动生成配图

## 任务二：Chatbot说明
1. **采用平台/API**：DeepSeek官方OpenAI兼容接口，同时兼容火山引擎DeepSeek R1模型API
2. **核心功能**：实现「用户文本输入→调用DeepSeek大模型→获取并打印模型回复」的完整对话流程
3. **运行环境**：Python 3.8+
4. **依赖安装**：`pip install -r requirements.txt`
5. **API配置方式**：将DeepSeek API Key配置为系统环境变量`DEEPSEEK_API_KEY`，避免硬编码密钥泄露
6. **运行命令**：`python chatbot_deepseek.py`

## 目录结构说明
hw02/├── 导读_Hybrid_CNN_Mamba_多尺度眼底图像增强.md # 论文导读完整文档├── chatbot_deepseek.py # Chatbot 可运行代码├── requirements.txt # Python 依赖├── README.md # 项目说明└── images/ # 导读配图文件夹