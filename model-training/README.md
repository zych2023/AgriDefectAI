# 模型训练模块 — 农作物病害识别

基于 ResNet50 深度卷积神经网络的 36 类农作物病害分类模型，涵盖 10 种作物（苹果、樱桃、玉米、葡萄、柑桔、桃、辣椒、马铃薯、草莓、番茄）。

## 目录结构

```
model-training/
├── models/              # ResNet50 模型定义 + 特征提取
├── Data/                # 数据集加载器（61→36标签映射）
├── api/                 # 推理服务（FastAPI + DiseasePredictor + Web UI）
│   ├── infer.py         #   推理引擎（PyTorch + ONNX 双后端）
│   ├── api.py           #   FastAPI RESTful 服务
│   └── static/          #   Web 可视化界面
├── checkpoints/         # 模型产物（.gitignore 排除）
├── docs/                # 文档
├── config.yaml          # 训练/推理配置
├── knowledge_base.json  # 36类病害防治知识库
├── train.py             # 训练主脚本
├── evaluate.py          # 模型评估
├── export.py            # PyTorch → ONNX 导出
└── build_index.py       # 特征库构建（相似检索）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

下载 AI Challenger 2018 农作物叶子图像数据集，放到 `../Data/` 目录下：

```
../Data/
├── AgriculturalDisease_trainingset/
│   ├── images/          # 训练图片
│   └── train_list.txt   # 训练标注
└── AgriculturalDisease_validationset/
    ├── images/          # 验证图片
    └── AgriculturalDisease_validation_annotations.json
```

数据集路径可在 `config.yaml` 中修改。

### 3. 训练模型

```bash
python train.py
```

### 4. 导出 ONNX

```bash
python export.py
```

### 5. 构建特征库（可选，用于相似病例检索）

```bash
python build_index.py
```

### 6. 启动推理服务

```bash
cd api && python api.py
# → http://localhost:8000/docs
```

## 模型指标

| 指标 | 数值 |
|------|------|
| 验证集 Top-1 准确率 | 90.18% |
| 分类类别 | 36 类（10 种作物） |
| ONNX 模型大小 | ~91MB |
| 单张推理时间 | <1s (GPU) / <3s (CPU) |

## 技术栈

- **框架**: PyTorch, ONNX Runtime
- **骨干网络**: ResNet50 (ImageNet 预训练)
- **训练策略**: AdamW + Cosine LR + Warmup + AMP + MixUp + Label Smoothing
- **部署**: FastAPI + ONNX（支持昇腾端侧推理）
