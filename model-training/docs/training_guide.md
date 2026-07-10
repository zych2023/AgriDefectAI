# 训练指南

## 环境要求

- Python 3.9+
- CUDA 11.8+ (GPU训练) 或 CPU
- 建议 GPU 显存 ≥ 8GB

## 数据集

AI Challenger 2018 — 农作物叶子图像数据集

- 10 种作物：苹果、樱桃、玉米、葡萄、柑桔、桃、辣椒、马铃薯、草莓、番茄
- 原始 61 个细粒度类别（区分一般/严重程度）
- 合并后 36 个分类类别
- 训练集 ~31,500 张，验证集 ~4,500 张

## 训练命令

```bash
# 在 model-training/ 目录下执行
python train.py
```

## 训练策略

1. **迁移学习**：加载 ImageNet 预训练 ResNet50 权重
2. **优化器**：AdamW (lr=3e-4, weight_decay=1e-4)
3. **学习率调度**：Cosine 衰减 + 前 2 epoch Warmup
4. **混合精度训练 (AMP)**：加速训练，减少显存占用
5. **Label Smoothing (0.05)**：缓解细粒度类别过拟合
6. **MixUp 数据增强**：前 20 epoch，增强泛化能力
7. **Early Stop (patience=15)**：验证集 loss 不降即停

## 在华为云 ModelArts 上训练

设置环境变量：

```bash
export D_MODELARTS=true
export D_DATA_URL="s3://your-bucket/data/"
export D_TRAIN_URL="s3://your-bucket/output/"
python train.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `checkpoints/best_model.pth` | 最佳 PyTorch 模型 (~271MB) |
| `checkpoints/final_model.pth` | 最终 PyTorch 模型 |
| `checkpoints/model.onnx` | ONNX 导出模型 (~91MB) |
| `checkpoints/feature_index.pt` | 相似检索特征库 (~261MB) |
