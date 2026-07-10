"""
模型评估脚本
- 混淆矩阵
- 每类准确率 / Precision / Recall / F1
- 可视化 top-k 误判案例
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)

from Data.dataset import get_dataloaders
from models.model import CropDiseaseClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model_path="checkpoints/best_model.pth"):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据集
    _, val_loader, _ = get_dataloaders(cfg)

    # 加载模型
    model = CropDiseaseClassifier(
        num_classes=cfg["data"]["num_classes"],
        pretrained=False,  # 用已训练的权重，不需要pretrained
        dropout=cfg["model"]["dropout"],
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[OK] 模型已加载: {model_path}")
    print(f"   训练轮次: {checkpoint.get('epoch', 'N/A')+1}")
    print(f"   验证集Acc@1: {checkpoint.get('val_acc', 'N/A'):.2f}%")

    # 全量预测
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="评估中"):
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 总体指标
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    print(f"\n{'='*60}")
    print(f"  总体准确率 (Overall Accuracy): {acc*100:.2f}%")
    print(f"  加权精度 (Weighted Precision):  {precision*100:.2f}%")
    print(f"  加权召回 (Weighted Recall):     {recall*100:.2f}%")
    print(f"  加权 F1   (Weighted F1):        {f1*100:.2f}%")
    print(f"{'='*60}")

    # 每类详细报告
    class_names_list = [cfg["class_names"].get(i, f"class_{i}") for i in range(cfg["data"]["num_classes"])]
    clf_report = classification_report(
        all_labels, all_preds,
        target_names=class_names_list,
        digits=3,
        zero_division=0,
    )
    print(f"\n[Report] 分类报告:\n{clf_report}")

    # 混淆矩阵 Top-k 误判
    cm = confusion_matrix(all_labels, all_preds)
    np.fill_diagonal(cm, 0)  # 排除正确分类
    # 找最易混淆的10对
    top_indices = np.argsort(cm.ravel())[-10:][::-1]
    rows, cols = np.unravel_index(top_indices, cm.shape)

    print("[Confusion] 最易混淆的10对类别:")
    for r, c in zip(rows, cols):
        actual = cfg["class_names"].get(r, f"class_{r}")
        predicted = cfg["class_names"].get(c, f"class_{c}")
        print(f"  {actual} → 误判为 {predicted} (次数: {cm[r,c]})")

    return acc, all_preds, all_labels


if __name__ == "__main__":
    evaluate()
